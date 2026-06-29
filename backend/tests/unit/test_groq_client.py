"""
Unit tests for core/groq_client.py - Rate-limited Groq API client.

Tests initialization, rate limiting, 429 retry with backoff, and error propagation.
Patches groq.Groq to avoid real SDK dependency.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import time


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_groq_sdk():
    """Patch the groq SDK so we don't need it installed for tests."""
    mock_client_instance = MagicMock()
    mock_groq_class = MagicMock(return_value=mock_client_instance)
    with patch.dict("sys.modules", {"groq": MagicMock(Groq=mock_groq_class)}):
        yield mock_client_instance, mock_groq_class


@pytest.fixture
def groq_client(mock_groq_sdk):
    """Create a RateLimitedGroqClient with mocked SDK."""
    mock_client_instance, _ = mock_groq_sdk
    from core.groq_client import RateLimitedGroqClient
    client = RateLimitedGroqClient(api_key="test-key-123")
    # Replace the internal client with our mock
    client._client = mock_client_instance
    # Reset timing to avoid inter-test interference
    client._last_request_time = 0.0
    return client


# ── Initialization ──────────────────────────────────────────────────────────

class TestGroqClientInit:
    """Test RateLimitedGroqClient initialization."""

    def test_init_creates_client(self, mock_groq_sdk):
        _, mock_groq_class = mock_groq_sdk
        from core.groq_client import RateLimitedGroqClient
        client = RateLimitedGroqClient(api_key="my-api-key")
        mock_groq_class.assert_called_once_with(api_key="my-api-key")

    def test_init_with_progress_callback(self, mock_groq_sdk):
        from core.groq_client import RateLimitedGroqClient
        cb = MagicMock()
        client = RateLimitedGroqClient(api_key="key", progress_callback=cb)
        assert client._progress == cb


# ── Successful calls ────────────────────────────────────────────────────────

class TestChatCompletion:
    """Test successful chat completion calls."""

    def test_successful_call_returns_response(self, groq_client, mock_groq_sdk):
        mock_client_instance, _ = mock_groq_sdk
        expected_response = MagicMock()
        mock_client_instance.chat.completions.create.return_value = expected_response

        result = groq_client.chat_completion(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "test"}],
        )
        assert result == expected_response
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "test"}],
        )

    def test_passes_all_kwargs_to_sdk(self, groq_client, mock_groq_sdk):
        mock_client_instance, _ = mock_groq_sdk
        mock_client_instance.chat.completions.create.return_value = MagicMock()

        groq_client.chat_completion(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.3,
            max_tokens=500,
        )
        call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 500


# ── Rate limiting ───────────────────────────────────────────────────────────

class TestRateLimiting:
    """Test rate-limiting behavior."""

    def test_rate_limit_delays_rapid_calls(self, groq_client, mock_groq_sdk):
        """Two rapid calls should be spaced by _MIN_REQUEST_INTERVAL."""
        mock_client_instance, _ = mock_groq_sdk
        mock_client_instance.chat.completions.create.return_value = MagicMock()

        with patch("core.groq_client.time.sleep") as mock_sleep:
            # Set last request to "just now"
            groq_client._last_request_time = time.time()
            groq_client.chat_completion(model="m", messages=[])
            # sleep should have been called with a positive value
            mock_sleep.assert_called()
            sleep_time = mock_sleep.call_args[0][0]
            assert sleep_time > 0


# ── 429 Retry logic ─────────────────────────────────────────────────────────

class TestRetryOn429:
    """Test exponential backoff on 429 rate limit responses."""

    def test_retries_on_429_then_succeeds(self, groq_client, mock_groq_sdk):
        """First call gets 429, second succeeds."""
        mock_client_instance, _ = mock_groq_sdk
        error_429 = Exception("Rate limit exceeded")
        error_429.status_code = 429
        expected_response = MagicMock()

        mock_client_instance.chat.completions.create.side_effect = [
            error_429,
            expected_response,
        ]

        with patch("core.groq_client.time.sleep"):
            result = groq_client.chat_completion(model="m", messages=[])
        assert result == expected_response
        assert mock_client_instance.chat.completions.create.call_count == 2

    def test_429_in_error_string_also_retries(self, groq_client, mock_groq_sdk):
        """429 detected via string match when no status_code attribute."""
        mock_client_instance, _ = mock_groq_sdk
        error_429 = Exception("Error 429: too many requests")
        expected_response = MagicMock()

        mock_client_instance.chat.completions.create.side_effect = [
            error_429,
            expected_response,
        ]

        with patch("core.groq_client.time.sleep"):
            result = groq_client.chat_completion(model="m", messages=[])
        assert result == expected_response

    def test_exhausts_retries_raises_runtime_error(self, groq_client, mock_groq_sdk):
        """All retries exhausted → RuntimeError."""
        mock_client_instance, _ = mock_groq_sdk
        error_429 = Exception("Rate limit")
        error_429.status_code = 429

        mock_client_instance.chat.completions.create.side_effect = error_429

        with patch("core.groq_client.time.sleep"):
            with pytest.raises(RuntimeError, match="rate limit exceeded"):
                groq_client.chat_completion(model="m", messages=[])

        # Should have tried _MAX_RETRIES times
        from core.groq_client import _MAX_RETRIES
        assert mock_client_instance.chat.completions.create.call_count == _MAX_RETRIES


# ── Non-rate-limit errors ───────────────────────────────────────────────────

class TestNonRateLimitErrors:
    """Test that non-429 errors are raised immediately without retry."""

    def test_auth_error_raises_immediately(self, groq_client, mock_groq_sdk):
        mock_client_instance, _ = mock_groq_sdk
        error = Exception("Invalid API key")
        error.status_code = 401

        mock_client_instance.chat.completions.create.side_effect = error

        with pytest.raises(Exception, match="Invalid API key"):
            groq_client.chat_completion(model="m", messages=[])
        # Only one attempt - no retry
        assert mock_client_instance.chat.completions.create.call_count == 1

    def test_generic_exception_raises_immediately(self, groq_client, mock_groq_sdk):
        mock_client_instance, _ = mock_groq_sdk
        mock_client_instance.chat.completions.create.side_effect = ValueError("Bad input")

        with pytest.raises(ValueError, match="Bad input"):
            groq_client.chat_completion(model="m", messages=[])
        assert mock_client_instance.chat.completions.create.call_count == 1
