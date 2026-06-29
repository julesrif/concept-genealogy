"""
Unit tests for agents/elsevier_client.py - Elsevier ScienceDirect full-text retrieval.

Tests successful retrieval, various HTTP error codes, XML fallback, and edge cases.
Mocks httpx.Client to avoid real network calls.
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.elsevier_client import ElsevierClient


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    return ElsevierClient(api_key="test-elsevier-key")


@pytest.fixture
def mock_httpx_response():
    """Factory for creating mock httpx responses."""
    def _make(status_code=200, text="", raise_error=False):
        response = MagicMock()
        response.status_code = status_code
        response.text = text
        if raise_error:
            import httpx
            raise httpx.RequestError("Connection failed", request=MagicMock())
        return response
    return _make


# ── Successful retrieval ────────────────────────────────────────────────────

class TestSuccessfulRetrieval:
    """Test successful full-text retrieval."""

    def test_200_with_long_text_returns_content(self, client):
        """200 response with >100 chars returns the text."""
        long_text = "A" * 200  # >100 chars
        mock_response = MagicMock(status_code=200, text=long_text)
        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.get.return_value = mock_response

        with patch("httpx.Client", return_value=mock_http_client):
            result = client.get_full_text("10.1234/test")
        assert result == long_text

    def test_200_with_short_text_returns_none(self, client):
        """200 response with <=100 chars returns None."""
        short_text = "Short"
        mock_response = MagicMock(status_code=200, text=short_text)
        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.get.return_value = mock_response

        with patch("httpx.Client", return_value=mock_http_client):
            result = client.get_full_text("10.1234/test")
        assert result is None

    def test_doi_is_normalized_lowercase(self, client):
        """DOI is lowercased before request."""
        long_text = "B" * 200
        mock_response = MagicMock(status_code=200, text=long_text)
        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.get.return_value = mock_response

        with patch("httpx.Client", return_value=mock_http_client):
            client.get_full_text("10.1234/TEST-DOI")
        # Check the URL used in the call
        call_url = mock_http_client.get.call_args[0][0]
        assert "10.1234/test-doi" in call_url


# ── XML fallback ────────────────────────────────────────────────────────────

class TestXmlFallback:
    """Test 400 → XML fallback behavior."""

    def test_400_triggers_xml_fallback(self, client):
        """400 on text/plain request triggers XML fallback."""
        text_response = MagicMock(status_code=400, text="")
        xml_content = "<root><body>" + "X" * 200 + "</body></root>"
        xml_response = MagicMock(status_code=200, text=xml_content)

        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.get.side_effect = [text_response, xml_response]

        with patch("httpx.Client", return_value=mock_http_client):
            result = client.get_full_text("10.1234/test")
        assert result is not None
        assert len(result) > 100
        # XML tags should be stripped
        assert "<root>" not in result


# ── HTTP error codes ────────────────────────────────────────────────────────

class TestHttpErrors:
    """Test handling of various HTTP error responses."""

    @pytest.mark.parametrize("status_code", [401, 403, 404, 429])
    def test_error_codes_return_none(self, client, status_code):
        """Non-200 status codes return None."""
        mock_response = MagicMock(status_code=status_code, text="")
        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.get.return_value = mock_response

        with patch("httpx.Client", return_value=mock_http_client):
            result = client.get_full_text("10.1234/test")
        assert result is None

    def test_unexpected_status_returns_none(self, client):
        """Unknown status code (500) returns None."""
        mock_response = MagicMock(status_code=500, text="")
        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.get.return_value = mock_response

        with patch("httpx.Client", return_value=mock_http_client):
            result = client.get_full_text("10.1234/test")
        assert result is None


# ── Connection errors ───────────────────────────────────────────────────────

class TestConnectionErrors:
    """Test network error handling."""

    def test_request_error_returns_none(self, client):
        """httpx.RequestError → returns None gracefully."""
        import httpx

        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.get.side_effect = httpx.RequestError(
            "Connection refused", request=MagicMock()
        )

        with patch("httpx.Client", return_value=mock_http_client):
            result = client.get_full_text("10.1234/test")
        assert result is None


# ── Edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test boundary conditions."""

    def test_no_api_key_returns_none(self):
        """Empty API key → returns None immediately without making request."""
        client = ElsevierClient(api_key="")
        result = client.get_full_text("10.1234/test")
        assert result is None

    def test_no_doi_returns_none(self, client):
        """Empty DOI → returns None immediately."""
        result = client.get_full_text("")
        assert result is None

    def test_none_doi_returns_none(self, client):
        """None DOI → returns None immediately."""
        result = client.get_full_text(None)
        assert result is None

    def test_whitespace_doi_returns_none(self, client):
        """Whitespace-only DOI → returns None after strip."""
        result = client.get_full_text("   ")
        assert result is None
