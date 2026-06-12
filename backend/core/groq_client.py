"""
core/groq_client.py - Rate-limited Groq API client.

Groq free-tier rate limits (as of 2025):
  - llama-3.1-8b-instant:  30 req/min, 14,400 tokens/min
  - llama-3.3-70b-versatile: 30 req/min, 6,000 tokens/min

This module provides a shared, rate-limited wrapper that all agents use
to avoid 429 responses. It enforces a minimum inter-request delay and
retries with exponential backoff when 429s are received.
"""

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Groq free tier: 30 req/min = 1 request every 2 seconds (with safety margin)
_MIN_REQUEST_INTERVAL = 2.0  # seconds between requests
_MAX_RETRIES = 5
_INITIAL_BACKOFF = 5.0  # seconds


class RateLimitedGroqClient:
    """
    Thread-safe, rate-limited wrapper around the Groq SDK client.

    All agents share a single instance of this class (via config injection)
    to ensure global rate-limit compliance across the entire pipeline.
    """

    def __init__(self, api_key: str, progress_callback=None):
        from groq import Groq
        self._client = Groq(api_key=api_key)
        self._lock = threading.Lock()
        self._last_request_time = 0.0
        self._progress = progress_callback or (lambda msg, level="INFO": None)

    def _wait_for_rate_limit(self) -> None:
        """Block until the minimum inter-request interval has elapsed."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < _MIN_REQUEST_INTERVAL:
                sleep_time = _MIN_REQUEST_INTERVAL - elapsed
                logger.debug("Rate limiter: sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)
            self._last_request_time = time.time()

    def chat_completion(self, **kwargs: Any) -> Any:
        """
        Call Groq chat completions with automatic rate limiting and retry.

        Accepts the same kwargs as groq.chat.completions.create().
        Retries on 429 with exponential backoff.
        """
        for attempt in range(_MAX_RETRIES):
            self._wait_for_rate_limit()
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response
            except Exception as e:
                error_str = str(e)
                status_code = getattr(e, 'status_code', None)

                if status_code == 429 or '429' in error_str:
                    backoff = _INITIAL_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "Groq 429 rate limit hit (attempt %d/%d). "
                        "Backing off %.1fs before retry.",
                        attempt + 1, _MAX_RETRIES, backoff
                    )
                    self._progress(
                        f"Rate limit reached, waiting {int(backoff)}s before continuing..."
                    )
                    time.sleep(backoff)
                    continue
                else:
                    # Non-rate-limit error: re-raise immediately
                    raise

        # All retries exhausted
        raise RuntimeError(
            f"Groq API: rate limit exceeded after {_MAX_RETRIES} retries. "
            "Consider upgrading from the free tier or reducing pipeline concurrency."
        )
