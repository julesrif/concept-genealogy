"""
agents/elsevier_client.py - Elsevier ScienceDirect full-text retrieval.

PURPOSE:
Fallback content source for papers without open-access PDFs. Uses the Elsevier
Article Retrieval API to fetch full-text XML/plain-text content by DOI.

ACADEMIC LICENSE CONSTRAINTS:
  - Data mining permitted under institutional/academic API keys
  - No storage of downloaded PDFs (text is processed in-memory only)
  - Rate limit: 2 requests/second for non-institutional, higher for institutional

API DOCUMENTATION:
  https://dev.elsevier.com/documentation/ArticleRetrievalAPI.wadl

FALLBACK CHAIN (CitationExtractor):
  1. Open-access PDF (Semantic Scholar openAccessPdf)
  2. Elsevier full-text via DOI (this module)
  3. Abstract-only (low confidence)
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class ElsevierClient:
    """Retrieves full-text article content from Elsevier's ScienceDirect API."""

    BASE_URL = "https://api.elsevier.com/content/article/doi"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_full_text(self, doi: str) -> Optional[str]:
        """
        Fetch full-text plain-text content for a paper by DOI.

        Returns the article body text, or None if not accessible.
        Tries text/plain first (direct extraction), then falls back to
        application/xml for structured content parsing.
        """
        if not self.api_key or not doi:
            return None

        # Normalise DOI: lowercase for consistent API behaviour
        doi = doi.strip().lower()

        headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "text/plain",
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{self.BASE_URL}/{doi}",
                    headers=headers,
                )

            if response.status_code == 200:
                text = response.text
                if text and len(text.strip()) > 100:
                    logger.info(
                        "Elsevier: retrieved %d chars for DOI %s", len(text), doi
                    )
                    return text
                else:
                    logger.debug("Elsevier: response too short for DOI %s", doi)
                    return None

            elif response.status_code == 400:
                # Some DOIs don't support text/plain — try XML fallback
                logger.debug("Elsevier: text/plain rejected for DOI %s, trying XML", doi)
                return self._try_xml_fallback(doi)

            elif response.status_code == 401:
                logger.warning("Elsevier: unauthorized (invalid API key) for DOI %s", doi)
                return None
            elif response.status_code == 403:
                logger.debug(
                    "Elsevier: access denied for DOI %s (not covered by subscription)", doi
                )
                return None
            elif response.status_code == 404:
                logger.debug("Elsevier: DOI %s not found in ScienceDirect", doi)
                return None
            elif response.status_code == 429:
                logger.warning("Elsevier: rate limited for DOI %s", doi)
                return None
            else:
                logger.debug(
                    "Elsevier: unexpected status %d for DOI %s",
                    response.status_code, doi
                )
                return None

        except httpx.RequestError as e:
            logger.warning("Elsevier: connection error for DOI %s: %s", doi, e)
            return None

    def _try_xml_fallback(self, doi: str) -> Optional[str]:
        """
        Fallback: request XML and extract body text by stripping tags.

        Some Elsevier articles don't support text/plain but do serve XML.
        """
        import re

        headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/xml",
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{self.BASE_URL}/{doi}",
                    headers=headers,
                )
            if response.status_code == 200:
                # Strip XML tags to get plain text
                text = re.sub(r"<[^>]+>", " ", response.text)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) > 100:
                    logger.info(
                        "Elsevier: XML fallback retrieved %d chars for DOI %s",
                        len(text), doi
                    )
                    return text
            logger.debug("Elsevier: XML fallback also failed for DOI %s (status %d)", doi, response.status_code)
            return None
        except httpx.RequestError as e:
            logger.debug("Elsevier: XML fallback connection error for DOI %s: %s", doi, e)
            return None
