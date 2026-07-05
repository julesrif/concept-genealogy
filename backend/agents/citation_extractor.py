"""
agents/citation_extractor.py - Citation Extractor agent.

ROLE IN ARCHITECTURE:
Stateless worker node. Receives an approved seed paper and returns extracted
inline citations, reference chains, and raw claims. No access to global state.

RESPONSIBILITIES:
  1. Download the open-access PDF via Semantic Scholar URL (if available)
  2. Extract text from PDF using PyMuPDF (Artifex Software, Inc., 2026)
  3. Prompt Groq LLM to extract structured claims and inline citations
  4. Fallback to abstract-only extraction if PDF is unavailable

WHY PYMUPDF OVER PDFPLUMBER OR PYPDF:
PyMuPDF (fitz) provides the best text extraction fidelity for academic PDFs,
particularly for multi-column layouts and embedded mathematical notation common
in CS/AI papers. It also handles scanned PDFs via its OCR integration, though
that feature is not used in the MVP sprint (requires Tesseract installation).
Source: Artifex Software, Inc. (2026) PyMuPDF. GitHub.

WHY LLM-BASED CLAIM EXTRACTION (NOT REGEX/NLP PIPELINE):
Academic claims are expressed in highly varied natural language. Regex-based
extraction would require extensive domain-specific pattern engineering and would
still fail on paraphrased claims. LLM extraction generalises across paper styles,
domains, and argumentation structures - at the cost of non-determinism, which the
Inspector's JSON validation layer (Inspector.validate_extracted_claims_json) handles.

FALLBACK STRATEGY (abstract-only):
When openAccessPdf is None (closed-access paper), we extract claims from the
abstract only. This is flagged in the ExtractedClaim.confidence field (0.5 vs 1.0)
so the Evaluator can weight abstract-extracted claims differently.
The design proposal (Section 5) documents this: "Extract abstract only - Flag low confidence."

References:
  Artifex Software, Inc. (2026) PyMuPDF. GitHub. https://github.com/pymupdf/pymupdf
  Atıl et al. (2025) Non-determinism of "deterministic" LLM system settings. ACL 2025.
  Maynez et al. (2020) On faithfulness and factuality. ACL 2020, pp. 1906-1919.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import httpx

from core.state import CandidatePaper, ExtractedClaim

logger = logging.getLogger(__name__)


# Prompt template for claim extraction.
# WHY EXPLICIT JSON SCHEMA IN PROMPT:
# Providing the exact JSON schema in the prompt significantly reduces structural
# validation failures (Cemri et al., 2025). The Inspector still validates output,
# but a well-specified prompt reduces retry frequency and token waste.
EXTRACTION_PROMPT = """You are a scientific claim extractor. Extract the key claims from the following academic paper text.

Return ONLY a JSON array with no preamble, no code fences, no explanation.
Each element must have exactly these fields:
  - "claim_text": string - one complete, standalone claim from the paper
  - "inline_citations": array of strings - DOIs or author-year references cited within this claim
  - "confidence": number between 0.0 and 1.0

Extract 5-10 of the most significant claims. Focus on claims that introduce, support, or challenge a core concept.

Paper title: {title}
Paper text:
{text}

JSON array:"""


class CitationExtractorAgent:
    """
    Extracts structured claims from academic paper PDFs or abstracts.

    Stateless: receives a CandidatePaper, returns a list of ExtractedClaim.
    The Inspector validates the LLM output before it is returned.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.groq = config.get("_groq_client")  # shared rate-limited client
        self.model = config.get("groq_model_fast", "llama-3.1-8b-instant")
        self.max_text_chars = 8000  # ~2000 tokens; keeps within context window safely
        self.abstract_confidence = float(config.get("abstract_confidence", 0.5))
        self._init_elsevier(config)

    def _init_elsevier(self, config: dict[str, Any]) -> None:
        """Initialize Elsevier client if API key is available."""
        from agents.elsevier_client import ElsevierClient
        elsevier_key = config.get("elsevier_api_key", "")
        self.elsevier = ElsevierClient(elsevier_key) if elsevier_key else None

    def run(self, paper: CandidatePaper, depth: int) -> list[ExtractedClaim]:
        """
        Main entry point. Returns extracted claims for a paper at a given depth.

        Extraction strategy (priority-ordered fallback chain):
          1. Open Access PDF (most reliable - no anti-bot, no API key needed)
          2. Elsevier Article Retrieval API (reliable with API key, covers 10.1016/ DOIs)
          3. Direct DOI PDF download (least reliable - often blocked by captchas/anti-bot)
          4. Abstract-only (last resort, flagged as low confidence)
          5. Call Groq LLM with extraction prompt
          6. Inspector validates JSON output (called from Orchestrator after return)

        For Elsevier DOIs (10.1016/): Elsevier API is attempted before OA PDF because
        Semantic Scholar's openAccessPdf URLs for Elsevier papers often point to
        ScienceDirect landing pages that serve captchas instead of actual PDFs.
        """
        logger.info(
            "CitationExtractor: processing paper '%s' (depth=%d, oa=%s, doi=%s)",
            paper.title, depth, bool(paper.open_access_url), paper.doi or "none"
        )

        text = ""
        confidence = self.abstract_confidence
        is_elsevier_doi = paper.doi and paper.doi.startswith("10.1016/")

        # For Elsevier DOIs: try Elsevier API first (more reliable than OA/DOI download)
        if is_elsevier_doi and self.elsevier:
            logger.info(
                "CitationExtractor: Elsevier DOI detected ('%s') - trying API first",
                paper.doi,
            )
            elsevier_text = self.elsevier.get_full_text(paper.doi)
            if elsevier_text:
                text = elsevier_text[:self.max_text_chars]
                confidence = 1.0
                logger.info(
                    "CitationExtractor: Elsevier API provided %d chars for '%s'",
                    len(text), paper.title,
                )

        # Strategy 1: Open-access PDF (skip if Elsevier already succeeded)
        if not text.strip() and paper.open_access_url:
            text, confidence = self._extract_from_pdf(paper.open_access_url, paper.title)

        # Strategy 2: Elsevier full-text via DOI (for non-10.1016 DOIs that Elsevier might still cover)
        if not text.strip() and paper.doi and self.elsevier and not is_elsevier_doi:
            logger.info("CitationExtractor: trying Elsevier API for DOI '%s'", paper.doi)
            elsevier_text = self.elsevier.get_full_text(paper.doi)
            if elsevier_text:
                text = elsevier_text[:self.max_text_chars]
                confidence = 1.0
                logger.info(
                    "CitationExtractor: Elsevier API provided %d chars for '%s'",
                    len(text), paper.title,
                )

        # Strategy 3: Direct DOI PDF download (least reliable - captchas/anti-bot)
        if not text.strip() and paper.doi:
            logger.info(
                "CitationExtractor: trying direct DOI PDF for '%s' (least reliable)",
                paper.title,
            )
            text, confidence = self._extract_from_doi_pdf(paper.doi, paper.title)

        # Strategy 4: Abstract-only fallback
        if not text.strip():
            text = paper.abstract or ""
            confidence = self.abstract_confidence
            logger.info(
                "CitationExtractor: no full text for '%s' - using abstract only "
                "(confidence=%.2f). Full text unavailable via OA PDF, Elsevier API, or DOI.",
                paper.title, confidence,
            )

        if not text.strip():
            logger.warning("CitationExtractor: no extractable text for paper '%s'", paper.title)
            return []

        raw_claims = self._llm_extract_claims(text, paper.title)
        return self._to_typed_claims(raw_claims, paper, depth, confidence)

    def _extract_from_pdf(self, pdf_url: str, title: str = "") -> tuple[str, float]:
        """
        Download PDF and extract text using PyMuPDF.

        Returns (text, confidence) where confidence=1.0 for full PDF extraction.

        WHY TEMPFILE (NOT IN-MEMORY):
        PyMuPDF requires a file path or bytes buffer. Using a tempfile ensures the
        PDF is not kept in memory after extraction - relevant when tracing deep
        citation chains where many PDFs may be processed sequentially.
        """
        try:
            import fitz  # PyMuPDF

            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                response = client.get(pdf_url)
                response.raise_for_status()
                pdf_bytes = response.content

            # Write to tempfile and extract
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            doc = fitz.open(tmp_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            Path(tmp_path).unlink(missing_ok=True)

            full_text = "\n".join(text_parts)
            # Truncate to max_text_chars to stay within LLM context window
            truncated = full_text[:self.max_text_chars]
            if len(full_text) > self.max_text_chars:
                logger.debug(
                    "CitationExtractor: truncated PDF text from %d to %d chars",
                    len(full_text), self.max_text_chars
                )

            return truncated, 1.0

        except ImportError:
            logger.error(
                "Unable to parse PDF for '%s': PyMuPDF not installed. "
                "Continuing with remaining extraction strategies. "
                "Run: pip install pymupdf",
                title,
            )
            return "", 0.0
        except httpx.HTTPStatusError as e:
            logger.warning(
                "Unable to parse PDF for '%s': HTTP %d from %s. "
                "Continuing with remaining extraction strategies.",
                title, e.response.status_code, pdf_url,
            )
            return "", 0.0
        except Exception as e:
            logger.warning(
                "Unable to parse PDF for '%s': %s. "
                "Continuing with remaining extraction strategies.",
                title, e,
            )
            return "", 0.0

    def _extract_from_doi_pdf(self, doi: str, title: str = "") -> tuple[str, float]:
        """
        Attempt to download a PDF by following the DOI redirect.

        Many publishers (Annual Reviews, PNAS, etc.) serve free PDFs that are
        not indexed in Semantic Scholar's openAccessPdf field. This follows the
        DOI URL, checks if the final response is a PDF, and extracts text.

        NOTE: This is the LEAST reliable strategy. Publisher sites frequently
        block automated requests with captchas, anti-bot systems, or redirects
        to login pages. Elsevier (10.1016/) and Springer (10.1007/) are notorious.
        """
        if doi.startswith("10.1016/"):
            logger.debug(
                "CitationExtractor: direct DOI download for Elsevier DOI %s "
                "(likely blocked by anti-bot - Elsevier API is preferred)",
                doi,
            )
        try:
            import fitz  # PyMuPDF

            doi_url = f"https://doi.org/{doi}"
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                # First try: see if the DOI resolves to a page with a PDF link
                response = client.get(doi_url, headers={
                    "Accept": "application/pdf",
                    "User-Agent": "ConceptGenealogy/1.0 (academic research)",
                })

                if response.status_code != 200:
                    return "", 0.0

                content_type = response.headers.get("content-type", "")
                if "application/pdf" not in content_type:
                    logger.debug(
                        "CitationExtractor: DOI %s did not return PDF (content-type: %s)",
                        doi, content_type
                    )
                    return "", 0.0

                pdf_bytes = response.content

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            doc = fitz.open(tmp_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            Path(tmp_path).unlink(missing_ok=True)

            full_text = "\n".join(text_parts)
            if len(full_text.strip()) < 200:
                logger.debug("CitationExtractor: DOI PDF text too short for '%s'", title)
                return "", 0.0

            truncated = full_text[:self.max_text_chars]
            logger.info(
                "CitationExtractor: DOI PDF provided %d chars for '%s'",
                len(truncated), title
            )
            return truncated, 1.0

        except ImportError:
            return "", 0.0
        except Exception as e:
            logger.debug("CitationExtractor: DOI PDF failed for '%s': %s", title, e)
            return "", 0.0

    def _llm_extract_claims(self, text: str, title: str) -> list[dict]:
        """
        Call Groq LLM to extract structured claims from paper text.

        Temperature=0 is set for maximum consistency, though Atıl et al. (2025)
        demonstrate that even zero-temperature settings exhibit non-determinism
        in hosted environments - hence the Inspector's validation layer remains
        necessary regardless of this setting.
        """
        if not self.groq:
            logger.error("CitationExtractor: Groq client not available")
            return []

        prompt = EXTRACTION_PROMPT.format(
            title=title,
            text=text
        )

        try:
            response = self.groq.chat_completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2000,
            )
            raw_output = response.choices[0].message.content

            # Basic JSON cleaning - Inspector does the full validation
            from agents.inspector import InspectorAgent
            return InspectorAgent.validate_extracted_claims_json(
                InspectorAgent,  # call staticmethod workaround
                raw_output,
                source_paper_id="unknown",
                depth=0,
            )

        except Exception as e:
            logger.error("CitationExtractor: Groq API call failed: %s", e)
            return []

    @staticmethod
    def _to_typed_claims(
        raw_claims: list[dict],
        paper: CandidatePaper,
        depth: int,
        confidence: float,
    ) -> list[ExtractedClaim]:
        """
        Convert raw LLM output dicts to typed ExtractedClaim objects.

        WHY NOT PYDANTIC PARSE HERE:
        Pydantic validation of ExtractedClaim happens in the Inspector, not here.
        The Extractor is responsible for producing output; validation is the
        Inspector's domain. This separation keeps each agent's scope minimal
        (stateless worker principle - single-responsibility design).
        """
        claims: list[ExtractedClaim] = []
        for raw in raw_claims:
            if not isinstance(raw, dict):
                continue
            try:
                claim = ExtractedClaim(
                    claim_text=raw.get("claim_text", ""),
                    source_paper_id=paper.paper_id,
                    source_title=paper.title,
                    depth=depth,
                    inline_citations=raw.get("inline_citations", []),
                    confidence=float(raw.get("confidence", confidence)),
                )
                if claim.claim_text.strip():
                    claims.append(claim)
            except Exception as e:
                logger.warning("CitationExtractor: skipping malformed claim: %s", e)
                continue

        logger.info(
            "CitationExtractor: extracted %d valid claims from '%s'",
            len(claims), paper.title
        )
        return claims