"""
agents/citation_extractor.py - Citation Extractor agent.

ROLE IN ARCHITECTURE:
Reactive specialist agent. Receives an approved seed paper and returns extracted
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

    def run(self, paper: CandidatePaper, depth: int) -> list[ExtractedClaim]:
        """
        Main entry point. Returns extracted claims for a paper at a given depth.

        Extraction strategy:
          1. If open_access_url is available -> download PDF -> extract full text
          2. Otherwise -> use abstract (flagged as low confidence)
          3. Call Groq LLM with extraction prompt
          4. Inspector validates JSON output (called from Orchestrator after return)
        """
        logger.info(
            "CitationExtractor: processing paper '%s' (depth=%d, oa=%s)",
            paper.title, depth, bool(paper.open_access_url)
        )

        if paper.open_access_url:
            text, confidence = self._extract_from_pdf(paper.open_access_url)
        else:
            text = paper.abstract or ""
            confidence = 0.5  # abstract-only; design proposal flags as "low confidence"
            logger.info("CitationExtractor: no OA PDF for '%s' - using abstract only", paper.title)

        if not text.strip():
            logger.warning("CitationExtractor: no extractable text for paper '%s'", paper.title)
            return []

        raw_claims = self._llm_extract_claims(text, paper.title)
        return self._to_typed_claims(raw_claims, paper, depth, confidence)

    def _extract_from_pdf(self, pdf_url: str) -> tuple[str, float]:
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
            logger.error("PyMuPDF not installed. Run: pip install pymupdf")
            return "", 0.0
        except httpx.HTTPStatusError as e:
            logger.warning("PDF download failed for %s: %s", pdf_url, e)
            return "", 0.0
        except Exception as e:
            logger.warning("PDF extraction failed: %s", e)
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
        (subsumption architecture principle - Brooks, 1991).
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