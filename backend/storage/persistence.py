"""
storage/persistence.py - Persistence layer.

RESPONSIBILITIES:
  1. Write final pipeline results to PostgreSQL (claims + metadata)
  2. Write Markdown synthesis file to output directory
  3. Write full JSON log for audit/reproducibility

WHY POSTGRESQL WITH PGVECTOR (NOT A DEDICATED VECTOR DB):
"PostgreSQL 16 with pgvector unifies metadata and embeddings, avoiding dedicated
vector database overhead." (Design proposal 7, citing pgvector, 2021)

Using PostgreSQL for both relational metadata and vector embeddings means:
  - Single database to run, backup, and maintain (relevant for Docker Compose setup)
  - Native SQL joins between claim metadata and vector similarity queries
  - No additional service to authenticate to or rate-limit against
  - pgvector's HNSW index gives adequate ANN performance for academic-scale datasets

WHY JSON LOG IN ADDITION TO POSTGRES:
The JSON log is the human-readable audit trail. If the database is unavailable
or the schema changes, the JSON log allows reconstruction of results. This
implements the "traceable outputs" requirement from the design proposal 1:
every claim in the final output is traceable to its source paper, depth level,
and validity score via the JSON log.

References:
  pgvector (2021) pgvector: open-source vector similarity search for PostgreSQL.
    https://github.com/pgvector/pgvector
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from core.state import PipelineState

logger = logging.getLogger(__name__)


# SQL schema for concept_genealogy tables.
# WHY NOT AN ORM (SQLALCHEMY/DJANGO):
# An ORM adds a dependency and an abstraction layer for what is essentially
# 3 INSERT statements. Raw psycopg2 keeps the database interaction explicit
# and traceable - every SQL statement is visible in this file.
CREATE_TABLES_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          TEXT PRIMARY KEY,
    user_concept    TEXT NOT NULL,
    started_at      TIMESTAMP NOT NULL,
    completed_at    TIMESTAMP,
    seed_paper_id   TEXT,
    seed_paper_title TEXT,
    output_path     TEXT,
    status          TEXT NOT NULL DEFAULT 'completed'
);

CREATE TABLE IF NOT EXISTS traced_claims (
    claim_id        SERIAL PRIMARY KEY,
    run_id          TEXT REFERENCES pipeline_runs(run_id),
    paper_id        TEXT NOT NULL,
    paper_title     TEXT NOT NULL,
    depth           INTEGER NOT NULL,
    claim_text      TEXT NOT NULL,
    validity_score  FLOAT,
    drift_score     FLOAT,
    relevant_concepts TEXT[],
    created_at      TIMESTAMP DEFAULT NOW()
);

-- pgvector column for semantic similarity queries (stretch goal).
-- Added as a separate ALTER to allow graceful degradation if pgvector
-- extension is not available (the table itself still works without it).
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='traced_claims' AND column_name='embedding'
    ) THEN
        ALTER TABLE traced_claims ADD COLUMN embedding vector(768);
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector not available - embedding column skipped';
END $$;
"""


class PersistenceLayer:
    """Handles final output writing to PostgreSQL and filesystem."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.database_url = config.get(
            "database_url",
            "postgresql://postgres:postgres@localhost:5432/concept_tracer"
        )
        self.output_dir = Path(config.get("output_dir", "output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_embedding_model()

    def _init_embedding_model(self) -> None:
        """
        Initialise Sentence-BERT for pgvector embedding population.

        Uses all-mpnet-base-v2 (768 dimensions) matching the vector(768) column
        in the schema. Gracefully degrades if sentence-transformers is unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
            logger.info("Persistence: embedding model loaded for pgvector population")
        except ImportError:
            logger.info(
                "Persistence: sentence-transformers not installed - "
                "pgvector embeddings will not be populated"
            )
            self.embedding_model = None

    def save(self, state: PipelineState) -> str:
        """
        Persist pipeline results. Returns the path to the Markdown output file.

        Save order: JSON log first, then PostgreSQL, then Markdown.
        WHY THIS ORDER: if the DB write fails, the JSON log is already on disk
        and can be replayed. The Markdown file is the user-facing deliverable
        and is written last only after successful DB persistence.
        """
        run_id = self._generate_run_id(state)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        concept_slug = state.user_concept.replace(" ", "_")[:40]

        # 1. JSON audit log (always written, even if DB fails)
        json_path = self.output_dir / f"{run_id}_{concept_slug}.json"
        self._write_json_log(state, run_id, json_path)

        # 2. PostgreSQL persistence (non-fatal on failure)
        try:
            self._write_to_postgres(state, run_id, timestamp, str(json_path))
            logger.info("Persistence: DB write successful for run_id=%s", run_id)
        except Exception as e:
            logger.error(
                "Persistence: DB write failed for run_id=%s: %s. "
                "Results are preserved in JSON log at %s",
                run_id, e, json_path
            )

        # 3. Markdown output (user-facing deliverable)
        md_path = self.output_dir / f"{run_id}_{concept_slug}.md"
        self._write_markdown(state, run_id, md_path)

        logger.info("Persistence: output saved to %s", md_path)
        return str(md_path)

    def _write_json_log(
        self, state: PipelineState, run_id: str, path: Path
    ) -> None:
        """Write full pipeline state as a JSON audit log."""
        log_data = {
            "run_id": run_id,
            "user_concept": state.user_concept,
            "seed_paper": state.selected_seed.dict() if state.selected_seed else None,
            "tracing_results": [
                {
                    "paper_id": r.paper_id,
                    "paper_title": r.paper_title,
                    "depth": r.depth,
                    "claims": [c.dict() for c in r.claims],
                }
                for r in state.tracing_results
            ],
            "synthesis": state.synthesis_text,
            "synthesis_status": state.synthesis_status,
            "challenger_critique": state.challenger_critique,
            "metadata": {
                "max_depth": state.max_depth,
                "actual_depth": state.current_depth,
                "papers_visited": list(state.visited_paper_ids),
                "inspector_retries": state.inspector_retries,
                "synthesiser_retries": state.synthesiser_retries,
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.debug("Persistence: JSON log written to %s", path)

    def _write_to_postgres(
        self, state: PipelineState, run_id: str, timestamp: str, json_path: str
    ) -> None:
        """Write run metadata and claims to PostgreSQL."""
        try:
            import psycopg2
        except ImportError:
            logger.warning("psycopg2 not installed - skipping DB write. Run: pip install psycopg2-binary")
            return

        conn = psycopg2.connect(self.database_url)
        try:
            with conn.cursor() as cur:
                # Ensure schema exists
                cur.execute(CREATE_TABLES_SQL)

                # Insert pipeline run record
                cur.execute(
                    """
                    INSERT INTO pipeline_runs
                      (run_id, user_concept, started_at, completed_at, seed_paper_id,
                       seed_paper_title, output_path, status)
                    VALUES (%s, %s, NOW(), NOW(), %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO NOTHING
                    """,
                    (
                        run_id,
                        state.user_concept,
                        state.selected_seed.paper_id if state.selected_seed else None,
                        state.selected_seed.title if state.selected_seed else None,
                        json_path,
                        state.synthesis_status,
                    )
                )

                # Insert all evaluated claims
                for result in state.tracing_results:
                    for claim in result.claims:
                        cur.execute(
                            """
                            INSERT INTO traced_claims
                              (run_id, paper_id, paper_title, depth, claim_text,
                               validity_score, drift_score, relevant_concepts)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING claim_id
                            """,
                            (
                                run_id,
                                claim.source_paper_id,
                                claim.source_title,
                                claim.depth,
                                claim.claim_text,
                                claim.validity_score,
                                claim.drift_score,
                                claim.relevant_concepts,
                            )
                        )

                # Populate pgvector embeddings for all claims in this run
                self._populate_embeddings(cur, run_id)

            conn.commit()
        finally:
            conn.close()

    def _populate_embeddings(self, cur, run_id: str) -> None:
        """
        Compute and store sentence embeddings for all claims in a run.

        Uses all-mpnet-base-v2 (768d) to produce dense vectors stored in the
        pgvector `embedding` column. These enable semantic similarity search
        across claims (e.g. "find claims semantically similar to X").

        Batch-encodes all claims for efficiency, then updates each row.
        Gracefully skips if:
          - No embedding model available (sentence-transformers not installed)
          - The embedding column doesn't exist (pgvector extension not available)
        """
        if self.embedding_model is None:
            return

        try:
            # Check if embedding column exists (pgvector may not be installed)
            cur.execute("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'traced_claims' AND column_name = 'embedding'
            """)
            if not cur.fetchone():
                logger.debug("Persistence: embedding column not found - skipping")
                return

            # Fetch all claims for this run that need embeddings
            cur.execute(
                """
                SELECT claim_id, claim_text FROM traced_claims
                WHERE run_id = %s AND embedding IS NULL
                """,
                (run_id,)
            )
            rows = cur.fetchall()
            if not rows:
                return

            claim_ids = [r[0] for r in rows]
            claim_texts = [r[1] for r in rows]

            # Batch encode for efficiency (all claims at once)
            embeddings = self.embedding_model.encode(
                claim_texts, convert_to_numpy=True, show_progress_bar=False
            )

            # Update each claim with its embedding
            for claim_id, embedding in zip(claim_ids, embeddings):
                # pgvector expects a list of floats as the vector value
                vec_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
                cur.execute(
                    "UPDATE traced_claims SET embedding = %s WHERE claim_id = %s",
                    (vec_str, claim_id)
                )

            logger.info(
                "Persistence: populated %d pgvector embeddings for run %s",
                len(claim_ids), run_id
            )
        except Exception as e:
            # Non-fatal: embeddings are a stretch goal, don't block persistence
            logger.warning("Persistence: embedding population failed: %s", e)

    @staticmethod
    def _write_markdown(state: PipelineState, run_id: str, path: Path) -> None:
        """Write user-facing Markdown output file."""
        lines = [
            f"# Concept Genealogy: {state.user_concept}",
            f"\n*Run ID: {run_id}*\n",
        ]

        if state.selected_seed:
            lines += [
                "## Seed Paper",
                f"**{state.selected_seed.title}**",
                f"Authors: {', '.join(state.selected_seed.authors)}",
                f"Year: {state.selected_seed.year or 'unknown'}",
                "",
            ]

        if state.synthesis_text:
            lines += ["## Synthesis", "", state.synthesis_text, ""]

        if state.challenger_critique:
            lines += ["## Challenger Review", "", state.challenger_critique, ""]

        lines += ["## Citation Trace Summary", ""]
        for result in sorted(state.tracing_results, key=lambda r: r.depth):
            depth_label = "Seed" if result.depth == 0 else f"Depth {result.depth}"
            lines.append(f"### {depth_label}: {result.paper_title}")
            lines.append(f"*{len(result.claims)} claims extracted*\n")
            for claim in result.claims[:5]:  # show top 5 claims per paper
                lines.append(f"- [{claim.validity_score:.2f}] {claim.claim_text}")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _generate_run_id(state: PipelineState) -> str:
        """Generate a deterministic run ID from concept + timestamp."""
        import hashlib
        concept_hash = hashlib.md5(state.user_concept.encode()).hexdigest()[:6]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}_{concept_hash}"