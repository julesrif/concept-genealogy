"""
main.py - Entry point for the Concept Genealogy pipeline.

Usage:
    python main.py "attention mechanism"
    python main.py "transformer architecture" --depth 2
    python main.py "reinforcement learning from human feedback" --depth 3

The pipeline runs interactively - it will pause at three HITL gates and
wait for user input. See agents/hitl.py for the gate implementations.
"""

import argparse
import logging
import sys
import warnings

from core.config import load_config
from core.orchestrator import Orchestrator

def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("log"),
        ]
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = argparse.ArgumentParser(
        description="Concept Genealogy - LLM-powered academic concept genealogy"
    )
    parser.add_argument(
        "concept",
        type=str,
        help="The academic concept to trace (e.g. 'attention mechanism')"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Maximum citation depth to trace (overrides .env MAX_DEPTH)"
    )
    args = parser.parse_args()

    # Load and validate config - fails fast on missing GROQ_API_KEY
    config = load_config()
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)

    # Apply CLI depth override if provided
    config_dict = config.dict()
    if args.depth is not None:
        config_dict["max_depth"] = args.depth

    logger.info("Starting Concept Genealogy for concept: '%s'", args.concept)
    logger.info("Max depth: %d | Model (fast): %s | Model (reasoning): %s",
                config_dict["max_depth"],
                config_dict["groq_model_fast"],
                config_dict["groq_model_reasoning"])

    orchestrator = Orchestrator(config_dict)
    final_state = orchestrator.run(args.concept)

    if final_state.stage.value == "done":
        print(f"\n[OK] Pipeline complete. Output saved to: {final_state.output_path}")
    else:
        print(f"\n✗ Pipeline ended in state: {final_state.stage.value}")
        if final_state.error_message:
            print(f"  Error: {final_state.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()