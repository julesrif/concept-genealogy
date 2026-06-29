"""
Unit tests for main.py - CLI entry point.

Minimal tests that the FastAPI app can be imported and main() invokes the Orchestrator.
"""

import pytest
from unittest.mock import patch, MagicMock

from core.state import PipelineState, PipelineStage


class TestMainImport:
    """Test that main.py can be imported without errors."""

    def test_import_main_module(self):
        """main module imports cleanly."""
        import main  # noqa: F401


class TestMainFunction:
    """Test main() function with mocked dependencies."""

    @patch("main.Orchestrator")
    @patch("main.load_config")
    @patch("sys.argv", ["main.py", "attention mechanism"])
    def test_main_runs_orchestrator(self, mock_load_config, mock_orchestrator_cls):
        """main() calls Orchestrator.run() with the concept arg."""
        mock_config = MagicMock()
        mock_config.log_level = "INFO"
        mock_config.dict.return_value = {
            "max_depth": 3,
            "groq_model_fast": "llama-3.1-8b-instant",
            "groq_model_reasoning": "llama-3.3-70b-versatile",
        }
        mock_load_config.return_value = mock_config

        mock_orch_instance = MagicMock()
        mock_orch_instance.run.return_value = PipelineState(
            stage=PipelineStage.DONE,
            output_path="output/result.json",
        )
        mock_orchestrator_cls.return_value = mock_orch_instance

        from main import main
        main()

        mock_orch_instance.run.assert_called_once_with("attention mechanism")

    @patch("main.Orchestrator")
    @patch("main.load_config")
    @patch("sys.argv", ["main.py", "attention mechanism", "--depth", "2"])
    def test_main_applies_depth_override(self, mock_load_config, mock_orchestrator_cls):
        """--depth CLI flag overrides config max_depth."""
        mock_config = MagicMock()
        mock_config.log_level = "INFO"
        mock_config.dict.return_value = {
            "max_depth": 3,
            "groq_model_fast": "llama-3.1-8b-instant",
            "groq_model_reasoning": "llama-3.3-70b-versatile",
        }
        mock_load_config.return_value = mock_config

        mock_orch_instance = MagicMock()
        mock_orch_instance.run.return_value = PipelineState(
            stage=PipelineStage.DONE,
            output_path="output/result.json",
        )
        mock_orchestrator_cls.return_value = mock_orch_instance

        from main import main
        main()

        # Check that config_dict passed to Orchestrator has max_depth=2
        init_kwargs = mock_orchestrator_cls.call_args[0][0]
        assert init_kwargs["max_depth"] == 2

    @patch("main.Orchestrator")
    @patch("main.load_config")
    @patch("sys.argv", ["main.py", "attention mechanism"])
    def test_main_exits_on_failure(self, mock_load_config, mock_orchestrator_cls):
        """Pipeline failure → sys.exit(1)."""
        mock_config = MagicMock()
        mock_config.log_level = "INFO"
        mock_config.dict.return_value = {
            "max_depth": 3,
            "groq_model_fast": "llama-3.1-8b-instant",
            "groq_model_reasoning": "llama-3.3-70b-versatile",
        }
        mock_load_config.return_value = mock_config

        mock_orch_instance = MagicMock()
        mock_orch_instance.run.return_value = PipelineState(
            stage=PipelineStage.FAILED,
            error_message="DOI not found",
        )
        mock_orchestrator_cls.return_value = mock_orch_instance

        from main import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
