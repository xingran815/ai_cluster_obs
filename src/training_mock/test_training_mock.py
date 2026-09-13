import json
import io
import logging
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from training_mock import JsonLogger, TrainingConfig, TrainingSimulator


PROJECT_DIR = Path(__file__).parent


def config(**overrides):
    values = {
        "run_id": "test-run",
        "total_steps": 10,
        "step_duration": 0,
        "checkpoint_interval": 5,
        "validation_interval": 2,
        "initial_loss": 6.0,
        "learning_rate": 0.0003,
        "warmup_steps": 2,
        "tokens_per_step": 4096,
        "steps_per_epoch": 10,
        "seed": 42,
    }
    values.update(overrides)
    return TrainingConfig(**values)


class TrainingSimulatorTests(unittest.TestCase):
    def test_log_levels_and_json_exceptions(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            logger = JsonLogger("INFO")
            logger.emit("hidden", level=logging.DEBUG)
            logger.emit("started", run_id="test-run")
            logger.emit("slow_step", level=logging.WARNING)
            try:
                raise RuntimeError("simulated failure")
            except RuntimeError:
                logger.emit("failed", level=logging.ERROR, exc_info=True)
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r["level"] for r in records], ["INFO", "WARNING", "ERROR"])
        self.assertEqual(records[0]["run_id"], "test-run")
        self.assertIn("RuntimeError: simulated failure", records[-1]["exception"])

    def test_loss_trends_downward(self):
        simulator = TrainingSimulator(config(), JsonLogger())
        early_samples = [simulator.loss_at(1) for _ in range(20)]
        late_samples = [simulator.loss_at(10) for _ in range(20)]
        self.assertGreater(sum(early_samples) / len(early_samples), sum(late_samples) / len(late_samples))

    def test_learning_rate_warms_up_and_decays(self):
        simulator = TrainingSimulator(config(), JsonLogger())
        self.assertAlmostEqual(simulator.learning_rate_at(1), 0.00015)
        self.assertAlmostEqual(simulator.learning_rate_at(2), 0.0003)
        self.assertAlmostEqual(simulator.learning_rate_at(10), 0.0)

    def test_cli_emits_expected_events(self):
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "training_mock.py"),
                "--run-id",
                "cli-test",
                "--total-steps",
                "2",
                "--step-duration",
                "0",
                "--validation-interval",
                "1",
                "--checkpoint-interval",
                "2",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "LOG_LEVEL": "DEBUG"},
        )
        records = [json.loads(line) for line in result.stdout.splitlines()]
        events = [record["event"] for record in records]
        self.assertEqual(events.count("train_step"), 2)
        self.assertTrue(all(r["level"] == "DEBUG" for r in records if r["event"] == "train_step"))
        self.assertIn("validation_completed", events)
        self.assertIn("checkpoint_saved", events)
        self.assertEqual(events[-1], "training_completed")


if __name__ == "__main__":
    unittest.main()
