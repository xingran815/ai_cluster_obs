import json
import io
import logging
import os
import subprocess
import sys
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

from training_mock import JsonLogger, TrainingConfig, TrainingSimulator, main


PROJECT_DIR = Path(__file__).parent


@contextmanager
def gateway_stub():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_PUT(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.path, body.decode()))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


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
        with gateway_stub() as (gateway_url, requests):
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
                env={**os.environ, "LOG_LEVEL": "DEBUG", "PUSHGATEWAY_URL": gateway_url},
                timeout=10,
            )
        records = [json.loads(line) for line in result.stdout.splitlines()]
        events = [record["event"] for record in records]
        self.assertEqual(events.count("train_step"), 2)
        self.assertTrue(all(r["level"] == "DEBUG" for r in records if r["event"] == "train_step"))
        self.assertIn("validation_completed", events)
        self.assertIn("checkpoint_saved", events)
        self.assertEqual(events[-1], "training_completed")

        self.assertEqual(len(requests), 1)
        path, body = requests[0]
        self.assertEqual(path, "/metrics/job/training-mock/run_id/cli-test")
        self.assertIn('training_success{reason="success",run_id="cli-test"} 1.0', body)
        self.assertIn('training_duration_seconds{run_id="cli-test"}', body)

    def test_failed_push_preserves_training_exit_code(self):
        for training_result in (0, 130, RuntimeError("training failed")):
            with self.subTest(training_result=training_result):
                output = io.StringIO()
                with (
                    patch("sys.argv", ["training_mock", "--run-id", "push-failure-test"]),
                    patch("sys.stdout", output),
                    patch("training_mock.signal.signal"),
                    patch("training_mock.start_http_server"),
                    patch("training_mock.TrainingSimulator.run") as run,
                    patch("training_mock.push_to_gateway", side_effect=ConnectionError("gateway unavailable")),
                ):
                    if isinstance(training_result, Exception):
                        run.side_effect = training_result
                        expected_code = 1
                    else:
                        run.return_value = training_result
                        expected_code = training_result
                    self.assertEqual(main(), expected_code)
                records = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(records[-1]["event"], "prometheus_push_failed")
                self.assertEqual(records[-1]["level"], "WARNING")


if __name__ == "__main__":
    unittest.main()
