#!/usr/bin/env python3
"""A lightweight, log-only simulation of an LLM training run."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from prometheus_client import start_http_server, push_to_gateway
import prometheus_metrics as metrics


PROMETHEUS_PORT = 8001

def env_value(name: str, default: str) -> str:
    return os.environ.get(name, default)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


@dataclass(frozen=True)
class TrainingConfig:
    run_id: str
    total_steps: int
    step_duration: float
    checkpoint_interval: int
    validation_interval: int
    initial_loss: float
    learning_rate: float
    warmup_steps: int
    tokens_per_step: int
    steps_per_epoch: int
    seed: int


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            **getattr(record, "fields", {}),
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


class JsonLogger:
    def __init__(self, level: str = "INFO") -> None:
        self.logger = logging.Logger("training_mock", level=level.upper())
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def emit(
        self, event: str, *, level: int = logging.INFO,
        exc_info: bool = False, **fields: Any
    ) -> None:
        self.logger.log(level, event, extra={"fields": fields}, exc_info=exc_info)


class TrainingSimulator:
    def __init__(self, config: TrainingConfig, logger: JsonLogger) -> None:
        self.config = config
        self.logger = logger
        self.random = random.Random(config.seed)
        self.stop_requested = False
        self.started_at = 0.0

    def request_stop(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True
        self.logger.emit(
            "shutdown_requested",
            run_id=self.config.run_id,
            signal=signal.Signals(signum).name,
        )

    def loss_at(self, step: int) -> float:
        progress = step / self.config.total_steps
        floor = self.config.initial_loss * 0.18
        decay = (self.config.initial_loss - floor) * math.exp(-4.2 * progress)
        noise = self.random.gauss(0, self.config.initial_loss * 0.012)
        return max(floor, floor + decay + noise)

    def learning_rate_at(self, step: int) -> float:
        warmup_steps = min(self.config.warmup_steps, self.config.total_steps)
        if warmup_steps and step <= warmup_steps:
            return self.config.learning_rate * step / warmup_steps

        decay_steps = max(1, self.config.total_steps - warmup_steps)
        decay_progress = (step - warmup_steps) / decay_steps
        return (
            self.config.learning_rate * 0.5 * (1 + math.cos(math.pi * decay_progress))
        )

    def run(self) -> int:
        config = self.config
        self.started_at = time.monotonic()
        self.logger.emit(
            "training_started",
            run_id=config.run_id,
            total_steps=config.total_steps,
            checkpoint_interval=config.checkpoint_interval,
            validation_interval=config.validation_interval,
            seed=config.seed,
        )

        metrics.TRAINING_STEPS_PLANNED.labels(
            run_id=config.run_id,
        ).set(config.total_steps)
        metrics.TRAINING_CHECKPOINT_INTERVAL.labels(
            run_id=config.run_id,
        ).set(config.checkpoint_interval)
        metrics.TRAINING_VALIDATION_INTERVAL.labels(
            run_id=config.run_id,
        ).set(config.validation_interval)
        for step in range(1, config.total_steps + 1):
            if self.stop_requested:
                return self._stop(step - 1)

            step_started = time.monotonic()
            time.sleep(config.step_duration)
            observed_duration = max(time.monotonic() - step_started, 0.000001)
            if config.step_duration > 0 and observed_duration > 2 * config.step_duration:
                self.logger.emit(
                    "slow_train_step",
                    level=logging.WARNING,
                    run_id=config.run_id,
                    step=step,
                    duration_seconds=round(observed_duration, 3),
                    expected_duration_seconds=config.step_duration,
                )
            throughput_noise = self.random.uniform(0.94, 1.06)
            tokens_per_second = (
                config.tokens_per_step / observed_duration * throughput_noise
            )
            elapsed = time.monotonic() - self.started_at
            average_step_time = elapsed / step
            remaining = average_step_time * (config.total_steps - step)

            loss = round(self.loss_at(step), 6)
            learning_rate = round(self.learning_rate_at(step), 10)
            epoch = round(step / config.steps_per_epoch, 4)
            self.logger.emit(
                "train_step",
                level=logging.DEBUG,
                run_id=config.run_id,
                step=step,
                total_steps=config.total_steps,
                epoch=epoch,
                loss=loss,
                learning_rate=learning_rate,
                tokens_per_second=round(tokens_per_second, 2),
                elapsed_seconds=round(elapsed, 3),
                estimated_remaining_seconds=round(remaining, 3),
            )

            metrics.TRAINING_LOSS.labels(
                run_id=config.run_id,
            ).set(loss)
            metrics.TRAINING_LEARNING_RATE.labels(
                run_id=config.run_id,
            ).set(learning_rate)
            metrics.TRAINING_STEPS.labels(run_id=config.run_id).inc()
            metrics.TRAINING_EPOCH.labels(run_id=config.run_id).set(epoch)
            metrics.TRAINING_THROUGHPUT.labels(run_id=config.run_id).set(tokens_per_second)
            metrics.TRAINING_ELAPSED_SECONDS.labels(
                run_id=config.run_id,
            ).set(elapsed)

            if config.validation_interval and step % config.validation_interval == 0:
                validation_loss = self.loss_at(step) * self.random.uniform(1.01, 1.06)
                self.logger.emit(
                    "validation_completed",
                    run_id=config.run_id,
                    step=step,
                    validation_loss=round(validation_loss, 6),
                )
                metrics.TRAINING_VALIDATION_LOSS.labels(
                    run_id=config.run_id,
                ).set(validation_loss)
                metrics.TRAINING_VALIDATION_STEPS.labels(
                    run_id=config.run_id,
                ).set(step)

            if config.checkpoint_interval and step % config.checkpoint_interval == 0:
                self.logger.emit(
                    "checkpoint_saved",
                    run_id=config.run_id,
                    step=step,
                    checkpoint=f"checkpoint-{step:06d}",
                )
                metrics.TRAINING_CHECKPOINT_STEPS.labels(
                    run_id=config.run_id,
                ).set(step)
                metrics.TRAINING_CHECKPOINT_LAST_TIMESTAMP.labels(
                    run_id=config.run_id,
                ).set_to_current_time()

        elapsed = time.monotonic() - self.started_at
        self.logger.emit(
            "training_completed",
            run_id=config.run_id,
            total_steps=config.total_steps,
            elapsed_seconds=round(elapsed, 3),
        )
        metrics.TRAINING_DURATION_TOTAL.labels(
            run_id=config.run_id,
        ).set(elapsed)
        metrics.TRAINING_SUCCESS.labels(
            run_id=config.run_id,
            reason="success",
        ).set(1)
        return 0

    def _stop(self, completed_steps: int) -> int:
        elapsed = time.monotonic() - self.started_at
        self.logger.emit(
            "training_stopped",
            run_id=self.config.run_id,
            completed_steps=completed_steps,
            elapsed_seconds=round(elapsed, 3),
        )
        metrics.TRAINING_SUCCESS.labels(
            run_id=self.config.run_id,
            reason="stopped",
        ).set(0)
        metrics.TRAINING_DURATION_TOTAL.labels(
            run_id=self.config.run_id,
        ).set(elapsed)
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", default=env_value("TRAINING_RUN_ID", str(uuid.uuid4()))
    )
    parser.add_argument(
        "--total-steps",
        type=positive_int,
        default=env_value("TRAINING_TOTAL_STEPS", "100"),
    )
    parser.add_argument(
        "--step-duration",
        type=non_negative_float,
        default=env_value("TRAINING_STEP_DURATION", "1.0"),
        help="simulated seconds per training step",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=non_negative_int,
        default=env_value("TRAINING_CHECKPOINT_INTERVAL", "25"),
        help="zero disables checkpoint events",
    )
    parser.add_argument(
        "--validation-interval",
        type=non_negative_int,
        default=env_value("TRAINING_VALIDATION_INTERVAL", "10"),
        help="zero disables validation events",
    )
    parser.add_argument(
        "--initial-loss",
        type=non_negative_float,
        default=env_value("TRAINING_INITIAL_LOSS", "6.0"),
    )
    parser.add_argument(
        "--learning-rate",
        type=non_negative_float,
        default=env_value("TRAINING_LEARNING_RATE", "0.0003"),
    )
    parser.add_argument(
        "--warmup-steps",
        type=non_negative_int,
        default=env_value("TRAINING_WARMUP_STEPS", "10"),
    )
    parser.add_argument(
        "--tokens-per-step",
        type=positive_int,
        default=env_value("TRAINING_TOKENS_PER_STEP", "4096"),
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=positive_int,
        default=env_value("TRAINING_STEPS_PER_EPOCH", "250"),
    )
    parser.add_argument("--seed", type=int, default=env_value("TRAINING_SEED", "42"))
    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(**vars(args))


def main() -> int:
    config = config_from_args(build_parser().parse_args())
    logger = JsonLogger(env_value("LOG_LEVEL", "INFO"))
    simulator = TrainingSimulator(config, logger)
    signal.signal(signal.SIGTERM, simulator.request_stop)
    signal.signal(signal.SIGINT, simulator.request_stop)
    try:
        start_http_server(PROMETHEUS_PORT)
        return_code = simulator.run()
    except Exception:
        logger.emit(
            "training_failed",
            level=logging.ERROR,
            exc_info=True,
            run_id=config.run_id,
        )
        metrics.TRAINING_SUCCESS.labels(
            run_id=config.run_id,
            reason="exception",
        ).set(0)
        return_code = 1

    try:
        # get PUSHGATEWAY_URL from environment variable
        PUSHGATEWAY_URL = env_value("PUSHGATEWAY_URL",
                                    "pushgateway.observability-lab.svc.cluster.local:9091")
        push_to_gateway(PUSHGATEWAY_URL, job='training-mock', grouping_key={'run_id': config.run_id}, registry=metrics.registry)
    except Exception:
        logger.emit(
            "prometheus_push_failed",
            level=logging.WARNING,
            exc_info=True,
            run_id=config.run_id,
        )
    return return_code


if __name__ == "__main__":
    sys.exit(main())
