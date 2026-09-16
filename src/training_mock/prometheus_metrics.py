
from prometheus_client import Gauge, Counter, CollectorRegistry

registry = CollectorRegistry()

TRAINING_STEPS_PLANNED = Gauge(
    name="steps_planned",
    documentation="total training steps",
    labelnames=["run_id"],
    namespace="training",
)
TRAINING_STEPS = Counter(
    name="steps",
    documentation="training steps",
    labelnames=["run_id"],
    namespace="training",
)
TRAINING_LOSS = Gauge(
    name="loss",
    documentation="training losses",
    labelnames=["run_id"],
    namespace="training",
)
TRAINING_LEARNING_RATE = Gauge(
    name="learning_rate",
    documentation="training learning rate",
    labelnames=["run_id"],
    namespace="training",
)
TRAINING_EPOCH = Gauge(
    name="epoch",
    documentation="training epoch",
    labelnames=["run_id"],
    namespace="training",
)
TRAINING_THROUGHPUT = Gauge(
    name="throughput",
    documentation="training throughtput",
    labelnames=["run_id"],
    namespace="training",
    unit="tokens_per_second",
)
TRAINING_SUCCESS = Gauge(
    name="success",
    documentation="training success, 1 = sucess, 0 = failure",
    labelnames=["run_id", "reason"],
    namespace="training",
    registry=registry
)
TRAINING_DURATION_TOTAL = Gauge(
    name="duration",
    documentation="total training duration",
    labelnames=["run_id"],
    namespace="training",
    unit="seconds",
    registry=registry
)
TRAINING_ELAPSED_SECONDS = Gauge(
    name="elapsed",
    documentation="training elapsed seconds",
    labelnames=["run_id"],
    namespace="training",
    unit="seconds",
)
TRAINING_VALIDATION_LOSS = Gauge(
    name="loss",
    documentation="validation losses",
    labelnames=["run_id"],
    namespace="training",
    subsystem="validation",
)
TRAINING_VALIDATION_STEPS = Gauge(
    name="steps",
    documentation="validation steps",
    labelnames=["run_id"],
    namespace="training",
    subsystem="validation",
)
TRAINING_VALIDATION_INTERVAL = Gauge(
    name="interval",
    documentation="validation interval",
    labelnames=["run_id"],
    namespace="training",
    subsystem="validation",
)
TRAINING_CHECKPOINT_STEPS = Gauge(
    name="steps",
    documentation="checkpoint steps",
    labelnames=["run_id"],
    namespace="training",
    subsystem="checkpoint",
)
TRAINING_CHECKPOINT_LAST_TIMESTAMP = Gauge(
    name="last_timestamp",
    documentation="UNIX timestamp when the latest checkpoint is saved",
    labelnames=["run_id"],
    namespace="training",
    subsystem="checkpoint",
    unit="seconds"
)
TRAINING_CHECKPOINT_INTERVAL = Gauge(
    name="interval",
    documentation="checkpoint interval",
    labelnames=["run_id"],
    namespace="training",
    subsystem="checkpoint",
)

