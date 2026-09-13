# Training mock

A standard-library-only Python process that simulates an LLM training run and writes JSON events to standard output. It does not expose a network endpoint or Prometheus metrics.

## Local environment

Create and activate an isolated environment:

```sh
cd src/training_mock
python3 -m venv .venv
source .venv/bin/activate
```

There are no packages to install. Run a short simulation:

```sh
python training_mock.py --run-id practice-001 --total-steps 10 --step-duration 0.2
```

Run the tests:

```sh
python -m unittest -v
```

## Container image

Build the image from this directory:

```sh
docker build -t training-mock:local .
```

Run it:

```sh
docker run --rm training-mock:local \
  --run-id container-001 \
  --total-steps 10 \
  --step-duration 0.2
```

To load the locally built image into a kind cluster later:

```sh
kind load docker-image training-mock:local --name YOUR_CLUSTER_NAME
```

That command only loads the image. It does not create any Kubernetes resources.

## Configuration

Every setting is available as a command-line option and an environment variable. Command-line values take precedence.

| Option | Environment variable | Default |
| --- | --- | ---: |
| `--run-id` | `TRAINING_RUN_ID` | random UUID |
| `--total-steps` | `TRAINING_TOTAL_STEPS` | `100` |
| `--step-duration` | `TRAINING_STEP_DURATION` | `1.0` |
| `--checkpoint-interval` | `TRAINING_CHECKPOINT_INTERVAL` | `25` |
| `--validation-interval` | `TRAINING_VALIDATION_INTERVAL` | `10` |
| `--initial-loss` | `TRAINING_INITIAL_LOSS` | `6.0` |
| `--learning-rate` | `TRAINING_LEARNING_RATE` | `0.0003` |
| `--warmup-steps` | `TRAINING_WARMUP_STEPS` | `10` |
| `--tokens-per-step` | `TRAINING_TOKENS_PER_STEP` | `4096` |
| `--steps-per-epoch` | `TRAINING_STEPS_PER_EPOCH` | `250` |
| `--seed` | `TRAINING_SEED` | `42` |

Set an interval to `0` to disable its validation or checkpoint events. The process handles `SIGTERM` and `SIGINT`, emits shutdown events, and exits with status 130.
