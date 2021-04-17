# Make targets

These `make` commands are run outside the Docker container:

- `make prebuild`: Runs the following steps:
  - `make build_docker`: Builds docker image.
  - `make docker_add_user`: Adds current user to the docker image.
  - `make attach_docker`: Runs docker image with an interactive session and with current working directory bound to `/work` in the container. You are left at a shell inside the container at `/work`.
- `make download_dataset`: Downloads datasets.
- `make preprocessed_data`: Preprocesses the downloaded datasets.

These `make` commands are run inside the Docker container shell, that you get by running `make prebuild` (see above):

- `make build`: Runs the following steps:
  - `make clone_loadgen`: Clone the official MLPerf inference GitHub repo.
  - `make build_plugins`: Builds TensorRT plugins.
  - `make build_loadgen`: Builds LoadGen source codes.
  - `make build_harness`: Builds the harnesses.
- `make run RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIOS> [OTHER RUN_ARGS FLAGS]"` runs the following steps:
  - `make generate_engines RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIOS> [OTHER RUN_ARGS FLAGS]"`: Generates TensorRT optimized engine files.
  - `make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=<SCENARIOS> [OTHER RUN_ARGS FLAGS]"`: Runs harnesses with engine files and LoadGen.
  - See [RUN_ARGS flags](#run_args-flags).
- `make calibrate RUN_ARGS="--benchmarks=resnet50"`: Generates calibration caches.
  - See [RUN_ARGS flags](#run_args-flags).
- `make clean`: Cleans up all the build directories. Please note that you will need to exit the docker container and run `make prebuild` again after the cleaning.
- `make clean_shallow`: Cleans up only the files needed to make a clean build.
- `make info`: Displays useful build information.
- `make shell`: Spawns an interactive shell that inherits all the environment variables that the Makefile sets to provide an environment that mirrors the build/run environment.

# RUN_ARGS flags

- `--benchmarks=resnet50`: See [Benchmarks](README.md#benchmarks).
- `--scenarios=comma,separated,list,of,scenario,names`: See [Scenarios](README.md#scenarios).
- `--config_ver=default`: See [Config Versions](#config-versions).
- `--test_mode=[PerformanceOnly,AccuracyOnly]`: Specifies which LoadGen mode to run with.
- `--force_calibration`: Forces recalculation of calibration cache.
- `--log_dir=path/to/logs`: Specifies where to save logs.
- `--log_copy_detail_to_stdout`: Prints LoadGen detailed logs to stdout as well as to the log files.
- `--verbose`: Prints out verbose logs.

## Config versions

There are four types of config versions which can be passed into the `--config_ver` flag:

1. `default`:
This is the default config, as shown in `config.json`.
No overrides are applied
Uses custom harness with low accuracy target.
Supported for all benchmarks.

`all`:
Runs every `config_ver` available in `config.json`.
This will override all other `config_ver` options specified by the flag.

Passing in comma-separated list of config versions or passing in `all` allows you to generate the engines or to run harness for multiple config versions in one command.