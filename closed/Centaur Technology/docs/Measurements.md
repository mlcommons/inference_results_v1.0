
To run benchmarks on Centaur Technology's AIC accelerator, use the provided docker container `closed/Centaur Technology/base.Dockerfile`.

Build and run the docker container. Replace `path-to-mlperf-datasets` with your specific path to the input datasets used by the MLPerf benchmark suite. Replace `path-to-custom-workspace` with the path to the `closed/Centaur Technology/code/` path provided.

```
docker image build -f base.Dockerfile -t ncore/base/develop:latest .
docker container run --rm                        \
    --device /dev/ncore_pci                      \
    --volume /path-to-custom-workspace/:/workspace \
    --volume /path-to-mlperf-datasets/:/datasets \
    -it ncore/base/develop:latest /bin/bash
```

After building `Centaur Technology/code/inference/imagenet/cc/main.cc` (and subsequent dependencies in the included code), run it with command-line arguments corresponding to the benchmark, scenario, and configuration desired. Applicable arguments are listed below.
```
OMP_NUM_THREADS={1-8} ./main \
  --model_name={resnet|ssd-mobilenet} \
  --scenario={Offline|SingleStream} \
  --export_model_path={path-to-model} \
  --preprocessed_data_path={path-to-preprocessed-data-path} \
  --num_worker_threads={1-8} \
  --mlperf_conf={path-to-conf} \
  --user_conf={path-to-conf} \
  --accuracy_mode={true|false}
```

The models used are available here:
https://www.dropbox.com/sh/3km3itjic530dpb/AABjOjFQ7TFtVrfzji59weQAa?dl=0

Note that the `preprocessed_data_path` refers to the numpy (`.npy`) preprocessed images that are generated from the reference flow provided by MLCommons / MLPerf. Note that `num_worker_threads` was set to `1` for `SingleStream`, `4` for `ssd-mobilenet Offline`, and `5` for `resnet Offline`; which determines the number of parallel threads running for multiple-sample inputs (e.g., `Offline` provides multiple samples per query). Note that the environment variable `OMP_NUM_THREADS` was set to `1` for `Offline` mode, and `4` for `SingleStream` mode; which determines the intra-operation parallelism.

