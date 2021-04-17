Run the following commands to perform DLRM inference in Offline mode:

1. Accuracy mode:
```
./run_local.sh pytorch dlrm terabyte cpu --scenario Offline --max-ind-range=40000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --samples-per-query-offline=204800 --accuracy --mlperf-bin-loader --profile dlrm-terabyte-neuchips
```

2. Performance mode:
```
./run_local.sh pytorch dlrm terabyte cpu --scenario Offline --max-ind-range=40000000 --samples-to-aggregate-quantile-file=./tools/dist_quantile.txt --max-batchsize=2048 --samples-per-query-offline=204800 --mlperf-bin-loader --profile dlrm-terabyte-neuchips
```

For more details, please refer to `https://github.com/mlperf/inference/tree/master/recommendation/dlrm/pytorch`.
