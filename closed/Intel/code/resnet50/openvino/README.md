# MLPerf Intel OpenVino OMP CPP v1.0 Inference Build

This SW has only be tested on Ubuntu 20.04.

Benchmarks and Scenarios in this release:
*  Resnet50-v1.5, ssd-resnet34 Offline and Server


## BKC on CPX, ICX systems
 - Turbo ON: echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
 - Set CPU governor to performance (Please rerun this command after reboot):  
	 - echo performance | sudo tee
	   /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor   
           OR
	 - cpupower frequency-set -g performance

## How to Build

Two ways to build
+ Run the build script: ```build-ovmlperf.sh```
+ Follow instructions in ```src```

## How to Run

1. Navigate to ```src/scripts/```
2. Modify ```setup_envs.sh``` with relevant library paths
3. From ```src/script```, navigate to ```1-node-<sockets>-<platform>-ov``` to run desired test

### Performance

Syntax to run a **Performance** benchmark

```
<model>-<scenario>.sh
```

For instance to run ```resnet50``` **Offline**:
```
resnet50-offline.sh
```

### Accuracy

+ User first runs the benchmark in ```Accuracy``` mode to generate ```mlperf_log_accuracy.json```
+ User then runs a dedicated accuracy tool provided by MLPerf

Syntax to generate Accuracy logs:

```
<model>-<scenario>-acc.sh
```

For instance:

```
resnet50-offline-acc.sh
```

+ To compute the **Top-1** accuracy for ```resnet50``` (after running the Accuracy script), run the command below:
```
python </path/to/mlperf-inference>/vision/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file mlperf_log_accuracy.json \
    --imagenet-val-file </path/to/dataset-imagenet-ilsvrc2012-val>/val_map.txt \
    --dtype float32
```

+ For ssd-resnet34:
```
python </path/to/mlperf-inference>/vision/classification_and_detection/tools/accuracy-coco.py --mlperf-accuracy-file mlperf_log_accuracy.json \
    --coco-dir </path/to/dataset-coco-2017-val>
```
    
## Known issues

* Issue:
terminate called after throwing an instance of 'InferenceEngine::details::InferenceEngineException'
  what():  can't protect

Solution:
Patch with the following your current and any further submission machines as root:

 1. Add the following line to **/etc/sysctl.conf**: 
    vm.max_map_count=2097152 
 
 2. You may want to check that current value is
    too small with `cat /proc/sys/vm/max_map_count` 
    
 3. Reload the config as
    root: `sysctl -p` (this will print the updated value)
