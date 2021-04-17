```
$ make
```


* SSD-MobileNet V1
```
# ./benchmark --config="mlperf.conf" --scenario="SingleStream" --mode="AccuracyOnly" --model="SSDMobileNetV1"
```
* SSD-ResNet
```
./benchmark --config="mlperf.conf" --scenario="SingleStream" --mode="AccuracyOnly" --model="SSDResNet"
```
* ResNet50
```
./benchmark --config="mlperf.conf" --scenario="SingleStream" --mode="AccuracyOnly" --model="ResNet50"
```

* Check out accuracies

SingleStream
MultiStream
Offline
Server

AccuracyOnly
PerformanceOnly

SSDMobileNetV1
SSDResNet
ResNet50