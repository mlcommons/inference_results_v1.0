# Qualcomm - MLPerf Inference v1.0

<a name="submit_r282_z93_q5_server"></a>
## Server

<a name="submit_r282_z93_q5_server_accuracy"></a>
### Accuracy

<pre>
<b>[anton@ax530b-03-giga ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q5 --library=qaic-v1.3.14-aic100 --model=ssd_resnet34 --scenario=server \
--mode=accuracy --target_qps=1555 --dataset_size=5000 --buffer_size=500
...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.198
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.381
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.179
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.120
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.254
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.200
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.192
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.427
mAP=19.810%
==========================================================================================
</pre>

<a name="submit_r282_z93_q5_server_performance"></a>
### Performance

<pre>
<b>[anton@ax530b-03-giga ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q5 --library=qaic-v1.3.14-aic100 --model=ssd_resnet34 --scenario=server \
--mode=performance --target_qps=1555 --dataset_size=5000 --buffer_size=64
</pre>

<a name="submit_r282_z93_q5_server_power"></a>
### Power

<pre>
<b>[anton@ax530b-03-giga ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q5 --library=qaic-v1.3.14-aic100 --model=ssd_resnet34 --scenario=server \
--mode=performance --target_qps=1555 --dataset_size=5000 --buffer_size=64 \
--power=yes --power_server_ip=172.24.66.69 --power_server_port=4951 --sleep_before_ck_benchmark_sec=90
</pre>

<a name="submit_r282_z93_q5_server_compliance"></a>
### Compliance

<pre>
<b>[anton@ax530b-03-giga ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=r282_z93_q5 --library=qaic-v1.3.14-aic100 --model=ssd_resnet34 --scenario=server \
--compliance,=TEST04-A,TEST04-B,TEST05,TEST01 --target_qps=1555 --dataset_size=5000 --buffer_size=64
</pre>

## Info

Please contact anton@krai.ai if you have any problems or questions.
