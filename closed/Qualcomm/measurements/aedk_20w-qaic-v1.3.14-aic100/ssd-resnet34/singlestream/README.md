# Qualcomm - MLPerf Inference v1.0

<a name="submit_aedk_16nsp_singlestream"></a>
## Single Stream

<a name="submit_aedk_16nsp_singlestream_accuracy"></a>
### Accuracy

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=aedk.16nsp --library=qaic-v1.3.14-aic100 --model=ssd_resnet34 --scenario=singlestream \
--mode=accuracy --dataset_size=5000 --buffer_size=500
...
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.199
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.381
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.180
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.120
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.255
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.233
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.200
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.363
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.192
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.427
mAP=19.870%
==========================================================================================
</pre>

<a name="submit_aedk_16nsp_singlestream_performance"></a>
### Performance

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=aedk.16nsp --library=qaic-v1.3.14-aic100 --model=ssd_resnet34 --scenario=singlestream \
--mode=performance --target_latency=30 --dataset_size=5000 --buffer_size=64
</pre>

<a name="submit_aedk_16nsp_singlestream_power"></a>
### Power

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=aedk.16nsp --library=qaic-v1.3.14-aic100 --model=ssd_resnet34 --scenario=singlestream \
--mode=performance --target_latency=30 --dataset_size=5000 --buffer_size=64 \
--power=yes --power_server_ip=192.168.0.19 --power_server_port=4951 --sleep_before_ck_benchmark_sec=30
</pre>

<a name="submit_aedk_16nsp_singlestream_compliance"></a>
### Compliance

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.object-detection.qaic-loadgen --verbose \
--sut=aedk.16nsp --library=qaic-v1.3.14-aic100 --model=ssd_resnet34 --scenario=singlestream \
--compliance,=TEST04-A,TEST04-B,TEST05,TEST01 --target_latency=30 --dataset_size=5000 --buffer_size=64
</pre>

## Info

Please contact anton@krai.ai if you have any problems or questions.
