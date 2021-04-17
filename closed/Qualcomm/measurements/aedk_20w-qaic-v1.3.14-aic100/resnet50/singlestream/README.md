# Qualcomm - MLPerf Inference v1.0

<a name="submit_aedk_16nsp_singlestream"></a>
## Single Stream

<a name="submit_aedk_16nsp_singlestream_accuracy"></a>
### Accuracy

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=aedk.16nsp --library=qaic-v1.3.14-aic100 --model=resnet50 --scenario=singlestream \
--mode=accuracy --target_latency=1 --dataset_size=50000 --buffer_size=500
...
accuracy=75.942%, good=37971, total=50000
==========================================================================================
</pre>

<a name="submit_aedk_16nsp_singlestream_performance"></a>
### Performance

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=aedk.16nsp --library=qaic-v1.3.14-aic100 --model=resnet50 --scenario=singlestream \
--mode=performance --target_latency=1 --dataset_size=50000 --buffer_size=1024
</pre>

<a name="submit_aedk_16nsp_singlestream_power"></a>
### Power

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=aedk.16nsp --library=qaic-v1.3.14-aic100 --model=resnet50 --scenario=singlestream \
--mode=performance --target_latency=1 --dataset_size=50000 --buffer_size=1024 \
--power=yes --power_server_ip=192.168.0.19 --power_server_port=4951 --sleep_before_ck_benchmark_sec=30
</pre>

<a name="submit_aedk_16nsp_singlestream_compliance"></a>
### Compliance

<pre>
<b>[anton@aedk3 ~]&dollar;</b> time ck run cmdgen:benchmark.image-classification.qaic-loadgen --verbose \
--sut=aedk.16nsp --library=qaic-v1.3.14-aic100 --model=resnet50 --scenario=singlestream \
--compliance,=TEST04-A,TEST04-B,TEST05,TEST01 --target_latency=1 --dataset_size=50000 --buffer_size=1024
</pre>

## Info

Please contact anton@krai.ai if you have any problems or questions.
