echo "Test start time: $(date)"
./Release/ov_mlperf --scenario Offline \
	--mode Accuracy \
	--mlperf_conf Configs/mlperf.conf \
	--user_conf Configs/resnet50/user.conf \
	--model_name resnet50 \
	--data_path /home/dell/CK-TOOLS/dataset-imagenet-ilsvrc2012-val \
	--nireq 224 \
	--nthreads 56 \
	--nstreams 112 \
	--total_sample_count 50000 \
	--warmup_iters 50 \
	--model_path Models/resnet50/resnet50_int8.xml
echo "Test end time: $(date)"
