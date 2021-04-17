echo "Test start time: $(date)"
./Release/ov_mlperf --scenario Server \
	--mode Accuracy \
	--mlperf_conf Configs/mlperf.conf \
	--user_conf Configs/resnet50/user.conf \
	--model_name resnet50 \
	--data_path /home/dell/CK-TOOLS/dataset-imagenet-ilsvrc2012-val \
	--nireq 28 \
	--nthreads 56 \
	--nstreams 14 \
	--total_sample_count 50000 \
	--warmup_iters 0 \
	--model_path Models/resnet50/resnet50_int8.xml
echo "Test end time: $(date)"

