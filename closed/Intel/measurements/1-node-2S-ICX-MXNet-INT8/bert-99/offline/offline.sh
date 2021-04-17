export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:$LD_LIBRARY_PATH

# run offline scenario
export BATCH_SIZE=64
export NUM_INSTANCE=20
export CPUS_PER_INSTANCE=8
./run.sh offline

