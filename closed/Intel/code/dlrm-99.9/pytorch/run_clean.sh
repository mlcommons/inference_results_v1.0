echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sudo echo 0 > /proc/sys/kernel/numa_balancing
sudo echo 100 > /sys/devices/system/cpu/intel_pstate/min_perf_pct


# Clean resources
echo never  > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
echo never  > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
echo always > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
echo always > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
echo 1 > /proc/sys/vm/compact_memory; sleep 1
echo 3 > /proc/sys/vm/drop_caches; sleep 1
