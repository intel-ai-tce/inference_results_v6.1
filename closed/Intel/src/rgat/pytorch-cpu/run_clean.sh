sudo echo never  > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
sudo echo never  > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
sudo echo always > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
sudo echo always > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
sudo echo 1 > /proc/sys/vm/compact_memory; sleep 1
sudo echo 3 > /proc/sys/vm/drop_caches; sleep 1
