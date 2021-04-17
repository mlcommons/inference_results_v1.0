# Boot/BIOS Firmware Settings

Out-of-the-box.

# Management Firmware Settings
  
Out-of-the-box.

# Power Management Settings

## TDP Settings

### Set 15W TDP

<pre>
<b>[anton@aedk1 ~]&dollar;</b> echo 15000000 | sudo tee /sys/class/hwmon/hwmon*/power1_max
15000000
</pre>

### Reboot

<pre>
<b>[anton@aedk1 ~]&dollar;</b> sudo reboot
</pre>

### Check 15W TDP

<pre>
<b>[anton@aedk1 ~]&dollar;</b> cat /sys/class/hwmon/hwmon*/power1_max
15000000
</pre>
