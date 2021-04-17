# Boot/BIOS Firmware Settings

Out-of-the-box.

# Management Firmware Settings
  
Out-of-the-box.

# Power Management Settings

## Set Power Mode

<pre><font color="#859900"><b>anton@xavier</b></font>:<font color="#268BD2"><b>~</b></font>$ sudo nvpmodel -m 0
</pre>

## Set Fan Mode

<pre><font color="#859900"><b>anton@xavier</b></font>:<font color="#268BD2"><b>~</b></font>$ sudo nvpmodel -d cool
</pre>

## Query Fan and Power Mode

<pre><font color="#859900"><b>anton@xavier</b></font>:<font color="#268BD2"><b>~</b></font>$ sudo nvpmodel -q
NV Fan Mode:cool
NV Power Mode: MAXN
0
</pre>
