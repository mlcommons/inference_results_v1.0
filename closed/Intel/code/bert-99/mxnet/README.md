# 1. Setup python environment
```bash
sudo apt install g++
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
chmod +x Anaconda3-2020.02-Linux-x86_64.sh
bash ./Anaconda3-2020.02-Linux-x86_64.sh
conda create -n env_name python=3.7.7
conda activate env_name
```
# 2. Install MKL
```bash

sudo bash
# <type your user password when prompted.  this will put you in a root shell>
# cd to /tmp where this shell has write permission
cd /tmp
# now get the key:
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# now install that key
apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# now remove the public key file exit the root shell
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
exit
sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update
sudo apt-get install intel-mkl-2019.5-075
```
# 3. Prepare enviroment, loadgen, MXNet, gluon-nlp and generate the quantized model 
For MXNet (https://github.com/apache/incubator-mxnet), BERT optimizations patch must be applied.
To build properly MXNet checkout repository to following commit-sha: cda1aeb1683dfd586b49bb7407a9d7df5a0b3c2a
Then on the top of above commit apply changes from PR: https://github.com/apache/incubator-mxnet/pull/20016 
`prepare_env.sh` script should build MXNet properly

```
mkdir <workdir>
git clone <path/to/this/repo>
cp <path/to/this/repo>/closed/Intel/code/bert-99/mxnet/prepare_env.sh . 
bash prepare_env.sh
```

# 4. Run Offline/Server Scenario
```bash
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:$LD_LIBRARY_PATH 

# run performance calibration to collect best batch size for different sequence length
# this step can take very long period
# for a quick run, do not run the calibrate step, copy the pre-calibrate profile instead
export BATCH_SIZE=24
export NUM_INSTANCE=14
export CPUS_PER_INSTANCE=4
rm prof.py
./run.sh calibrate
cp prof_new.py prof.py

# instead of the step above, we can use a pre calibrated prof.py
cp <path/to/this/repo>/closed/Intel/calibration/MXNet/bert/profiles/prof_clx28c.py prof.py

# Please update setup_env_offline.sh and setup_env_server.sh and user.conf according to your platform resource.
# run offline scenario
sudo bash ./run_clean.sh
source setup_env_offline.sh
./run.sh offline

# run offline with accuracy scenario
source setup_env_offline.sh
sudo bash ./run_clean.sh
./run.sh offline accuracy

# run server scenario
source setup_env_server.sh
sudo bash ./run_clean.sh
./run.sh server

# run server with accuracy scenario
source setup_env_server.sh
sudo bash ./run_clean.sh
./run.sh server accuracy
```
