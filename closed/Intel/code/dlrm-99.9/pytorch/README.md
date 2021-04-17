# DLRM MLPerf Inference v1.0 Intel Submission

## HW and SW requirements
### 1. HW requirements
| HW | configuration |
| -: | :- |
| CPU | CPX-6 @ 8 sockets/Node |
| DDR | 384G/socket @ 3200 MT/s |
| SSD | 1 SSD/Node @ >= 1T |

### 2. SW requirements
| SW |configuration  |
|--|--|
| GCC | GCC 9.3  |

## Steps to run DLRM

### 1. Install anaconda 3.0
```
  wget -c https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ~/anaconda3.sh -b -p ~/anaconda3
  ~/anaconda3/bin/conda create -n dlrm python=3.7
  export PATH=~/anaconda3/bin:$PATH
  source ~/anaconda3/bin/activate dlrm
```
### 2. Install dependency packages and Pytorch/IPEX
```
  mkdir <workfolder>
  cd <workfolder>
  git clone <path/to/this/repo>
  cp <path/to/this/repo>/closed/Intel/code/dlrm-99.9/pytorch/prepare_env.sh .
  bash prepare_env.sh
```
### 3. Prepare DLRM dataset and code    
(1) Prepare DLRM dataset
```
   Create a directory (such as ${WORKDIR}\dataset\terabyte_input) which contain day_day_count.npz, day_fea_count.npz and terabyte_processed_test.bin, for directions about how to get the dataset please refer to https://github.com/mlcommons/inference/tree/master/recommendation/dlrm/pytorch.
```
(2) Prepare pre-trained DLRM model
```
   ln -s <path/to/this/repo>/closed/Intel/code/dlrm-99.9/pytorch dlrm_pytorch
   cd dlrm_pytorch/python/model
   wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt -O dlrm_terabyte.pytorch
```
### 4. Run command for server and offline mode

(1) cd dlrm_pytorch

(2) configure DATA_DIR and MODEL_DIR #you can modify the setup_dataset.sh, then 'source ./setup_dataset.sh'
```
   export DATA_DIR=           # the path of dataset, for example as ${WORKDIR}\dataset\terabyte_input
   export MODEL_DIR=          # the path of pre-trained model, for example as ${WORKDIR}\dlrm_pytorch\python\model
```
(3) configure offline/server mode options # currenlty used options for each mode is in setup_env_xx.sh, You can modify it, then 'source ./setup_env_xx.sh'
```
   export NUM_SOCKETS=        # i.e. 8
   export CPUS_PER_SOCKET=    # i.e. 28
   export CPUS_PER_PROCESS=   # i.e. 14. which determine how many cores for one processe running on one socket
                              #   process_number = $CPUS_PER_SOCKET / $CPUS_PER_PROCESS
   export CPUS_PER_INSTANCE=  # i.e. 14. which determine how many cores used for one instance inside one process
                              #   instance_number_per_process = $CPUS_PER_PROCESS / CPUS_PER_INSTANCE
                              #   total_instance_number_in_system = instance_number_per_process * process_number
```
(4) command line
   Please updae setup_env_server.sh and setup_env_offline.sh and user.conf according to your platform resource.
```
   # server-performance-mode
   sudo ./run_clean.sh
   source ./setup_env_server.sh
   ./run_main.sh server

   # server-accuracy-mode
   sudo ./run_clean.sh
   source ./setup_env_server.sh
   ./run_main.sh server accuracy

   # offline-performance-mode
   sudo ./run_clean.sh
   source ./setup_env_offline.sh
   ./run_main.sh offline

   # offline-accuracy-mode
   sudo ./run_clean.sh
   source ./setup_env_offline.sh
   ./run_main.sh offline accuracy
```
