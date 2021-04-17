# TODO 1 -- change logging to 1.0
# TODO 2 -- change ipex to public repo
# RNNT MLPerf Inference v1.0 BKC

## SW requirements
###
| SW |configuration |
|--|--|
| GCC | GCC 9.3 |

## Steps to run RNNT

### 1. Install anaconda 3.0
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ~/anaconda3.sh -b -p ~/anaconda3
  ~/anaconda3/bin/conda create -n rnnt python=3.7

  export PATH=~/anaconda3/bin:$PATH
  source ~/anaconda3/bin/activate rnnt
```
### 2. Prepare code and enviroment
```
  mkdir <workdir>
  git clone <path/to/this/repo>
  cp <path/to/this/repo>/closed/Intel/code/rnnt/prepare_env.sh .
  bash prepare_env.sh
  
```
### 3. Dataset and model
```
  work_dir=mlperf-rnnt-librispeech
  local_data_dir=$work_dir/local_data
  mkdir -p $local_data_dir
  librispeech_download_dir=.
  wget https://www.openslr.org/resources/12/dev-clean.tar.gz
  wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt?download=1 -O $work_dir/rnnt.pt
  # suggest you check run.sh to locate the dataset
  python pytorch/utils/download_librispeech.py \
         pytorch/utils/librispeech-inference.csv \
         $librispeech_download_dir \
         -e $local_data_dir --skip_download
  python pytorch/utils/convert_librispeech.py \
         --input_dir $local_dir/LibriSpeech/dev-clean \
         --dest_dir $local_data_dir/dev-clean-wav \
         --output_json $local_data_dir/dev-clean-wav.json
```
### 4. run rnnt
  Please update the setup_env_offline.sh or setup_evn_server.sh and user.conf according to your platform resource.
```
  export TCMALLOC_DIR=$CONDA_PREFIX/lib
  # offline
  sudo ./run_clean.sh
  source ./setup_env_offline.sh
  ./run_inference_cpu.sh
  # offline accuracy
  sudo ./run_clean.sh
  source ./setup_env_offline.sh
  ./run_inference_cpu.sh --accuracy
  # server scenario
  sudo ./run_clean.sh
  source ./setup_env_server.sh
  ./run_inference_cpu.sh --server
  # server accuracy
  sudo ./run_clean.sh
  source ./setup_env_server.sh
  ./run_inference_cpu.sh --accuracy
```
### Note on Server scenario
```
For server scenario, we exploit the fact that incoming data have different sequence lengths (and inference times) by bucketing according to sequence length 
and specifying batch size for each bucket such that latency can be satisfied. The settings are specified in machine.conf file and required fields 
are cores_per_instance, num_instances, waveform_len_cutoff, batch_size. 

```
