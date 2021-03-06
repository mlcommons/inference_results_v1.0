
# Setup instructions

  

## Pre-docker instructions

Choose your workspace to be the directory where this README and the model scripts are located. Pull a copy of AIMET (dev-branch) into your workspace:

```
WORKSPACE=`pwd`
git clone https://github.com/quic/aimet.git
source $WORKSPACE/aimet/packaging/envsetup.sh
```
  

Ensure that mlcommons/inference is installed for utilizing model definitions and preprocessing routines. You can install a local copy:

`git clone https://github.com/mlcommons/inference.git`

  

Follow the instructions for building the AIMET docker [here](https://github.com/quic/aimet/blob/develop/packaging/docker_install.md#docker-information). Use the `latest` tag.

  

## Launching the docker

Make sure that no container with the same name is running (from previous runs):

`docker ps -a | grep ${docker_container_name} && docker kill ${docker_container_name}`

  

For cpu-only, run the AIMET docker with the command below. Add `-v` mount points to the command if your datasets and base ssd-resnet model are located outside of $HOME, $WORKSPACE, or /local/mnt/workspace.

  

```
docker run --rm -it -u $(id -u ${USER}):$(id -g ${USER}) \
-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
-v ${HOME}:${HOME} -v ${WORKSPACE}:${WORKSPACE} \
-v "/local/mnt/workspace":"/local/mnt/workspace" \
--entrypoint /bin/bash -w ${WORKSPACE} --hostname aimet-dev ${docker_image_name}
```

For gpu support, ensure that nvidia-docker is installed on the host. For more instructions, follow this [guide](https://github.com/quic/aimet/blob/develop/packaging/docker_install.md#start-docker-container-manually).

  

## Post-docker build instructions

After succesfully launching the docker, choose the same workspace:

`WORKSPACE=<path/to/workspace> && cd $WORKSPACE`

  

Follow these instructions for [building](https://github.com/quic/aimet/blob/develop/packaging/docker_install.md#build-code-and-install) the AIMET code and [setting paths](https://github.com/quic/aimet/blob/develop/packaging/docker_install.md#setup-paths).

  

# Usage instructions

  

## Generating AIMET-optimized model with encodings

To export the model to onnx and generate encodings, and depending on your mlcommons/inference installation location, you will need to update your path: 
```
export PYTHONPATH=$PYTHONPATH:<path_to_mlcommons>/inference/vision/classification_and_detection/python
```

  

A small change must be made to `inference/vision/classification_and_detection/python/coco.py`, since it depends on "pycoco" which is no longer available (it is available as pycocotools.coco). Comment out `import pycoco` in the file (line 15). This package is not needed for our purposes anyways as we do not run evaluation in our script.

  

The onnx model and encodings will be generated by the below command, which is formatted to expose the syntax.
```
python ssd_resnet_aimet.py <path/to/resnet34-ssd1200.pytorch> <path/to/calibration-annotations> <path/to/calibration-images>
```

The base model used in our experiments can be downloaded [here](https://zenodo.org/record/3236545/files/resnet34-ssd1200.pytorch). The outputs are saved as "ssd_resnet34_aimet.[onnx|pth|encodings|encodings.yaml]" to a timestamped directory under the "outputs" directory. Note that the pytorch file (i.e., .pth) is used for debugging. A parameters file is also generated: "ssd_resnet34_aimet_params.pkl". You can verify the parameters of your run by executing `python ssd_resnet34_aimet.py show-params <path/to/params-file.pkl>`.

  

# Details
These sections go over the concrete details of the environment, model parameters, preprocessing, AIMET optimization, encodings generation, as they were configured during our runs. Although we opt for Docker and try to match and freeze as many parameters as possible, discrepancies may still arise in other environments. So these sections are provided to help resolve potential discrepancies.
  
## Environment details
- onnx == 1.7.0
- torch == 1.4.0+cu100
- torchvision == 0.5.0+cu100 

## Preprocessing details

- input_shape = [1,3,1200,1200]
- dataset_name = 'coco-1200-pt'
- image_format = 'NCHW'
- pre_proc_func = pre_process_coco_resnet34
- cache = 0

Minor changes:
- Image file paths do not prepend "val2017" as in `image_name = os.path.join("val2017", img["file_name"])` (line 75 in coco.py).
- Dataset converts images to torch tensors rather than numpy arrays, instead of the backend being the object that converts images to torch tensors. In our case, there is no backend object.

## Model details

The base model used can be found [here](https://zenodo.org/record/3236545/files/resnet34-ssd1200.pytorch). The model is slightly modified, and the full details of the changes can be seen in the `modifications/ssd_r34_mod.py` file. Below is a summary of the changes.

- ResNet Bottleneck forward function changed to have multiple ReLU modules. This is so that AIMET generates a unique encoding for each ReLU.
- ResNet Block forward function changed to have multiple ReLU modules. This is so that AIMET generates a unique encoding for each ReLU.
- Concatenation modules instead of concatenation functionals. This is so that AIMET notices and generates encodings for these ops
- bbox_view method changed to call concatenation modules
- forward method changed to reflect all the above changes

## Export details

The exported ONNX model contains the ResNet34 backbone, SSD layers, and softmax. It does not contain ABP and NMS.

  

## Quantization details

The techniques used in these scripts:

- Cross Layer Equalization:
	- Batch Norm Folding
	- Cross Layer Scaling
	- High Bias Folding
- Encodings generation:
	- 'Tf' algorithm with all 500 calibration images used
	- Mixed encodings symmetry: uses both symmetric and asymmetric encodings
	- Nearest rounding
	- Mixed precision. All weights and activations are int8 with a few exceptions:
		- Bias parameters: All biases are encoded, and all encodings represent a 31 bit range.
		- Softmax activations: No encodings are generated by this script.
		- ABP+NMS: No encodings are generated by this script. There are no ABP or NMS modules as part of exported model.