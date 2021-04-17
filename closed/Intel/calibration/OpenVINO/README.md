#  Generic Setup for Model Quantization using OpenVINO
In order to generate quantized model, OpenVINO, Model Optimizer & Post-Training 
Optimization toolkit must be setup first. This explains general OpenVINO INT8  
workflow. Model dependent instructions are provided under specific model folder.
For example 3d_unet folder includes a dedicated README file.

#  OpenVINO Installation

OpenVINO toolkit must be installed first. First register [here](https://software.seek.intel.com/openvino-toolkit?os=windows), and proceed to download from OpenVINO 2020.4 from
[OpenVINO Download](https://registrationcenter.intel.com/en/products/postregistration/?sn=CN7V-MMZBCD5S&Sequence=2343869&encema=Wg/bUFJY2qspv9ef8QA1f1+hU9B0hD5Y1QPm/PLiHQtlzwS4UTT1JA4yxydzX5Dz5zNMC9cqbXaMhuZ4hj66XBWnKNxMliql9t9dW81KvP8=&dnld=t&pass=yes) (Make sure to select ```2020 04``` from the Choose Version drop menu.)

Follow the instructions at [OpenVINO Installation](https://docs.openvinotoolkit.org/2020.4/openvino_docs_install_guides_installing_openvino_linux.html) to make sure OpenVINO
is fully setup including its dependencies.Please note that OpenVINO needs Python
64-bit version 3.6 or higher. It warns if the required specifications are not met.

To be able to run inference benchmarks using OpenVINO, an internal representation 
of the model must be obtained first. This is handled by Model Optimizer.

# Setting up OpenVINO Model Optimizer & Generating Intermediate Representation (IR)

In order to have a better understanding of Model Optimizer, please have a look at
[Model Optimizer](https://docs.openvinotoolkit.org/2020.4/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html).

Follow the instructions at [Model Optimizer Configuration](https://docs.openvinotoolkit.org/2020.4/openvino_docs_MO_DG_prepare_model_Config_Model_Optimizer.html) to configure the
Model Optimizer.

When running install_prerequisites or any other framework specific install_prerequisites,
please pay special attention to compatibility issues of some of installed packages
including numpy package. Make sure you install a compatible version of the package.

# Setting up Post-Training Optimization Toolkit
To setup Post-Training Optimization, follow the instructions presented at 
[Post Training Optimization](https://docs.openvinotoolkit.org/2020.4/pot_README.html).
Please note that you need to install accuracy checker before using optimization toolkit
as explained at [Post Training Optimization](https://docs.openvinotoolkit.org/2020.4/pot_README.html).
