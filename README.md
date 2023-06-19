# ESRGAN-ONNX

Inference-only ONNX implementation of ESRGAN.  
Requires minimal additional dependencies.

## Installation

To install it as a package:

Choose a version of onnxruntime suitable for your hardware.  
For CPU inference
 > pip install onnxruntime

For NVIDIA GPU  
 > pip install onnxruntime-gpu

Install wheel from the releases.  
 > pip install esrgan_onnx-0.1.0-py3-none-any.whl  

## Usage

The package only consists of 1 class, ESRGAN.  
Object's positional arguments are onnxruntime session, tile size, padding, and manual set scale.  
If manual scale is not set - on init it autodetects scale.  
**get** function accepts numpy image **array[height, width, channel]**.  


## Convert PyTorch models

I recommend [chaiNNer](https://chainner.app/)  
Download chaiNNer and run *PyTorch2ONNX_convert.chn*  
Sample Compact model is in releases.

## Test run

To determine optimal tile size for your image:  
 > git clone <https://github.com/fAIseh00d/ESRGAN-ONNX.git>

 Put sample or your own model in the script folder  
 > cd ESRGAN-ONNX  
 pip install -r requirements-run.txt  
 python run.py  

Please see parser arguments and tile_size list for reference.

## Reference

1. [ESRGAN](https://github.com/xinntao/ESRGAN)
2. [Model Database](https://upscale.wiki/wiki/Model_Database)  
3. [Original ESRGAN-ONNX repo](https://github.com/Sg4Dylan/ESRGAN-ONNX)
