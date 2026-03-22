# GGMLSharp Introduction

GGMLSharp is an API for C# to use [ggml](https://github.com/ggerganov/ggml).</br>

``` commandline
git submodule update --init --recursive
git submodule init 
git submodule update --remote --merge
```

ggml is a wonderful C-language machine-learning tool, and now you can use it with C#.</br>
GGMLSharp contains all ggml shared libs and some demos.

``` 
mkdir build
cd build
cmake ../
cmake --build . --config Release

```


## Feature

- Written in C# only
- Only depends on ggml
- All Demos can use safe code only!
- Support for Convolution operations (1D, Depthwise, Transpose)
- Support for Activation functions (Sigmoid, Tanh)
- Dynamic backend selection (CPU/CUDA)


### mnist_cpu

  [mnist_cpu](./Demos/MNIST_CPU/) is a basic demo for learning how to use GGMLSharp. It contains two Linears.
  
### mnist_cnn

  [mnist_cnn](./Demos/MNIST_CNN/) is a demos show how to use convolution. In this demo, there are two conv2d and pool max.  
  Now also includes support for 1D convolution operations.

### mnist_train

  [mnist_train](./Demos/MNIST_Train/) is a demo shows how to train a model. The model is same as mnist_cpu.

### simple_backend

  [simple_backend](./Demos/SimpleBackend/) shows how to use GGMLSharp with cuda. In this demo, you shold take ggml.dll for cuda. You can get it with the help of [ggml](https://github.com/ggerganov/ggml) or you can download it from [llama.cpp](https://github.com/ggerganov/llama.cpp/releases).

### magika

[magika](./Demos/Magika/) is a useful tool from google. It can help to get the style of a file in high speed.

### Converter

[Converter](./Demos/Converter/) is a useful tool for converting llm models from bin/ckpt/safetensors to gguf without any python environment. 

### ModelLoader

[ModelLoader](./Demos/ModelLoader/) is a tool for loading safetensors or pickle file directly from binary data. This demo can help to learn how to read a model file without any help of python.

### SAM

[SAM](./Demos/SAM/) (Segment Anything Model) can help us seprate things from an image.

### TestOpt

[TestOpt](./Demos/TestOpt/) is a basic demo for optimizar.

### Yolov3Tiny

  [Yolov3Tiny](./Demos/Yolov3Tiny/) is a Demo shows how to implement YOLO object detection with ggml using pretrained model. The weight have been converted to gguf.

## Tests

The test suite is organized into a few broad groups:

- **Convolution and pooling** - 1D/2D convolution, depthwise convolution, transpose convolution, and pooling-related coverage.
- **Tensor and operator behavior** - tensor layout and transformation operations such as `cont`, `dup`, `roll`, padding, interpolation, and custom operators.
- **Backend and execution** - backend-related execution paths and compute behavior on supported backends.
- **Quantization and optimizer** - quantization helpers, quantization performance checks, and optimizer-related coverage.
- **Embedding and positional ops** - arange, relative position, timestep embedding, and similar utility operators.

Run tests:
```bash
cd Tests
dotnet run
```
