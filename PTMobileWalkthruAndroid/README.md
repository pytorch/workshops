# Android Hackathon Example

This project was made for the PyTorch Hackathon 2020, as a companion to a brief video showing how to stand up an Android project using the PyTorch Mobile runtime. You can view this video on the PyTorch YouTube channel: https://www.youtube.com/watch?v=O_2KBhkIvnc

## Project Setup

To set this project up and run it in Android Studio, clone this git repository, and open the `PTMobileWalkthruAndroid` folder as a project. With the project open you should be able to build and run the project.

## Model Preparation

The video describes a general process for getting your model into your project. If you'd like to try the pre-trained MobileNetV2 model used in the video, follow these steps in a Python environment with PyTorch 1.5 or higher and TorchVision 0.6 or higher installed:

1. `python` - open a Python REPL; the rest of the steps will be completed there
2. `import torch` and `import torchvision`
3. `model = torchvision.models.mobilenet_v2(pretrained=True)` will give you an instance of the model
4. **Optional:** Follow the model optimization procedure shown in the video
5. **Required:** `scripted_model = torch.jit.script(model)` will export your model to TorchScript
6. `torch.jit.save(scripted_model, 'model.pt')` saves your model to a file

For your convenience, copy the code below to test out the model creation:
```
import torch
import torchvision
model = torchvision.models.mobilenet_v2(pretrained=True)
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, 'model.pt')
```

## Project Notes

* This project is intended to be used with PyTorch 1.5 or higher, and Android Studio 3.5 or higher.
* If you want to upgrade the version of the PyTorch mobile runtime you're using, you can edit the `build.gradle` file for your app (not the file for the project).
* The first time you run the app and press the "Infer" button, it may take a while to load the model and run the inference. Subsequent inferences will be quicker.

**For more information on PyTorch Mobile for Android, visit the [PyTorch Mobile homepage](https://pytorch.org/mobile/home/).**