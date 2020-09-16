# iOS Hackathon Example

This project was made for the PyTorch Hackathon 2020, as a companion to a brief video showing how to stand up an iOS project using the PyTorch Mobile runtime. You can view this video on the PyTorch YouTube channel: https://www.youtube.com/watch?v=JFy3uHyqXn0

## Project Setup

To set this project up and run it in Xcode:

1. Clone this git repository.
2. From the command line, `cd` to the `workshops/PTMobileWalkthruIOS` folder.
3. Run `pod install` to get project dependencies.
4. Open `PTMobileDemo.xcworkspace` in Xcode.
5. Build and run the example in Xcode.

## Model Preparation

The video describes a general process for getting your model into your project. If you'd like to try the pre-trained MobileNetV2 model used in the video, follow these steps in a Python environment with PyTorch 1.6 or higher and TorchVision 0.7 or higher installed:

1. `python` - open a Python REPL; the rest of the steps will be completed there
2. `import torch` and `import torchvision`
3. `model = torchvision.models.mobilenet_v2(pretrained=True)` will give you an instance of the model
4. **Optional:** Follow the model optimization procedure shown in the video
5. **Required:** `scripted_model = torch.jit.script(model)` will export your model to TorchScript
6. `torch.jit.save(scripted_model, 'model.pt')` saves your model to a file

You should now have a file called `model.pt` that you can use in your mobile project.

For your convenience, copy the code below and run in Python with PyTorch 1.6 to test out the model creation:
```
import torch
import torchvision
model = torchvision.models.mobilenet_v2(pretrained=True)
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, 'model.pt')
```

## Project Notes

* This project has been updated and tested with PyTorch 1.6, and Xcode 11.7.
* If you want to upgrade the version of the PyTorch mobile runtime you're using, you can use the `pod update` command.
* The first time you run the app and press the "Infer" button, it may take a while to load the model and run the inference. Subsequent inferences will be quicker.

**For more information on PyTorch Mobile for iOS, visit the [PyTorch Mobile homepage](https://pytorch.org/mobile/home/).**
