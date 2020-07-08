# iOS Hackathon Example

This project was made for the PyTorch Hackathon 2020, as a companion to a brief video showing how to stand up an iOS project using the PyTorch Mobile runtime. You can view this video on the PyTorch YouTube channel: https://www.youtube.com/watch?v=JFy3uHyqXn0

## Project Setup

To set this project up and run it in Xcode:

1. Clone this git repository.
2. From the command line, `cd` to the `workshops/PTMobileWalkthruIOS` folder.
3. Run `pod install` to get project dependencies.
4. Open `PTMobileDemo.xcworkspace` in Xcode.
5. Build and run the example in Xcode.

## Project Notes

* This project is intended to be used with PyTorch 1.5 or higher, and Xcode 11.4 or higher.
* If you want to upgrade the version of the PyTorch mobile runtime you're using, you can use the `pod update` command.
* The first time you run the app and press the "Infer" button, it may take a while to load the model and run the inference. Subsequent inferences will be quicker.

**For more information on PyTorch Mobile for iOS, visit the [PyTorch Mobile homepage](https://pytorch.org/mobile/home/).**