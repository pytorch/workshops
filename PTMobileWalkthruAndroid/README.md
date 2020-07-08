# Android Hackathon Example

This project was made for the PyTorch Hackathon 2020, as a companion to a brief video showing how to stand up an Android project using the PyTorch Mobile runtime. You can view this video on the PyTorch YouTube channel: https://www.youtube.com/watch?v=O_2KBhkIvnc

## Project Setup

To set this project up and run it in Android Studio, clone this git repository, and open the `PTMobileWalkthruAndroid` folder as a project. With the project open you should be able to build and run the project.

## Project Notes

* This project is intended to be used with PyTorch 1.5 or higher, and Android Studio 3.5 or higher.
* If you want to upgrade the version of the PyTorch mobile runtime you're using, you can edit the `build.gradle` file for your app (not the file for the project).
* The first time you run the app and press the "Infer" button, it may take a while to load the model and run the inference. Subsequent inferences will be quicker.

**For more information on PyTorch Mobile for Android, visit the [PyTorch Mobile homepage](https://pytorch.org/mobile/home/).**