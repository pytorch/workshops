# CS7643 Deep Learing | Fall 2020 at Georgia Tech (Prof. Zsolt Kira)

This repo contains materials for students to learn PyTorch and help them through the various course modules for the Fall 2020 CS7643 deep learning course at Georgia Tech. Each of the modules below provides some selected content and notebooks that align with the analogous course module. Here are some general resources in addition:
- [PyTorch.org](https://pytorch.org/) - the general cite for all PyTorch docs, tutorials, educational pointers, installation and blogs. 
- [PyTorch Hub](https://pytorch.org/hub/) - a central place to find pretrained models targeting anything from audio to nlp to generative networks.
- [Tools & Libraries](https://pytorch.org/ecosystem/) - a place to learn about the community projects that are well tested, supported and include everything from medical imaging to operationalization of PyTorch models.
- [Discussion forums](https://discuss.pytorch.org/) - A community of over 31 thousand users and experts helping eachother use PyTorch.


## Usage Instructions 
1. Open [Colab](https://colab.research.google.com/)
2. Select 'GitHub' in the top selector
3. Type in 'pytorch' into the search
4. Select 'pytorch/workshops' in the repository field (branch should be master)
5. A number of notebooks should autopopulate, double click on a notebook to start a Colab session with that notebook
6. Once the notebook is open, you can go to menu and select 'Runtime'->'Change runtime type' and pick GPU. This is the prefered compute backend for PyTorch.

## Module 1: Introduction to Neural Networks

### Lessons covered:
- Lesson 1: Linear Classifiers and Gradient Descent
- Lesson 2: Matrix and Vector Calculus, Vectorization
- Lesson 3: Neural Networks
- Lesson 4: Backpropagation
- Lesson 5: Optimization Deep Neural Networks

### Learning objectives:
1. Understand machine learning components (data, features, loss functions, and regularization), and optimization (gradient descent)
2. Understand how deep learning differs from existing ML methods"
3. Have the ability to vectorize underlying machine learning computations
4. Understand and implement simple multi-layered classifiers
5. Understand backpropagation, explain its underlying concepts, and implement it
6. Understand PyTorch
7. Understand various elements of optimizing neural networks (initialization, normalization, regularization, etc.)

### Links to materials

1. [PyTorch Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html#pytorch-cheat-sheet) - A handy reference for many of the commonly used PyTorch APIs
2. [What is PyTorch Tutorial](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py) - a short tutorial that walks through tensor manipulation, numpy and using cuda tensors.
3. [Visualizing with Tensorboard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) - TensorBoard is a popular tool for visualizing embedding, loss curves, data and much more. 
4. [torch.nn tutorial showing NNs from scratch (by Jeremy Howard)](https://pytorch.org/tutorials/beginner/nn_tutorial.html) - walks through the various levels of abstraction for creating neural networks.
5. [Autograd: Automatic Differentiation walkthrough](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py) - a short primer on backpropogation and the power of autograd
6. [Neural Networks Intro](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py) - an intro tutorial on how to build a neural network program and the various components
7. [torch.optim API documentation](https://pytorch.org/docs/stable/optim.html) - How to use optimizers in PyTorch
8. [torch.nn.init walkthrough](https://pytorch.org/docs/stable/nn.init.html) - How to initialize neural networks in PyTorch

## Module 2: Convolutional Neural Networks

### Lessons covered:
- Lesson 6: Convolution and Pooling Layers
- Lesson 7: CNN Architectures
- Lesson 8: PyTorch and Scalable Training
- Lesson 9: Advanced Computer Vision Architectures

### Learning objectives:
1. Understand convolution and pooling layers
2. Understand current state of art CNN architectures
3. Understand PyTorch framework layout, usage, and tips/tricks
4. Understand PyTorch framework layout, usage, and tips/tricks

### Links to materials
1. [torch.nn.conv](https://pytorch.org/docs/stable/nn.html#convolution-layers) - API docs for conv layers
2. [torch.nn.pooling](https://pytorch.org/docs/stable/nn.html#pooling-layers) - API docs for pooling layers
3. [Training a Classifier Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) - Short tutorial on defining and training an image classifier. 
4. [Getting started with distributed data parallel training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) - A walkthrough of how to use distributed training on PyTorch

## Module 3: Recurrent Neural Networks & Sequence to Sequence Models

### Lessons covered:
- Lesson 10: Recurrent Neural Network Fundamentals
- Lesson 11: Long-Short Term Memory (LSTMs)
- Lesson 12: Natural Language Processing (NLP)

### Learning objectives:
1. Understand fundamentals of sequence-based problems and recurrent neural networks
2. Understand LSTMs
3. Understand attention mechanisms, the transformer architecture, and NLP applications

### Links to materials
1. [torch.nn recurrent layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers) - APIs docs for recurrent layers
2. [torch.nn.transformer layers](https://pytorch.org/docs/stable/nn.html#transformer-layers) - API docs for transformer layers
3. [torch.nn.MultiheadAttention](https://pytorch.org/docs/stable/nn.html?highlight=attention#torch.nn.MultiheadAttention) - API docs for Multihead Attention
4. [Sequence to sequence tutorial using torch.nn.transformer and multihead attention](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

## Module 4: Advanced Topics

### Lessons covered:
- Lesson 13: Unsupervised and Semi-Supervised Learning
- Lesson 14: Generative Models
- Lesson 15: Deep Reinforcement Learning (DRL)

### Learning objectives:
1. Understand unsupervised learning fundamentals and key methods used
2. Understand generative models
3. Understand basics of reinforcement learning

### Links to materials
1. 

