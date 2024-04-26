PyTorch 2 Tutorial and Paper presentation @ ASPLOS'2024
=======================================================
*Peng Wu, Jason Ansel, Horace He, Animesh Jain, Mario Lezcano, Michael Lazos, Peter Bell, Avik Chaudhuri, Bin Bao, Brian Hirsh, Elias Ellison Yang Chen, Boyuan Feng*

## PyTorch 2 Tutorial

### Date:
Full-day on April 27 (Saturday), 2024


### Summary:
PyTorch 2 marks a significant leap for the popular open-source machine learning framework. Announced in December 2022, it prioritizes accelerating deep learning models in both training and inference. This tutorial delves into the core components that drive PyTorch 2's performance improvements.

### TorchDynamo: Efficient JIT Compilation
At the heart of PyTorch 2 lies TorchDynamo, a Python-level just-in-time (JIT) compiler. It empowers you to compile graphs within your PyTorch programs without compromising Python's flexibility. This feat is achieved through dynamic bytecode modification before execution. TorchDynamo extracts sequences of PyTorch operations and constructs an FX graph, which is then JIT-compiled using a choice of extensible backends.

### TorchInductor: Optimizing for Hardware
TorchInductor acts as the default compiler backend for TorchDynamo. It optimizes and translates PyTorch programs into hardware-specific representations. For GPUs, it leverages OpenAI's Triton, while CPU programs are translated into C++.

### TorchExport: Unleashing Model Portability
TorchExport empowers you to export PyTorch models into standardized formats. These formats enable seamless execution in environments beyond Python. AOTInductor, a specialized mode within TorchInductor, takes this a step further. It compiles exported models into binaries, allowing them to run in Python-less environments.

### Benchmarking: A Foundation for Progress
PyTorch 2's capabilities are rigorously evaluated using a comprehensive suite of over 180 real-world models. This benchmark set not only validates PyTorch 2's performance but also establishes a valuable platform for assessing future research advancements in machine learning frameworks.

This tutorial equips researchers with a thorough understanding of PyTorch 2's internal workings. By delving into these mechanisms, researchers can unlock the full potential of PyTorch 2 as a platform and push the boundaries of what's possible in machine learning frameworks.


### Agenda

April 27th, 2024 (Saturday)

| Time | Sessions | Speakers | Slides and Notebooks |
| ------------- | ------------- | ------------- | ------------- |
| 8:00am - 8:15am  | The Road to PyTorch 2 | Peng Wu | |
| 8:15am - 10:00am | PT2 Graph Capturing: TorchDynamo and AOTAutograd | Animesh Jain, Mario Lezcano, Brian Hirsh, Michael Lazos | [notebook](https://colab.research.google.com/drive/19JURKGhy_L82Y-2MUc2jurJwARCPy-YL?usp=sharing) |
| 10:00am - 10:30am | Morning Break | | |
| 10:30am - Noon| PT2 Compiler: TorchInductor | Horace He, Peter Bell | |
| Noon - 1:30pm | Lunch | | |
| 1:30pm - 2:15pm  | PT2 Export Path and Ahead-Of-Time Compilation | Avik Chaudhuri, Bin Bao | [notebook](https://colab.research.google.com/drive/1YoKqydw3PmbTSwKCSEb4Ao4D55o4AII8?usp=sharing) |
| 2:15pm - 3:00pm | Tensor Subclass Integration in PT2 | Brian Hirsh, Elias Ellison | |
| 3:00pm - 3:30pm | Afternoon break | | |
| 3:30pm - 4:15 pm | Performance Benchmarking and Tuning |  Yang Chen, Boyuan Feng | [notebook](https://colab.research.google.com/drive/1XQwio7DsqB5LP2D574f_uIb8G7KhirNa?usp=sharing) |
| 4:15pm - 5pm | Crazy Research Ideas | Jason Ansel, Horace He | |

## PyTorch 2 Paper

The [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation Paper](https://pytorch.org/assets/pytorch2-2.pdf) will also be presented at the [ASPLOS'24 Conference](https://www.asplos-conference.org/asplos2024/main-program/) (14:30 PDT – 15:30 PDT: Session 6C: Optimization of Tensor Programs, Tuesday, April 30, 2024)
