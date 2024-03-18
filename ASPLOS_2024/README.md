PyTorch 2 Tutorial - ASPLOS 2024 Workshop
=========================================
*Peng Wu, Jason Ansel, Horace He, Animesh Jain, Mario Lezcano, Mario Lezcano, Peter Bell, Avik Chaudhuri, Bin Bao, Brian Hirsh, Elias Ellison Yang Chen*

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

8:00am - 8:30am: The Road to PyTorch 2:

| Time | Sessions | Speakers |
| ------------- | ------------- | ------------- |
| 8:00am - 8:30am  | The Road to PyTorch 2  | Peng Wu  |
| 8:30am - 10:00am | PT2 Graph Capturing: TorchDynamo  | Animesh Jain, Mario Lezcano, Michael Lazos |
| 10:00am - 10:30am | Morning Break | |
| 10:30am - Noon| PT2 Compiler: TorchInductor | Horace He, Peter Bell |
| Noon - 1:30pm | Lunch | |
| 1:30pm - 2:15pm  | PT2 Export Path and Ahead-Of-Time Compilation | Avik Chaudhuri, Bin Bao  |
| 2:15pm - 3:00pm | Tensor Subclass Integration in PT2 | Brian Hirsh, Elias Ellison |
| 3:00pm - 3:30pm | Afternoon break | |
| 3:30pm - 4:15 pm | Performance Benchmarking and Tuning | Bin Bao, Yang Chen |
| 4:15pm - 5pm | Crazy Research Ideas | Jason Ansel, Horace He |

### References

PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation [Paper](https://pytorch.org/assets/pytorch2-2.pdf)
