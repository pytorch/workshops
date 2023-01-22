# NeurIPS 2022 PyTorch Workshop: New advances for large-scale training and performance optimizations


Large language models and Generative AI have been key drivers for new innovations in large-scale training and performance optimizations. In this workshop, we will dive deeper into new features and solutions in PyTorch that enable training and performance optimizations @ scale. 

Following topics will be covered by the PyTorch team in this workshop. The sessions are divided over two days, Nov 28th will cover the PyTorch Distributed and Profiling topics, and Dec: 5th session will cover the PyTorch Compiler based solutions.

## Part 1: Nov 28 (Hybrid, in-person and remote), 9:30a-12:30p CST (UTC-6), Room # 291
-------------------------------------------------------------------------------------------------------

*1. FSDP Production Readiness, Speakers: Rohan Varma, Andrew Gu* [slides](FSDP%20-%20NeurIPS%2020222%20.pdf)

We will dive deep into recent advances in FSDP which have enabled better throughput, memory savings and extensibility. These improvements have unblocked using FSDP for models of different modalities and varying sizes(model and data). We will share best practices to apply these features to specific use cases such as XLMR, FLAVA, ViT, DHEN and GPT3 style models.

*2. Automated Pipeline Parallelism for PyTorch, Speaker: Ke Wen* [slides](PiPPy%20-%20NeurIPS%202022.pdf)

PiPPy is a library that provides automated pipeline parallelism for PyTorch models. PiPPy consists of a compiler stack capable of automatically splitting a model into stages without requiring intrusive code changes to the model. It also provides a distributed runtime that helps users to distribute the split stages to multiple devices and multiple hosts and orchestrates micro-batch execution in an overlapped fashion. We are going to demonstrate the use of PiPPy for Hugging Face models on clouds.

*3. PyTorch Profiler, Speaker: Taylor Robie*  [slides](PyTorch%20Profiler%20-%20NeurIPS%202022.pdf)

Dive into recent enhancements to the PyTorch profiler capabilities, Python function tracing, data flow capture, and memory profiling, and how they enable previously impossible performance analysis.

*4. Profiling Distributed Training Workloads, Speaker: Anupam Bhatnagar*  [slides](Profiling%20Distributed%20Training%20Workloads%20-%20NeurIPS%202022.pdf)

We will present Holistic Trace Analysis (HTA), a tool to identify computation, communication and memory bottlenecks in distributed training. HTA identifies these bottlenecks by analyzing the traces collected using the PyTorch Profiler.

*5. TorchBench, Speaker: Xu Zhao*  [slides](TorchBench%20-%20NeurIPS%202022.pdf)

In this talk we present PyTorch Benchmark(TorchBench), a benchmarking suite to provide quick and stable performance signals to hold the line of performance in PyTorch development. TorchBench identifies performance regressions and provides CI services for PyTorch developers to test their PRs. It can also be used to profile specific models and identify optimization opportunities.


## Part 2: Dec 7 (Virtual), 9:30a - 11:30a PST (UTC-8) / 11:30a - 1:30p CST (UTC-6)
------------------------------------------------------------------------------------------------
Focus on the new PyTorch Compiler features (https://pytorch.org/get-started/pytorch-2.0/)

*6. A deep dive into TorchDynamo,  Speaker: Animesh Jain*  [slides](TorchDynamo%20Deep%20Dive%20-%20NeurIPS%202022.pdf)

This talk presents a deep dive into TorchDynamo. TorchDynamo is a Python-level JIT compiler designed to make unmodified PyTorch programs faster. It rewrites Python bytecode in order to extract sequences of PyTorch operations into a graph which is then just-in-time compiled with a customizable backend. It is designed to mix Python execution with compiled backends to get the best of both worlds: usability and performance

*7. A deep dive into TorchInductor, Speakers: Bin Bao, Natalia Gimelshein*  [slides](TorchInductor%20Deep%20Dive%20-%20NeurIPS%202022.pdf)

This talk presents a deep dive into the design principles of TorchInductor, pytorch compiler backend, the lowering stack that it uses to transform pytorch programs, and the optimization techniques and codegen technologies that it uses. 

*8: How do backends integrate to PyTorch compiler stack, Speaker: Sherlock Huang*  [slides](PyTorch%202.0%20Backend%20Integration%20-%20NeurIPS%202022.pdf)

This talk deep dives into the backend integration points in Pytorch compiler stack. It will explain three types of IR used across the stack, torch IR produced by Dynamo, AtenIR produced by AoTAutograd, and loop-level IR used in Inductor. It will introduce the infrastructure and utilities available for backend integration, including a IR-agnostic Pattern Matcher and a Graph Partitioner.
