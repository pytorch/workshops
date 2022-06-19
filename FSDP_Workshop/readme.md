# t5_grammar_checker
T5 Grammar Checker model and workshop, trained using PyTorch FSDP

Goal - train a 800M / 3B / 11B T5 Model to serve as a grammar checker using PyTorch FSDP.  We'll cover both single node (one machine, multi-gpu) and multi-node (2+ machines, each with multi-gpus) training scenarios.  
Result - a trained model that can accept incoming sentences and correct a number of common grammar mistakes.

Examples:</br>
<img width="595" alt="3B_demo_grammar_samples" src="https://user-images.githubusercontent.com/46302957/172918714-8b11944c-0268-4de7-b120-1f993edeb35b.png">



## Getting Started - the environment

For single node, we'll use an A100 (p4dn on AWS) or V100 (p3* on AWS).  
For multi-node, we'll use AWS ParallelCluster and Slurm. 

## Single Node (one machine, multi-gpu): 

1 - Install this repo on your machine:
~~~
git clone ...
~~~

2 - Install the dependencies (cd into the above install folder):
~~~
pip install -r requirements.txt
~~~
You should receive something similar to:
~~~
Successfully installed datasets-2.2.2 dill-0.3.4 huggingface-hub-0.7.0 multiprocess-0.70.12.2 responses-0.18.0 tokenizers-0.12.1 transformers-4.19.3 xxhash-3.0.0
~~~~

3 - Uninstall any existing torch and torch libraries.  We will use the latest torch nightly build to ensure all FSDP features are available:
~~~
pip uninstall torch torchaudio torchvision -y 
~~~

Assuming Linux:
~~~~
pip3 install --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu113
~~~~
(or check for the command line needed for other OS at: https://pytorch.org/get-started/locally/ )

## Verify everything is working with benchmark
Let's run a simple benchmark of 3 epochs using t5-small to quickly confirm FSDP is working properly on your system before beginning training of the full size model.

We'll launch using a bash script which has all the parameters needed for torchrun.  It defaults to asssuming 8 GPU's.  Thus, if needed please adjust run_benchmark.sh to your total GPUs in the 'nproc per node' setting:
~~~
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=101 --rdzv_endpoint="localhost:5679" main_benchmark.py
~~~~

After confirming GPU count, you can run the benchmark directly with:
~~~
bash run_benchmark.sh
~~~

Training a large model:
Training of the full grammar checker will use the settings found in config/defaults.py.  
The default model is t5_1_1-large, or 737M parameter, but you are welcome to go larger or smaller depending on your single node capacity. You can adjust it easily in the defaults.py file:






