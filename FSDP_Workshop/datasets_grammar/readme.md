# T5 Grammar Checker - Datasets!
T5 Grammar Checker model and workshop, trained using PyTorch FSDP

This directory contains ready to use grammer datasets based on JFLEG and C4_200M synthetic dataset. 
The gtrain_150K contains 150K samples from C4 and 13K from JFLEG. 

You can expand your dataset up to 200M by using the included pull and process data notebook.  This will walk you through the steps to preprocess and ultimately merge and save a custom dataset of the size needed, up to 200M!

# Dataset source credits:
We've used two datasets for training our grammar checker.

1 - JFLEG:

Courtney Napoles, Keisuke Sakaguchi and Joel Tetreault. (EACL 2017): JFLEG: A Fluency Corpus and Benchmark for Grammatical Error Correction. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics. Valencia, Spain. April 03-07, 2017. https://github.com/keisks/jfleg

and downloadable from: https://huggingface.co/datasets/jfleg

We've included the processed/cleaned up dataset as train and test CSV files already as grammar_train.csv and grammar_validation.csv.

2 - C4 Synthetic grammar data:

This is a massive synthetic dataset of 200M. We only pull a subset of 150K, but show you how to create a much larger dataset as you wish in the dataset creation notebook (under datasets_grammer folder).

General info: https://ai.googleblog.com/2021/08/the-c4200m-synthetic-dataset-for.html

Github repo: https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction

Paper and credit: https://aclanthology.org/2021.bea-1.4/

