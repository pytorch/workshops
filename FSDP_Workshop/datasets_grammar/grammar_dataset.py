import argparse
import csv
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from pathlib import Path


class grammar(Dataset):
    def __init__(
        self,
        tokenizer,
        type="train",
        csv_name=None,
        num_samples=None,
        input_length=512,
        output_length=512,
        print_text=False,
    ):

        self.dataset = load_dataset(
            "csv",
            data_files={"train": [csv_name]},  # "eval": "grammar_validation.csv"},
            delimiter=",",
        )

        # self.dataset = load_dataset("wikihow", "all", data_dir="data/", split=type_path)
        # if num_samples:
        #    self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = False  # print_text

    def __len__(self):
        return self.dataset["train"].shape[0]

    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)

        # model_inputs = self.tokenizer(
        #    example_batch["input"], max_length=512, truncation=True
        # )

        # Setup the tokenizer for targets
        # with self.tokenizer.as_target_tokenizer():
        #    labels = self.tokenizer(
        #       example_batch["target"], max_length=512, truncation=True
        #    )

        # model_inputs["labels"] = labels["input_ids"]
        # print(model_inputs[0])
        # return model_inputs

        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch["text"]))
        #         input_ = self.clean_text(example_batch['text']) + " </s>"
        #         target_ = self.clean_text(example_batch['headline']) + " </s>"

        input_ = example_batch["input"]
        target_ = example_batch["target"]

        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        targets = self.tokenizer.batch_encode_plus(
            [target_],
            max_length=self.output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return source, targets  # , model_inputs

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset["train"][index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }


def get_dataset(
    tokenizer, csv_name=None, num_samples=None, input_length=512, output_length=512
):
    """cover function for handling loading the working dataset"""
    """dataset loading"""
    if csv_name is None:
        currPath = Path.cwd() / "datasets_grammar" / "grammar_train.csv"
        print(f"Loading dataset {currPath}")
        csv_name = str(currPath)

    return grammar(
        tokenizer=tokenizer,
        csv_name=csv_name,
        num_samples=num_samples,
        input_length=input_length,
    )
