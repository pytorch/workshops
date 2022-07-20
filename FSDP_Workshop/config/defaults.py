# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy


@dataclass
class train_config:
    # general
    host_port: str = "12368"

    # seed
    seed: int = 2022
    
    
    # model
    model_name = "google/t5-v1_1-large"  # << - adjust model size here
    
    # available models
    # google/t5-v1_1-small  # 60 M
    # google/t5-v1_1-base   # 223 M
    # google/t5-v1_1-large  # 737 M
    # google/t5-v1_1-xl     # 3 Billion
    # google/t5-v1_1-xxl    # 11 Billion 

    tokenizer = "t5-large"   # no need to adjust, tokenizer works for all model sizes

    # save models
    save_model: bool = True
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )


    # sharding policy
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    print_sharding_plan: bool = False

    # dataloaders
    num_workers_dataloader: int = 0

    # policies - this will default to BF16, but if no native support detected, will 
    # use FP16.  (note that FP16 is not recommended for larger models...)

    use_mixed_precision: bool = True

    HF_activation_checkpointing: bool = True
    FSDP_activation_checkpointing: bool = False

    # datasets
    dataset_train = "datasets_grammar/grammar_train.csv" # gtrain_150K.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = 8
    num_epochs: int = 8

    # validation
    run_validation: bool = True
    val_batch_size = 8
    block_for_validation: bool = False

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distributed_debug: bool = True

    # Fine Tuning
    use_child_tuning: bool = True
    lr: float = 4e-8

    use_task_free: bool = True
    use_fisher_matrix: bool = False
    percent_F: float = 0.35
