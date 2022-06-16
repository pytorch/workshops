from dataclasses import dataclass


@dataclass
class benchmark_config:
    # general
    host_port: str = "12368"

    # model
    model_name = "t5-small"  # "google/t5-v1_1-small"
    tokenizer = "t5-large"
    # available models
    ## t5-base
    # google/t5-v1_1-small
    # google/t5-v1_1-base
    # google/t5-v1_1-large
    # google/t5-v1_1-xl  #3b
    # google/t5-v1_1-xxl #11b
    save_model: bool = False
    model_checkpoint = "t5_small_save.pt"
    print_sharding_plan: bool = False

    # dataloaders
    num_workers_dataloader: int = 0

    # policies
    fsdp_unit_size = 1000000
    use_mixed_precision: bool = False
    use_fp16: bool = False

    hf_activation_checkpointing: bool = False
    fsdp_activation_checkpointing: bool = False

    # datasets
    dataset_train = "datasets_grammar/grammar_train.csv"  # grammar_13k.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = 4
    num_epochs: int = 3

    # validation
    run_validation: bool = True
    val_batch_size = 4
    block_for_validation: bool = False

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distributed_debug: bool = True

    # Fine Tuning
    use_child_tuning: bool = False

    use_task_free: bool = False
    use_fisher_matrix: bool = False
    percent_F: float = 0.3
