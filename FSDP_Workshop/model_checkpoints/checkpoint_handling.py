from pathlib import Path


def prune_checkpoints(rank, dq, cfg):
    """ensures we only keep a fixed amount of model checkpoints during training"""
    if rank > 0:
        print(f"incorrect call to prune_checkpoints...only rank 0")
        return  # only rank 0 controls files to avoid issues

    if len(dq) > cfg.checkpoint_max_save_count:
        file_to_remove = Path(dq.popleft())
        file_to_remove.unlink()
        print(f"--> removed checkpoint {file_to_remove}")
