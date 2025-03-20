import os
import numpy as np
import cv2
import torch
import random
import pandas as pd
import pytorch_lightning as pl

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)

def split_into_folds(df: pd.DataFrame, n_splits: int, seed: int = 42) -> pd.DataFrame:
    """
    Splits the CSV data into n_splits folds based on episode_id and adds a new fold column.

    Args:
        df (pd.DataFrame): A DataFrame containing an episode_id column.
        n_splits (int): The number of folds to split the data into.
        seed (int): A seed value for consistency in results.

    Returns:
        pd.DataFrame: A DataFrame with an additional fold column.
    """
    # Get a unique list of episode_id values
    unique_episodes = df["episode_id"].unique()
    
    # Generate consistent hash values
    np.random.seed(seed)  # シードを設定
    hashed_episodes = np.array([hash(ep) for ep in unique_episodes])
    
    # Sort for consistency (optional)
    sorted_indices = np.argsort(hashed_episodes)
    sorted_episodes = unique_episodes[sorted_indices]

    # Evenly distribute folds
    folds = np.arange(len(sorted_episodes)) % n_splits
    
    # Add the fold column to the DataFrame
    episode_to_fold = dict(zip(sorted_episodes, folds))
    df["fold"] = df["episode_id"].map(episode_to_fold)
    
    return df