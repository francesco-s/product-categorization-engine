# src/data/dataset.py
from typing import Dict, List, Any, Optional, Union
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizer


class ProductDataset(Dataset):
    """
    PyTorch Dataset for product categorization.
    """

    def __init__(
            self,
            texts: Union[List[str], pd.Series],
            labels: Union[List[int], pd.Series],
            tokenizer: PreTrainedTokenizer,
            max_length: int = 128
    ):
        """
        Initialize dataset.

        Args:
            texts: List of product text strings (preprocessed)
            labels: List of category labels (encoded as integers)
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
        """
        # Convert to list if Series to avoid indexing issues
        if isinstance(texts, pd.Series):
            self.texts = texts.values.tolist()
        else:
            self.texts = texts

        if isinstance(labels, pd.Series):
            self.labels = labels.values.tolist()
        else:
            self.labels = labels

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Get dataset length.

        Returns:
            Number of examples
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Extract tensors and squeeze batch dimension
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        return item


class ProductInferenceDataset(Dataset):
    """
    PyTorch Dataset for product categorization inference.
    """

    def __init__(
            self,
            texts: Union[List[str], pd.Series],
            tokenizer: PreTrainedTokenizer,
            max_length: int = 128
    ):
        """
        Initialize dataset.

        Args:
            texts: List of product text strings (preprocessed)
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
        """
        # Convert to list if Series to avoid indexing issues
        if isinstance(texts, pd.Series):
            self.texts = texts.values.tolist()
        else:
            self.texts = texts

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """
        Get dataset length.

        Returns:
            Number of examples
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary with input_ids and attention_mask
        """
        text = str(self.texts[idx])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Extract tensors and squeeze batch dimension
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

        return item


def create_data_loaders(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        text_column: str = 'combined_text',
        label_column: str = 'category_id',
        batch_size: int = 32,
        max_length: int = 128
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create PyTorch DataLoaders from dataframes.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        tokenizer: Tokenizer for encoding texts
        text_column: Name of the text column
        label_column: Name of the label column
        batch_size: Batch size
        max_length: Maximum sequence length

    Returns:
        Dictionary with train_loader, val_loader, and test_loader
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = ProductDataset(
        train_df[text_column],
        train_df[label_column],
        tokenizer,
        max_length
    )

    val_dataset = ProductDataset(
        val_df[text_column],
        val_df[label_column],
        tokenizer,
        max_length
    )

    test_dataset = ProductDataset(
        test_df[text_column],
        test_df[label_column],
        tokenizer,
        max_length
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # Add DataLoader configurations for better handling
        num_workers=0,  # Single process loading to avoid issues
        pin_memory=False,  # No GPU memory pinning
        drop_last=False  # Keep partial batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }
