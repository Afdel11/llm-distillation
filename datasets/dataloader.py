"""
datasets/dataloader.py
=======================
Dataset PyTorch pour les paires (prompt, réponse) tokenisées, avec un
collate_fn qui pad dynamiquement chaque batch (labels paddés à IGNORE_INDEX,
input_ids paddés au pad_token du tokenizer, attention_mask à 0 sur le padding).
"""

import torch
from torch.utils.data import Dataset

from datasets.prompts import IGNORE_INDEX, build_example


class PromptResponseDataset(Dataset):
    """
    Args:
        examples:  liste de dicts {"prompt": str, "response": str}
        tokenizer: tokenizer partagé (voir datasets/tokenizer.py)
        max_length: longueur max de séquence (troncature)
    """

    def __init__(self, examples: list, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded = [
            build_example(tokenizer, ex["prompt"], ex["response"], max_length=max_length)
            for ex in examples
        ]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


def make_collate_fn(pad_token_id: int):
    """Retourne un collate_fn qui pad dynamiquement au plus long élément du batch."""

    def collate_fn(batch: list) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)

        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
            labels.append(ex["labels"] + [IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn
