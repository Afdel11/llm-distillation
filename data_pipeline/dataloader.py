"""
datasets/dataloader.py
=======================
Dataset PyTorch pour les paires (prompt, réponse) tokenisées, avec un
collate_fn qui pad dynamiquement chaque batch (labels paddés à IGNORE_INDEX,
input_ids paddés au pad_token du tokenizer, attention_mask à 0 sur le padding).
"""

import torch
from torch.utils.data import Dataset

from data_pipeline.prompts import IGNORE_INDEX, build_example


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
            {**build_example(tokenizer, ex["prompt"], ex["response"], max_length=max_length), "idx": i}
            for i, ex in enumerate(examples)
        ]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


def make_collate_fn(pad_token_id: int):
    """Retourne un collate_fn qui pad dynamiquement au plus long élément du batch."""

    def collate_fn(batch: list) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)

        input_ids, attention_mask, labels, idxs = [], [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
            labels.append(ex["labels"] + [IGNORE_INDEX] * pad_len)
            idxs.append(ex["idx"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "idx": torch.tensor(idxs, dtype=torch.long),
        }

    return collate_fn


def make_cached_collate_fn(pad_token_id: int, teacher_logits_cache: dict):
    """
    Comme make_collate_fn, mais ajoute aussi "teacher_logits" au batch,
    en repêchant les logits pré-calculés (voir datasets/cache.py) via l'index
    de chaque exemple, et en les paddant à la même longueur que le batch.
    Le padding ajouté est arbitraire (zéros) : ces positions sont de toute
    façon masquées par IGNORE_INDEX dans les labels, donc ignorées par la loss.
    """
    base_collate_fn = make_collate_fn(pad_token_id)

    def collate_fn(batch: list) -> dict:
        out = base_collate_fn(batch)
        max_len = out["input_ids"].shape[1]

        padded_logits = []
        for i in out["idx"].tolist():
            logits_i = teacher_logits_cache[i]  # (seq_len_i, num_teachers, vocab_size)
            pad_len = max_len - logits_i.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, *logits_i.shape[1:], dtype=logits_i.dtype)
                logits_i = torch.cat([logits_i, pad], dim=0)
            padded_logits.append(logits_i)

        out["teacher_logits"] = torch.stack(padded_logits, dim=0).float()  # (batch, seq, teachers, vocab)
        return out

    return collate_fn
