"""
datasets/cache.py
==================
Les Teachers sont gelés : recalculer leur forward pass à chaque epoch est un
gaspillage pur (voir la discussion coût computationnel du mémoire — le
forward des Teachers domine largement le coût d'ARCD lui-même). On les
calcule UNE FOIS, par exemple, à l'aide de TeacherEnsemble, et on les stocke.

Stockage en float16 (pas float32) pour diviser par 2 la taille sur disque :
le consensus robuste (médiane/MAD) n'a pas besoin de la pleine précision.

Format : un seul fichier .pt contenant un dict {idx: tensor(seq_len_i, num_teachers, vocab_size)}.
Suffisant pour un dataset de quelques dizaines de milliers d'exemples. Pour un
corpus beaucoup plus grand, il faudrait sharder (plusieurs fichiers) plutôt
que de charger un seul objet en mémoire — non implémenté ici, hors du
périmètre du mémoire (22 jours).
"""

import torch
from torch.utils.data import DataLoader


def build_teacher_cache(teacher_ensemble, dataset, pad_token_id: int, device: str = "cpu",
                         batch_size: int = 8) -> dict:
    """
    Calcule les logits Teachers pour tout le dataset et les retourne sous
    forme de dict {idx: logits (seq_len_i, num_teachers, vocab_size), float16}.

    Chaque exemple garde sa longueur réelle (pas de padding stocké) : le
    padding est réappliqué au moment du collate, batch par batch, via
    datasets.dataloader.make_cached_collate_fn — ce qui donne des logits
    identiques à un calcul non-batché (le padding ne change pas les logits
    des positions réelles, grâce à l'attention_mask causal).
    """
    from data_pipeline.dataloader import make_collate_fn

    collate_fn = make_collate_fn(pad_token_id=pad_token_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    cache = {}
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        teacher_logits = teacher_ensemble(input_ids, attention_mask)  # (batch, seq, teachers, vocab)

        for i, idx in enumerate(batch["idx"].tolist()):
            real_len = int(attention_mask[i].sum().item())
            cache[idx] = teacher_logits[i, :real_len].to(torch.float16).cpu()

    return cache


def save_teacher_cache(cache: dict, path: str):
    torch.save(cache, path)


def load_teacher_cache(path: str) -> dict:
    return torch.load(path, map_location="cpu")
