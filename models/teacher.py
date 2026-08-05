"""
models/teacher.py
==================
TeacherEnsemble : charge N Teachers Qwen2.5-Instruct (même famille, donc
même tokenizer — voir datasets/tokenizer.py), les gèle, et expose leurs
logits empilés dans la convention ARCD : (batch, seq_len, num_teachers, vocab_size).

Convention stricte, documentée dans README.md : ce module ne renvoie QUE des
logits bruts, jamais des probabilités. Le softmax (avec température) est
centralisé dans arcd/consensus.py — un seul endroit où la température peut
être mal appliquée plutôt que deux qui pourraient diverger silencieusement.
"""

import torch
import torch.nn as nn

DEFAULT_TEACHER_NAMES = {
    "large": "Qwen/Qwen2.5-1.5B-Instruct",
    "small": "Qwen/Qwen2.5-0.5B-Instruct",
}


class TeacherEnsemble:
    """
    Args:
        model_names: dict {nom_court: nom_hf} ex. {"large": "Qwen/Qwen2.5-1.5B-Instruct"}.
                     Si None, utilise DEFAULT_TEACHER_NAMES.
        device:      "cuda" ou "cpu".
        dtype:       torch.bfloat16 recommandé sur GPU (poids natifs Qwen2.5),
                     torch.float32 par défaut sur CPU (bf16 n'accélère rien sur CPU).

    Usage:
        ensemble = TeacherEnsemble(cfg["models"]["teacher_names"], device="cuda")
        teacher_logits = ensemble(input_ids, attention_mask)  # (batch, seq, N, vocab)
        ensemble.teacher_names   -> ["large", "small"]
        ensemble.parameter_counts() -> {"large": 1_500_000_000, "small": 500_000_000}
    """

    def __init__(self, model_names: dict = None, device: str = "cpu",
                 dtype: torch.dtype = None):
        from transformers import AutoModelForCausalLM

        self.model_names = model_names or DEFAULT_TEACHER_NAMES
        self.device = device
        self.dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)

        self.models = {}
        for short_name, hf_name in self.model_names.items():
            model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=self.dtype)
            model.to(device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self.models[short_name] = model

    @property
    def teacher_names(self) -> list:
        return list(self.models.keys())

    def parameter_counts(self) -> dict:
        return {name: sum(p.numel() for p in m.parameters()) for name, m in self.models.items()}

    def to(self, device: str):
        self.device = device
        for m in self.models.values():
            m.to(device)
        return self

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            (batch, seq_len, num_teachers, vocab_size), dans l'ordre de self.teacher_names.
        """
        all_logits = []
        for name in self.teacher_names:
            out = self.models[name](input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(out.logits.float())  # remonte en float32 pour la stabilité d'ARCD
        return torch.stack(all_logits, dim=2)  # insère l'axe Teachers en position -2


# ---------------------------------------------------------------------------
# Version "debug" — AUCUN accès réseau requis (poids aléatoires, vocabulaire
# factice), pour les tests locaux. Toujours en float32 : bf16 sur CPU ne sert
# à rien et certaines opérations (ex: torch.sort) sont plus lentes en bf16 CPU.
# ---------------------------------------------------------------------------

class DebugTeacherEnsemble:
    """Même interface que TeacherEnsemble, mais 100% locale (GPT-2 miniatures)."""

    def __init__(self, vocab_size: int = 1000, num_teachers: int = 2):
        from transformers import GPT2Config, GPT2LMHeadModel

        self.models = {}
        for i in range(num_teachers):
            n_embd = 64 if i == 0 else 32  # tailles différentes -> vraie diversité
            config = GPT2Config(vocab_size=vocab_size, n_embd=n_embd, n_layer=2,
                                 n_head=2, n_positions=64)
            model = GPT2LMHeadModel(config)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self.models[f"debug_{i}"] = model

    @property
    def teacher_names(self) -> list:
        return list(self.models.keys())

    def parameter_counts(self) -> dict:
        return {name: sum(p.numel() for p in m.parameters()) for name, m in self.models.items()}

    def to(self, device: str):
        for m in self.models.values():
            m.to(device)
        return self

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        all_logits = [self.models[name](input_ids=input_ids, attention_mask=attention_mask).logits
                      for name in self.teacher_names]
        return torch.stack(all_logits, dim=2)
