"""
tests/conftest.py
==================
Fixtures partagées par tous les fichiers de test. pytest les découvre
automatiquement, aucun import explicite nécessaire dans les tests.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class DebugTokenizer:
    """Tokenizer minimal, 100% local (aucun réseau requis) : un "token" par
    mot, id déterministe basé sur la longueur du mot. Sert uniquement à
    tester le padding/masquage indépendamment du vrai tokenizer Qwen."""

    def __init__(self):
        self.eos_token_id = 1
        self.pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = [min(len(w), 50) + 2 for w in text.split()]
        return {"input_ids": ids}


@pytest.fixture
def debug_tokenizer():
    return DebugTokenizer()
