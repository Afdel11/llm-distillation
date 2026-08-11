"""
tests/test_tokenizer.py
=========================
Teste le VRAI tokenizer Qwen2.5 (nécessite internet vers huggingface.co).
En sandbox de dev (sans internet), ces tests se marquent automatiquement
`skipped` plutôt que d'échouer bruyamment. Sur le GPU distant, ils
s'exécutent pour de vrai et valident le téléchargement.
"""

import pytest

from data_pipeline.tokenizer import get_tokenizer, DEFAULT_TOKENIZER_NAME


@pytest.fixture(scope="module")
def real_tokenizer():
    try:
        return get_tokenizer(DEFAULT_TOKENIZER_NAME)
    except Exception as e:
        pytest.skip(f"Tokenizer Qwen2.5 inaccessible (pas d'internet dans cet environnement) : {e}")


def test_tokenizer_loads(real_tokenizer):
    assert real_tokenizer.name_or_path == DEFAULT_TOKENIZER_NAME


def test_vocab_size_matches_qwen25_documentation(real_tokenizer):
    # Qwen2.5 documente un vocabulaire de 151 936 tokens.
    assert len(real_tokenizer) == 151936


def test_tokenizer_encodes_french_text(real_tokenizer):
    encoded = real_tokenizer("Bonjour, comment vas-tu ?")
    assert len(encoded["input_ids"]) > 0
