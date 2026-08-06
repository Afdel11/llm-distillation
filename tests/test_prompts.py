"""
tests/test_prompts.py
=======================
Nécessite le vrai tokenizer Qwen2.5 (internet). Skip automatique en sandbox.
"""

import pytest

from data_pipeline.prompts import TOY_EXAMPLES, IGNORE_INDEX, build_example
from data_pipeline.tokenizer import get_tokenizer, DEFAULT_TOKENIZER_NAME


@pytest.fixture(scope="module")
def real_tokenizer():
    try:
        return get_tokenizer(DEFAULT_TOKENIZER_NAME)
    except Exception as e:
        pytest.skip(f"Tokenizer Qwen2.5 inaccessible (pas d'internet dans cet environnement) : {e}")


def test_prompt_tokens_are_masked(real_tokenizer):
    example = build_example(real_tokenizer, TOY_EXAMPLES[0]["prompt"], TOY_EXAMPLES[0]["response"])
    n_prompt_tokens = len(real_tokenizer(TOY_EXAMPLES[0]["prompt"], add_special_tokens=False)["input_ids"])
    assert all(label == IGNORE_INDEX for label in example["labels"][:n_prompt_tokens])


def test_input_ids_and_labels_same_length(real_tokenizer):
    example = build_example(real_tokenizer, TOY_EXAMPLES[0]["prompt"], TOY_EXAMPLES[0]["response"])
    assert len(example["input_ids"]) == len(example["labels"]) == len(example["attention_mask"])
