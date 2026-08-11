from data_pipeline.dataloader import PromptResponseDataset, make_collate_fn


EXAMPLES = [
    {"prompt": "Question courte", "response": "Réponse brève ici"},
    {"prompt": "Question un peu plus longue avec plus de mots", "response": "Réponse"},
]


def test_dataset_builds_expected_number_of_examples(debug_tokenizer):
    dataset = PromptResponseDataset(EXAMPLES, debug_tokenizer, max_length=32)
    assert len(dataset) == len(EXAMPLES)


def test_prompt_tokens_are_masked(debug_tokenizer):
    dataset = PromptResponseDataset(EXAMPLES, debug_tokenizer, max_length=32)
    example = dataset[0]
    n_prompt_tokens = len(debug_tokenizer(EXAMPLES[0]["prompt"])["input_ids"])
    assert all(label == -100 for label in example["labels"][:n_prompt_tokens])


def test_collate_fn_pads_to_longest_in_batch(debug_tokenizer):
    dataset = PromptResponseDataset(EXAMPLES, debug_tokenizer, max_length=32)
    collate_fn = make_collate_fn(pad_token_id=debug_tokenizer.pad_token_id)

    batch = collate_fn([dataset[0], dataset[1]])

    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert batch["input_ids"].shape[0] == 2
