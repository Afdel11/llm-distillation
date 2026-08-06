"""
datasets/prompts.py
====================
Mise en forme prompt/réponse et construction des labels masqués : on ne
veut faire prédire au Student QUE les tokens de la réponse, pas ceux du
prompt (convention standard du fine-tuning instructif).

Jeu d'exemples minimal fourni ici pour amorcer le pipeline. À remplacer par
un vrai dataset (ex: un sous-ensemble d'Alpaca, de Dolly, ou tes propres
paires prompt/réponse) sur le GPU distant.
"""

TOY_EXAMPLES = [
    {"prompt": "Question: Quelle est la capitale de la France ?\nRéponse:",
     "response": " La capitale de la France est Paris."},
    {"prompt": "Question: Combien font 2 plus 2 ?\nRéponse:",
     "response": " 2 plus 2 font 4."},
    {"prompt": "Question: Quel est le rôle d'un Teacher en distillation de connaissances ?\nRéponse:",
     "response": " Un Teacher fournit une supervision plus riche qu'une simple étiquette, "
                  "que le Student apprend à imiter."},
]

IGNORE_INDEX = -100


def build_example(tokenizer, prompt: str, response: str, max_length: int = 128) -> dict:
    """
    Tokenise (prompt + réponse) et construit les labels avec le prompt
    masqué à IGNORE_INDEX.

    Returns:
        dict avec input_ids, attention_mask, labels (listes d'entiers)
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id

    input_ids = prompt_ids + response_ids + [eos_id]
    input_ids = input_ids[:max_length]

    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids + [eos_id]
    labels = labels[:max_length]

    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
