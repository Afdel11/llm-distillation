"""
datasets/tokenizer.py
======================
UN SEUL tokenizer, partagé par tous les Teachers ET le Student. C'est ce
partage qui garantit que la position i du vecteur de logits désigne le
même token partout dans le pipeline ARCD (voir models/teacher.py).

Nécessite internet vers huggingface.co -> à charger sur le GPU distant.
"""

DEFAULT_TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # même famille que les Teachers


def get_tokenizer(name: str = DEFAULT_TOKENIZER_NAME):
    """
    Charge le tokenizer partagé. À appeler UNE SEULE FOIS et à réutiliser
    pour tokeniser les données ET pour dimensionner le Student
    (vocab_size = len(tokenizer)).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
