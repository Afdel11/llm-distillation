#!/usr/bin/env python3
"""
scripts/check_tokenizer_compatibility.py
===========================================
Vérifie que deux modèles partagent EXACTEMENT le même tokenizer (même
taille ET même mapping token<->id) AVANT de télécharger leurs poids complets
(plusieurs Go). Ne télécharge que les fichiers de tokenizer (quelques Mo).

Un vocab_size identique ne suffit PAS à garantir la compatibilité -- deux
tokenizers différents pourraient coïncidentalement avoir la même taille
avec un mapping totalement différent. Seule l'égalité complète du
dictionnaire vocab->id est une preuve suffisante.

Usage :
    python scripts/check_tokenizer_compatibility.py \\
        Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-Coder-1.5B-Instruct
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_a", type=str)
    parser.add_argument("model_b", type=str)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoConfig

    print(f"Chargement du tokenizer de {args.model_a}...")
    tok_a = AutoTokenizer.from_pretrained(args.model_a)
    print(f"Chargement du tokenizer de {args.model_b}...")
    tok_b = AutoTokenizer.from_pretrained(args.model_b)

    len_a, len_b = len(tok_a), len(tok_b)
    vocab_a, vocab_b = tok_a.get_vocab(), tok_b.get_vocab()

    print(f"\nlen(tokenizer) — {args.model_a}: {len_a}")
    print(f"len(tokenizer) — {args.model_b}: {len_b}")

    same_len = (len_a == len_b)
    same_vocab = (vocab_a == vocab_b)

    print(f"\nMême longueur       : {'OUI' if same_len else 'NON'}")
    print(f"Mapping identique   : {'OUI' if same_vocab else 'NON'}")

    if not same_vocab:
        diff_keys = set(vocab_a.keys()) ^ set(vocab_b.keys())
        mismatched = {k for k in set(vocab_a) & set(vocab_b) if vocab_a[k] != vocab_b[k]}
        print(f"  Tokens présents dans un seul des deux : {len(diff_keys)}")
        print(f"  Tokens présents dans les deux mais avec un id différent : {len(mismatched)}")
        if diff_keys:
            print(f"  Exemples de tokens divergents : {list(diff_keys)[:10]}")

    print(f"\nmodel.config.vocab_size — {args.model_a}: ", end="")
    cfg_a = AutoConfig.from_pretrained(args.model_a)
    print(cfg_a.vocab_size)
    print(f"model.config.vocab_size — {args.model_b}: ", end="")
    cfg_b = AutoConfig.from_pretrained(args.model_b)
    print(cfg_b.vocab_size)

    print()
    if same_vocab and cfg_a.vocab_size == cfg_b.vocab_size:
        print("✅ COMPATIBLE — même tokenizer ET même vocab_size de sortie. "
              "Sûr d'utiliser ces deux modèles ensemble comme Teachers ARCD.")
        sys.exit(0)
    else:
        print("❌ INCOMPATIBLE — NE PAS utiliser ces deux modèles ensemble comme "
              "Teachers ARCD : la médiane pondérée comparerait des positions de "
              "vocabulaire non alignées, silencieusement.")
        sys.exit(1)


if __name__ == "__main__":
    main()
