#!/usr/bin/env python3
"""
scripts/load_model.py
=======================
Charge un Student entraîné + son tokenizer, pour des tests manuels directs
depuis le terminal. Conçu pour être lancé avec `python -i`, afin de rester
dans un shell Python interactif juste après le chargement.

Usage :
    python -i scripts/load_model.py --regime arcd --seed 0

Une fois le prompt Python affiché :
    >>> ask("Quelle est la capitale de la France ?")
    >>> ask("Explique la photosynthèse.", max_new_tokens=150)
    >>> ask("### Instruction:\nDonne trois conseils pour bien dormir.\n\n### Réponse:\n")

ask() accepte soit une instruction brute (le gabarit d'entraînement
"### Instruction: / ### Réponse:" est ajouté automatiquement), soit un
prompt déjà formaté (détecté et laissé tel quel).

Pour changer de régime sans quitter Python :
    >>> exec(open("scripts/load_model.py").read().replace("arcd", "hinton_kd"))
    (ou plus simplement : quitte avec Ctrl+D et relance avec --regime différent)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM

from data_pipeline.tokenizer import get_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--regime", type=str, default="arcd",
                     help="student_alone | hinton_kd | multi_teacher_fixed | arcd | arcd_diverse")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
parser.add_argument("--checkpoint", type=str, default=None,
                     help="Chemin direct vers un checkpoint, pour court-circuiter --regime/--seed.")
parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

checkpoint_path = args.checkpoint or os.path.join(
    args.checkpoint_dir, f"{args.regime}_seed{args.seed}_final"
)

if not os.path.isdir(checkpoint_path):
    print(f"ATTENTION : {checkpoint_path} n'existe pas. Vérifie --regime/--seed/--checkpoint_dir.")
    print(f"Contenu de {args.checkpoint_dir} :")
    if os.path.isdir(args.checkpoint_dir):
        for name in sorted(os.listdir(args.checkpoint_dir)):
            print(f"  {name}")
else:
    print(f"Chargement de {checkpoint_path} sur {args.device}...")
    tokenizer = get_tokenizer(args.tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, dtype=torch.float32).to(args.device)
    model.eval()
    print(f"Modèle chargé ({args.regime}, seed {args.seed}). Utilise ask(\"ta question\") pour tester.\n")


def ask(text, max_new_tokens=80, do_sample=False, temperature=1.0):
    """Envoie une instruction (ou un prompt déjà formaté) au modèle chargé,
    affiche la réponse générée, et la retourne."""
    prompt = text if "### Instruction:" in text else f"### Instruction:\n{text}\n\n### Réponse:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(response if response else "(réponse vide)")
    return response
