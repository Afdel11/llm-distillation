#!/usr/bin/env python3
"""
scripts/generate_samples.py
=============================
Charge les Students entraînés (checkpoints finaux, un par régime) et génère
leurs réponses sur les mêmes prompts, en génération déterministe (greedy,
do_sample=False) pour que la comparaison soit reproductible d'un run à
l'autre. Complément qualitatif aux métriques quantitatives (eval_L_CE) déjà
présentées au chapitre 5 du mémoire — voir aussi les points suivants avant
d'interpréter les sorties :

- Le Student (MiniQwen) est petit (4 couches, hidden_size=512) et entraîné
  from scratch sur seulement 900 exemples : même le meilleur régime ne
  produira pas des réponses dignes d'un modèle de production. L'objectif
  ici est la comparaison RELATIVE entre régimes, pas la qualité absolue.
- La génération déterministe élimine la variance due à l'échantillonnage,
  pour isoler l'effet du régime d'entraînement sur la sortie.

Usage :
    python scripts/generate_samples.py
    python scripts/generate_samples.py --regimes arcd,hinton_kd --seed 0
    python scripts/generate_samples.py --prompts_file mes_prompts.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM

from data_pipeline.tokenizer import get_tokenizer

ALL_REGIMES = ["student_alone", "hinton_kd", "multi_teacher_fixed", "arcd", "arcd_diverse"]

REGIME_DISPLAY = {
    "student_alone": "Baseline (Student seul)",
    "hinton_kd": "Hinton KD",
    "multi_teacher_fixed": "Multi-Teacher poids fixe",
    "arcd": "ARCD",
    "arcd_diverse": "ARCD (Teacher diversifié)",
}

DEFAULT_PROMPTS = [
    "### Instruction:\nQuelle est la capitale de la France ?\n\n### Réponse:\n",
    "### Instruction:\nDonne trois conseils pour bien dormir.\n\n### Réponse:\n",
    "### Instruction:\nExplique la photosynthèse en une phrase simple.\n\n### Réponse:\n",
    "### Instruction:\nÉcris une phrase sur le printemps.\n\n### Réponse:\n",
    "### Instruction:\nPropose un titre pour un roman de science-fiction.\n\n### Réponse:\n",
]


def generate(model, tokenizer, prompt, max_new_tokens, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy, déterministe -> comparaison reproductible entre régimes
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", type=str, default=None,
                         help=f"Liste séparée par des virgules parmi: {', '.join(ALL_REGIMES)}. "
                              f"Par défaut : tous les régimes disponibles.")
    parser.add_argument("--seed", type=int, default=0,
                         help="Quel checkpoint de seed charger pour chaque régime (défaut: 0).")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompts_file", type=str, default=None,
                         help="Fichier JSON: liste de chaînes de prompts. Sinon, prompts par défaut.")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default="outputs/analysis/generation_samples.json")
    args = parser.parse_args()

    regimes = ALL_REGIMES if args.regimes is None else args.regimes.split(",")
    prompts = json.load(open(args.prompts_file, encoding="utf-8")) if args.prompts_file else DEFAULT_PROMPTS

    print(f"Device: {args.device}")
    tokenizer = get_tokenizer(args.tokenizer_name)

    results = {}
    for regime in regimes:
        ckpt = os.path.join(args.checkpoint_dir, f"{regime}_seed{args.seed}_final")
        if not os.path.isdir(ckpt):
            print(f"\n[IGNORÉ] {regime}: checkpoint introuvable ({ckpt})")
            continue

        print(f"\n{'=' * 70}\n{REGIME_DISPLAY.get(regime, regime)}  ({ckpt})\n{'=' * 70}")
        model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32).to(args.device)
        model.eval()

        results[regime] = []
        for prompt in prompts:
            response = generate(model, tokenizer, prompt, args.max_new_tokens, args.device)
            results[regime].append({"prompt": prompt, "response": response})
            instruction = prompt.split("### Instruction:\n")[-1].split("\n\n###")[0]
            print(f"\n  Q: {instruction}")
            print(f"  R: {response if response else '(réponse vide)'}")

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nRésultats sauvegardés -> {args.output_json}")
    print("Récupère ce fichier (cat + copier-coller, ou upload) pour l'intégrer au mémoire.")


if __name__ == "__main__":
    main()
