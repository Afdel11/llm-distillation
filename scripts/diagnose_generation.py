#!/usr/bin/env python3
"""
scripts/diagnose_generation.py
=================================
Compare, position par position, ce que le modèle prédit :
  (a) en teacher forcing (vrai contexte à chaque position, comme eval_L_CE)
  (b) en génération autonome (son propre contexte généré)

Sur un même exemple réel de validation. Si (a) reste correct loin dans la
séquence mais que (b) décroche après quelques tokens, c'est la signature
de l'exposure bias : le modèle n'a jamais appris à se rattraper de ses
propres erreurs, seulement à prédire sachant un contexte parfait.

Usage :
    python scripts/diagnose_generation.py --checkpoint outputs/checkpoints/arcd_topk_seed1_final
    python scripts/diagnose_generation.py --checkpoint outputs/checkpoints/student_alone_seed0_final
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM

from data_pipeline.tokenizer import get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_file", type=str, default="outputs/data/val.json")
    parser.add_argument("--n_examples", type=int, default=3)
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Chargement de {args.checkpoint} sur {args.device}...")
    tokenizer = get_tokenizer(args.tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32).to(args.device)
    model.eval()

    val_examples = json.load(open(args.val_file, encoding="utf-8"))[:args.n_examples]

    for ex_idx, example in enumerate(val_examples):
        prompt = example["prompt"]
        true_response = example["response"]

        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)
        true_response_ids = tokenizer(true_response, return_tensors="pt")["input_ids"].to(args.device)[0]

        print(f"\n{'=' * 78}")
        print(f"Exemple {ex_idx + 1} — Prompt : {prompt.strip()[:70]}...")
        print(f"Réponse réelle : {true_response.strip()[:70]}...")
        print(f"{'=' * 78}")

        # --- (a) Teacher forcing : à chaque position, le VRAI préfixe ---
        print("\n(a) TEACHER FORCING (vrai contexte à chaque position, comme eval_L_CE) :")
        full_ids = torch.cat([prompt_ids[0], true_response_ids]).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids=full_ids).logits[0]

        n_prompt = prompt_ids.shape[1]
        correct_count, total_count = 0, 0
        end_pos = min(n_prompt + 14, full_ids.shape[1] - 1)
        for pos in range(n_prompt - 1, end_pos):
            true_next = full_ids[0, pos + 1].item()
            probs = torch.softmax(logits[pos], dim=-1)
            pred = probs.argmax().item()
            pred_prob = probs[pred].item()
            true_prob = probs[true_next].item()
            match = "OK" if pred == true_next else "--"
            total_count += 1
            correct_count += int(pred == true_next)
            print(f"  pos {pos - n_prompt + 1:>3} [{match}]  predit={tokenizer.decode([pred])!r:>14} "
                  f"(p={pred_prob:.3f})   vrai={tokenizer.decode([true_next])!r:>14} (p={true_prob:.3f})")
        print(f"  -> {correct_count}/{total_count} tokens correctement predits en teacher forcing")

        # --- (b) Génération autonome : le modèle utilise SA PROPRE sortie ---
        print("\n(b) GENERATION AUTONOME (le modele recycle sa propre sortie) :")
        generated = prompt_ids.clone()
        with torch.no_grad():
            for step in range(15):
                logits_step = model(input_ids=generated).logits[0, -1]
                probs = torch.softmax(logits_step, dim=-1)
                next_token = probs.argmax().item()
                next_prob = probs[next_token].item()
                print(f"  pos {step:>3}      genere={tokenizer.decode([next_token])!r:>14} (p={next_prob:.3f})")
                generated = torch.cat([generated, torch.tensor([[next_token]], device=args.device)], dim=1)

    print(f"\n{'=' * 78}")
    print("LECTURE :")
    print("- Si (a) reste globalement correct/varie loin dans la sequence, mais que")
    print("  (b) decroche apres quelques tokens vers une repetition -- signature de")
    print("  l'exposure bias : le modele sait predire sachant le bon contexte, mais")
    print("  n'a jamais appris a se rattraper de ses propres erreurs.")
    print("- Si (a) est DEJA degenere (probabilite ecrasante sur un token bizarre")
    print("  des les premieres positions, meme avec le vrai contexte) -- le probleme")
    print("  est dans les poids eux-memes, pas dans la boucle de generation.")


if __name__ == "__main__":
    main()
