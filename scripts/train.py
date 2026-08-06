"""
scripts/train.py
=================
Point d'entrée de PRODUCTION, basé sur transformers.Trainer (voir
trainers/hf_trainer.py). Scheduler, accumulation de gradient, reprise sur
coupure (--resume_from_checkpoint) obtenus gratuitement.

Usage :
    python scripts/train.py --config configs/arcd.yaml
    python scripts/train.py --config configs/hinton.yaml
    python scripts/train.py --config configs/baseline.yaml
"""

import argparse
import os
import sys

import torch
import yaml
from transformers import Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.tokenizer import get_tokenizer
from data_pipeline.prompts import TOY_EXAMPLES
from data_pipeline.dataloader import PromptResponseDataset, make_collate_fn, make_cached_collate_fn
from data_pipeline.cache import load_teacher_cache
from models.teacher import TeacherEnsemble
from models.student import build_student
from trainers.hf_trainer import ARCDTrainer, HintonTrainer, drop_keys_collate


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model_vocab_size(teacher_names: dict) -> int:
    """
    model.config.vocab_size de chaque Teacher (léger : ne télécharge que
    config.json). Vérifie qu'ils coïncident tous -- une divergence
    signalerait des Teachers qui ne partagent en fait pas le même
    tokenizer/vocabulaire malgré leurs noms.
    """
    from transformers import AutoConfig

    sizes = {name: AutoConfig.from_pretrained(hf_name).vocab_size
             for name, hf_name in teacher_names.items()}
    unique_sizes = set(sizes.values())
    assert len(unique_sizes) == 1, (
        f"Les Teachers n'ont pas le même vocab_size de sortie: {sizes}. "
        "Vérifie qu'ils partagent bien le même tokenizer avant de continuer."
    )
    return unique_sizes.pop()


def load_examples(path: str = None) -> list:
    if path is None:
        return TOY_EXAMPLES
    import json
    with open(path, "r") as f:
        return json.load(f)


def build_training_arguments(cfg: dict, regime: str) -> TrainingArguments:
    device = cfg["training"]["device"]
    use_bf16 = (device == "cuda" and torch.cuda.is_available())

    return TrainingArguments(
        output_dir=os.path.join(cfg["output"]["checkpoint_dir"], regime),
        per_device_train_batch_size=cfg["data"]["batch_size"],
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["lr"],
        logging_steps=cfg["training"].get("logging_steps", 1),
        save_strategy=cfg["training"].get("save_strategy", "epoch"),
        bf16=use_bf16,
        seed=cfg["training"]["seed"],
        report_to=[],  # pas de W&B/TensorBoard configuré par défaut
        remove_unused_columns=False,  # nos batches ont des clés (idx, teacher_logits)
                                       # que le modèle ne consomme pas directement
    )


def main(config_path: str):
    cfg = load_config(config_path)
    torch.manual_seed(cfg["training"]["seed"])

    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("ATTENTION : cuda demandé mais indisponible, retour sur cpu.")
        device = "cpu"
        cfg["training"]["device"] = "cpu"
    print(f"Device: {device}")

    # --- Tokenizer + vocab_size réel des Teachers (voir get_model_vocab_size) ---
    tokenizer = get_tokenizer(cfg["data"]["tokenizer_name"])
    model_vocab_size = get_model_vocab_size(cfg["models"]["teacher_names"])
    print(f"Tokenizer: {cfg['data']['tokenizer_name']} — "
          f"len(tokenizer)={len(tokenizer)}, model.config.vocab_size={model_vocab_size}")

    # --- Données ---
    examples = load_examples(cfg["data"].get("examples_path"))
    dataset = PromptResponseDataset(examples, tokenizer, max_length=cfg["data"]["max_length"])
    print(f"Dataset: {len(dataset)} exemples")

    # --- Student MiniQwen (from scratch, dimensionné sur la SORTIE réelle des Teachers) ---
    student_cfg = cfg["models"]["student"]
    student = build_student(
        vocab_size=model_vocab_size,
        hidden_size=student_cfg["hidden_size"],
        num_hidden_layers=student_cfg["num_hidden_layers"],
        num_attention_heads=student_cfg["num_attention_heads"],
        num_key_value_heads=student_cfg["num_key_value_heads"],
        intermediate_size=student_cfg["intermediate_size"],
    )

    regime = cfg["training"]["regime"]  # "student_alone" | "hinton_kd" | "arcd"
    print(f"\nRégime: {regime}")
    training_args = build_training_arguments(cfg, regime)

    if regime == "student_alone":
        # "labels" est déjà dans le batch -> le modèle calcule sa propre
        # cross-entropy nativement (voir Qwen2ForCausalLM.forward). Il suffit
        # de retirer "idx", que le modèle ne sait pas interpréter.
        collate_fn = drop_keys_collate(make_collate_fn(pad_token_id=tokenizer.pad_token_id), keys=("idx",))
        trainer = Trainer(model=student, args=training_args, train_dataset=dataset, data_collator=collate_fn)
        trainer.train()

    elif regime == "hinton_kd":
        first_name = list(cfg["models"]["teacher_names"].keys())[0]
        ensemble = TeacherEnsemble({first_name: cfg["models"]["teacher_names"][first_name]}, device=device)
        teacher = ensemble.models[first_name]

        collate_fn = drop_keys_collate(make_collate_fn(pad_token_id=tokenizer.pad_token_id), keys=("idx",))
        trainer = HintonTrainer(
            model=student, args=training_args, train_dataset=dataset, data_collator=collate_fn,
            teacher=teacher, temperature=cfg["training"]["temperature"], alpha=cfg["training"]["hinton_alpha"],
        )
        trainer.train()

    elif regime == "arcd":
        cache_path = cfg["output"].get("teacher_cache_path", "outputs/teacher_cache.pt")
        teacher_ensemble = None

        if os.path.exists(cache_path):
            print(f"Cache de logits Teachers trouvé: {cache_path} — aucun forward Teacher pendant l'entraînement.")
            teacher_logits_cache = load_teacher_cache(cache_path)
            base_collate_fn = make_cached_collate_fn(pad_token_id=tokenizer.pad_token_id,
                                                      teacher_logits_cache=teacher_logits_cache)
            collate_fn = drop_keys_collate(base_collate_fn, keys=("idx",))
        else:
            print(f"Pas de cache trouvé ({cache_path}) — les Teachers tourneront à chaque batch. "
                  f"Lance scripts/build_teacher_cache.py avant un vrai entraînement pour éviter ça.")
            teacher_ensemble = TeacherEnsemble(cfg["models"]["teacher_names"], device=device)
            collate_fn = drop_keys_collate(make_collate_fn(pad_token_id=tokenizer.pad_token_id), keys=("idx",))

        trainer = ARCDTrainer(
            model=student, args=training_args, train_dataset=dataset, data_collator=collate_fn,
            teacher_ensemble=teacher_ensemble, temperature=cfg["training"]["temperature"],
        )
        trainer.train()

    else:
        raise ValueError(f"regime inconnu: {regime!r}")

    trainer.save_model(os.path.join(cfg["output"]["checkpoint_dir"], f"{regime}_final"))
    print(f"\nStudent sauvegardé: {cfg['output']['checkpoint_dir']}/{regime}_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/arcd.yaml")
    args = parser.parse_args()
    main(args.config)
