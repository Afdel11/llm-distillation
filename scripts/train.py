"""
scripts/train.py
=================
Point d'entrée de PRODUCTION, basé sur transformers.Trainer (voir
trainers/hf_trainer.py). Scheduler, accumulation de gradient, reprise sur
coupure (--resume_from_checkpoint) obtenus gratuitement.

Usage :
    python scripts/train.py --config configs/arcd.yaml
    python scripts/train.py --config configs/hinton.yaml
    python scripts/train.py --config configs/multi_teacher_fixed.yaml
    python scripts/train.py --config configs/baseline.yaml
"""

import argparse
import os
import sys

import torch
import yaml
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.tokenizer import get_tokenizer
from data_pipeline.prompts import TOY_EXAMPLES
from data_pipeline.dataloader import PromptResponseDataset, make_collate_fn, make_cached_collate_fn
from data_pipeline.cache import load_teacher_cache
from models.teacher import TeacherEnsemble
from models.student import build_student
from trainers.hf_trainer import ARCDTrainer, HintonTrainer, FixedWeightTrainer, drop_keys_collate


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_resume_checkpoint(output_dir: str, force_restart: bool):
    """
    Détecte un checkpoint interrompu (veille, coupure, Ctrl+C...) dans
    output_dir et le retourne pour reprise automatique. Retourne None si
    force_restart=True (repart de zéro volontairement) ou si aucun
    checkpoint n'existe.
    """
    if force_restart:
        return None
    if not os.path.isdir(output_dir):
        return None
    last_ckpt = get_last_checkpoint(output_dir)
    if last_ckpt is not None:
        print(f"Checkpoint existant trouvé: {last_ckpt} — reprise automatique "
              f"(training arguments, optimizer, scheduler, RNG state, tout est restauré).")
        print("Pour repartir de zéro à la place, relance avec --restart.")
    return last_ckpt


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


def build_training_arguments(cfg: dict, regime: str, has_eval: bool) -> TrainingArguments:
    device = cfg["training"]["device"]
    use_bf16 = (device == "cuda" and torch.cuda.is_available())

    # eval_strategy DOIT être identique à save_strategy quand load_best_model_at_end=True
    # (contrainte de transformers.Trainer). Sans eval_dataset, pas d'évaluation possible :
    # on retombe sur le comportement précédent (sauvegarde par epoch, pas de sélection
    # automatique) plutôt que de planter.
    strategy = cfg["training"].get("eval_save_strategy", "epoch") if has_eval else \
        cfg["training"].get("save_strategy", "epoch")

    # IMPORTANT : pour hinton_kd et arcd, eval_loss est un MÉLANGE (alpha ou
    # lambda(x) pondèrent KD et CE). Ce mélange dérive au fil de
    # l'entraînement (lambda s'effondre progressivement -> eval_loss se
    # rapproche de plus en plus de L_CE seul), ce qui peut faire diverger le
    # classement "meilleur epoch" entre eval_loss et la vraie métrique de
    # comparaison (L_CE seule). On sélectionne donc le meilleur checkpoint et
    # on déclenche l'early stopping directement sur L_CE — pas sur eval_loss
    # composite — pour que le modèle finalement conservé soit celui qui est
    # réellement le meilleur sur la métrique comparable aux autres régimes.
    metric_for_best_model = {
        "student_alone": "eval_loss",          # déjà de la CE pure, pas de mélange
        "hinton_kd": "eval_hinton/L_CE",
        "multi_teacher_fixed": "eval_fixedmt/L_CE",
        "arcd": "eval_arcd/L_CE",
        "arcd_diverse": "eval_arcd/L_CE",   # même ARCDTrainer, même préfixe de log -> même clé
        "arcd_topk": "eval_arcd/L_CE",
        "arcd_topk_diverse": "eval_arcd/L_CE",
        "arcd_topk_10k": "eval_arcd/L_CE",
    }.get(regime, "eval_loss")

    return TrainingArguments(
        output_dir=os.path.join(cfg["output"]["checkpoint_dir"], regime, f"seed_{cfg['training']['seed']}"),
        per_device_train_batch_size=cfg["data"]["batch_size"],
        per_device_eval_batch_size=cfg["data"]["batch_size"],
        # Batch EFFECTIF = batch_size * gradient_accumulation_steps. Utile pour
        # réduire le pic mémoire par step (surtout pour arcd : le tenseur
        # teacher_logits (batch, seq, teachers, vocab) domine la mémoire) SANS
        # changer la dynamique d'entraînement ni casser la comparabilité entre
        # régimes/seeds déjà lancés à un batch effectif donné.
        gradient_accumulation_steps=cfg["data"].get("gradient_accumulation_steps", 1),
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["lr"],
        logging_steps=cfg["training"].get("logging_steps", 1),
        save_strategy=strategy,
        eval_strategy=strategy if has_eval else "no",
        load_best_model_at_end=has_eval,
        metric_for_best_model=metric_for_best_model if has_eval else None,
        greater_is_better=False if has_eval else None,
        save_total_limit=cfg["training"].get("save_total_limit", 3),
        bf16=use_bf16,
        seed=cfg["training"]["seed"],
        report_to=[],  # pas de W&B/TensorBoard configuré par défaut
        remove_unused_columns=False,  # nos batches ont des clés (idx, teacher_logits)
                                       # que le modèle ne consomme pas directement
        # Le collate du régime "arcd" reconstruit un tenseur (batch, seq, teachers,
        # vocab) potentiellement > 1 Go (voir data_pipeline/dataloader.py). Sans
        # workers, ce travail CPU bloque le GPU entre chaque step. Avec des workers
        # + pin_memory, le collate du batch N+1 tourne en arrière-plan pendant que
        # le GPU calcule le batch N — recouvrement, pas d'accélération magique du
        # collate lui-même, mais son coût cesse d'être payé en série.
        # ATTENTION : num_workers > 0 avec le cache ARCD (un dict Python de
        # potentiellement plusieurs dizaines de Go de tenseurs) peut provoquer un
        # OOM : le fork des workers ne partage la mémoire que par copy-on-write,
        # et CHAQUE accès en lecture à un objet Python touche son compteur de
        # références, ce qui invalide ce partage page par page. Défaut à 0
        # (sûr, pas de recouvrement CPU/GPU) ; à n'augmenter que si le cache
        # est un jour restructuré en accès disque paresseux plutôt qu'un seul
        # gros dict préchargé en RAM.
        dataloader_num_workers=cfg["training"].get("dataloader_num_workers", 0),
        dataloader_pin_memory=cfg["training"].get("dataloader_num_workers", 0) > 0,
    )


def main(config_path: str, force_restart: bool = False, seed_override: int = None):
    cfg = load_config(config_path)
    if seed_override is not None:
        cfg["training"]["seed"] = seed_override
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
    print(f"Dataset train: {len(dataset)} exemples")

    val_examples_path = cfg["data"].get("val_examples_path")
    val_dataset = None
    if val_examples_path and os.path.exists(val_examples_path):
        val_examples = load_examples(val_examples_path)
        val_dataset = PromptResponseDataset(val_examples, tokenizer, max_length=cfg["data"]["max_length"])
        print(f"Dataset val: {len(val_dataset)} exemples — évaluation activée, "
              f"meilleur checkpoint sélectionné automatiquement (eval_loss).")
    else:
        print("Pas de data.val_examples_path configuré (ou fichier absent) — "
              "AUCUNE évaluation, AUCUN garde-fou contre le surapprentissage. "
              "Le nombre d'epochs devra être choisi à l'aveugle.")

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

    regime = cfg["training"]["regime"]  # "student_alone" | "hinton_kd" | "multi_teacher_fixed" | "arcd" | "arcd_diverse"
    print(f"\nRégime: {regime}")
    training_args = build_training_arguments(cfg, regime, has_eval=(val_dataset is not None))

    resume_ckpt = find_resume_checkpoint(training_args.output_dir, force_restart)

    callbacks = []
    if val_dataset is not None:
        patience = cfg["training"].get("early_stopping_patience", 3)
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        print(f"Early stopping activé: arrêt si eval_loss ne s'améliore plus pendant {patience} évaluations.")

    if regime == "student_alone":
        # "labels" est déjà dans le batch -> le modèle calcule sa propre
        # cross-entropy nativement (voir Qwen2ForCausalLM.forward). Il suffit
        # de retirer "idx", que le modèle ne sait pas interpréter.
        collate_fn = drop_keys_collate(make_collate_fn(pad_token_id=tokenizer.pad_token_id), keys=("idx",))
        trainer = Trainer(model=student, args=training_args, train_dataset=dataset,
                           eval_dataset=val_dataset, data_collator=collate_fn, callbacks=callbacks)
        trainer.train(resume_from_checkpoint=resume_ckpt)

    elif regime == "hinton_kd":
        first_name = list(cfg["models"]["teacher_names"].keys())[0]
        ensemble = TeacherEnsemble({first_name: cfg["models"]["teacher_names"][first_name]}, device=device)
        teacher = ensemble.models[first_name]

        # Hinton ne met jamais les Teachers en cache (un seul Teacher, coût
        # modéré) -> le même Teacher en direct sert aussi bien à l'entraînement
        # qu'à l'évaluation, sans cache séparé nécessaire.
        collate_fn = drop_keys_collate(make_collate_fn(pad_token_id=tokenizer.pad_token_id), keys=("idx",))
        trainer = HintonTrainer(
            model=student, args=training_args, train_dataset=dataset, eval_dataset=val_dataset,
            data_collator=collate_fn, callbacks=callbacks,
            teacher=teacher, temperature=cfg["training"]["temperature"], alpha=cfg["training"]["hinton_alpha"],
        )
        trainer.train(resume_from_checkpoint=resume_ckpt)

    elif regime == "multi_teacher_fixed":
        # RÉGIME DE CONTRÔLE : réutilise EXACTEMENT le même cache que le
        # régime "arcd" (mêmes 2 Teachers, même cible de consensus robuste
        # P*) -> aucun nouveau calcul Teacher nécessaire si le cache ARCD
        # existe déjà. Seule la loss diffère (poids fixe vs adaptatif).
        cache_path = cfg["output"].get("teacher_cache_path", "outputs/teacher_cache.pt")
        val_cache_path = cfg["output"].get("teacher_cache_val_path", "outputs/teacher_cache_val.pt")
        teacher_ensemble = None

        if os.path.exists(cache_path):
            print(f"Cache de logits Teachers trouvé: {cache_path} — aucun forward Teacher pendant l'entraînement.")
            teacher_logits_cache = load_teacher_cache(cache_path)
            base_collate_fn = make_cached_collate_fn(pad_token_id=tokenizer.pad_token_id,
                                                      teacher_logits_cache=teacher_logits_cache)
            collate_fn = drop_keys_collate(base_collate_fn, keys=("idx",))
        else:
            print(f"Pas de cache trouvé ({cache_path}) — les Teachers tourneront à chaque batch.")
            teacher_ensemble = TeacherEnsemble(cfg["models"]["teacher_names"], device=device)
            collate_fn = drop_keys_collate(make_collate_fn(pad_token_id=tokenizer.pad_token_id), keys=("idx",))

        eval_collate_fn = collate_fn
        if val_dataset is not None and teacher_ensemble is None:
            if os.path.exists(val_cache_path):
                val_teacher_logits_cache = load_teacher_cache(val_cache_path)
                eval_collate_fn = drop_keys_collate(
                    make_cached_collate_fn(pad_token_id=tokenizer.pad_token_id,
                                            teacher_logits_cache=val_teacher_logits_cache),
                    keys=("idx",),
                )
            else:
                print(f"ATTENTION : cache de validation absent ({val_cache_path}) — l'évaluation va échouer. "
                      f"Relance scripts/build_teacher_cache.py.")

        trainer = FixedWeightTrainer(
            model=student, args=training_args, train_dataset=dataset, eval_dataset=val_dataset,
            data_collator=collate_fn,
            eval_data_collator=(eval_collate_fn if eval_collate_fn is not collate_fn else None),
            callbacks=callbacks, teacher_ensemble=teacher_ensemble, temperature=cfg["training"]["temperature"],
            alpha=cfg["training"].get("fixed_alpha", 0.5),
        )
        trainer.train(resume_from_checkpoint=resume_ckpt)

    elif regime in ("arcd", "arcd_diverse", "arcd_topk", "arcd_topk_diverse", "arcd_topk_10k"):
        # "arcd_diverse" est un ALIAS d'"arcd" : même ARCDTrainer, mêmes clés
        # de log ("arcd/...", "eval_arcd/..."), SEULE la config des Teachers
        # diffère (voir configs/arcd_diverse.yaml). Chemins de checkpoint et
        # de cache dérivés de `regime`, donc automatiquement séparés de ceux
        # du régime "arcd" standard -> aucun risque d'écraser les runs existants.
        cache_path = cfg["output"].get("teacher_cache_path", "outputs/teacher_cache.pt")
        val_cache_path = cfg["output"].get("teacher_cache_val_path", "outputs/teacher_cache_val.pt")
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

        # Le cache de validation est INDÉPENDANT du cache d'entraînement (indices
        # propres à chaque dataset). Sans lui, ARCD ne peut évaluer que si un
        # teacher_ensemble en direct est disponible (mode "pas de cache" ci-dessus).
        eval_collate_fn = collate_fn
        if val_dataset is not None and teacher_ensemble is None:
            if os.path.exists(val_cache_path):
                val_teacher_logits_cache = load_teacher_cache(val_cache_path)
                eval_collate_fn = drop_keys_collate(
                    make_cached_collate_fn(pad_token_id=tokenizer.pad_token_id,
                                            teacher_logits_cache=val_teacher_logits_cache),
                    keys=("idx",),
                )
            else:
                print(f"ATTENTION : cache de validation absent ({val_cache_path}) alors qu'un cache "
                      f"d'entraînement existe — l'évaluation ARCD va échouer. Relance "
                      f"scripts/build_teacher_cache.py (il génère maintenant aussi ce cache).")

        trainer = ARCDTrainer(
            model=student, args=training_args, train_dataset=dataset, eval_dataset=val_dataset,
            data_collator=collate_fn,
            eval_data_collator=(eval_collate_fn if eval_collate_fn is not collate_fn else None),
            callbacks=callbacks, teacher_ensemble=teacher_ensemble, temperature=cfg["training"]["temperature"],
            top_k=cfg["training"].get("consensus_top_k"),
        )
        trainer.train(resume_from_checkpoint=resume_ckpt)

    else:
        raise ValueError(f"regime inconnu: {regime!r}")

    final_dir = os.path.join(cfg["output"]["checkpoint_dir"], f"{regime}_seed{cfg['training']['seed']}_final")
    trainer.save_model(final_dir)
    print(f"\nStudent sauvegardé: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/arcd.yaml")
    parser.add_argument("--seed", type=int, default=None,
                         help="Surcharge training.seed de la config (utile pour un balayage multi-seeds).")
    parser.add_argument("--restart", action="store_true",
                         help="Ignore tout checkpoint existant et repart de zéro "
                              "(par défaut, un checkpoint interrompu est repris automatiquement).")
    args = parser.parse_args()
    main(args.config, force_restart=args.restart, seed_override=args.seed)
