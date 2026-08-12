"""
tests/test_pipeline.py
========================
Test d'intégration de bout en bout, ENTIÈREMENT local (aucun accès réseau) :
Teachers factices -> ARCDTrainer/HintonTrainer/Trainer -> Student factice.
Couvre aussi le chemin "cache de logits" (build_teacher_cache + collate caché).
"""

import tempfile

import torch
from transformers import Trainer, TrainingArguments

from models.teacher import DebugTeacherEnsemble
from models.student import build_student
from data_pipeline.dataloader import PromptResponseDataset, make_collate_fn, make_cached_collate_fn
from data_pipeline.cache import build_teacher_cache
from trainers.hf_trainer import ARCDTrainer, HintonTrainer, drop_keys_collate

VOCAB_SIZE = 128  # doit couvrir les ids générés par DebugTokenizer (mots courts)

EXAMPLES = [
    {"prompt": "Question courte simple", "response": "Réponse brève ici maintenant"},
    {"prompt": "Une question un peu plus longue avec plein de mots différents",
     "response": "Une réponse également assez longue pour tester le padding"},
    {"prompt": "Test", "response": "Ok court"},
    {"prompt": "Autre exemple de question test", "response": "Autre réponse de test complète"},
]


def _dataset(debug_tokenizer):
    return PromptResponseDataset(EXAMPLES, debug_tokenizer, max_length=32)


def _new_student():
    return build_student(VOCAB_SIZE, hidden_size=32, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1,
                          intermediate_size=64, max_position_embeddings=64)


def _training_args(tmpdir):
    return TrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=5e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=True,
    )


def test_student_alone_trainer_runs_without_error(debug_tokenizer):
    torch.manual_seed(0)
    dataset = _dataset(debug_tokenizer)
    collate_fn = drop_keys_collate(make_collate_fn(pad_token_id=debug_tokenizer.pad_token_id), keys=("idx",))
    student = _new_student()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(model=student, args=_training_args(tmpdir),
                           train_dataset=dataset, data_collator=collate_fn)
        trainer.train()


def test_hinton_trainer_runs_without_error(debug_tokenizer):
    torch.manual_seed(0)
    dataset = _dataset(debug_tokenizer)
    collate_fn = drop_keys_collate(make_collate_fn(pad_token_id=debug_tokenizer.pad_token_id), keys=("idx",))

    ensemble = DebugTeacherEnsemble(vocab_size=VOCAB_SIZE, num_teachers=1)
    teacher = ensemble.models[ensemble.teacher_names[0]]
    student = _new_student()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = HintonTrainer(model=student, args=_training_args(tmpdir),
                                 train_dataset=dataset, data_collator=collate_fn,
                                 teacher=teacher, temperature=2.0, alpha=0.5)
        trainer.train()


def test_arcd_trainer_live_teachers_runs_without_error(debug_tokenizer):
    """Mode "direct" : les Teachers tournent à chaque batch."""
    torch.manual_seed(0)
    dataset = _dataset(debug_tokenizer)
    collate_fn = drop_keys_collate(make_collate_fn(pad_token_id=debug_tokenizer.pad_token_id), keys=("idx",))

    ensemble = DebugTeacherEnsemble(vocab_size=VOCAB_SIZE, num_teachers=2)
    student = _new_student()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ARCDTrainer(model=student, args=_training_args(tmpdir),
                               train_dataset=dataset, data_collator=collate_fn,
                               teacher_ensemble=ensemble, temperature=2.0)
        trainer.train()


def test_arcd_trainer_cached_teachers_runs_without_error(debug_tokenizer):
    """Mode "cache" : logits Teachers pré-calculés, aucun forward Teacher pendant l'entraînement."""
    torch.manual_seed(0)
    dataset = _dataset(debug_tokenizer)

    ensemble = DebugTeacherEnsemble(vocab_size=VOCAB_SIZE, num_teachers=2)
    cache = build_teacher_cache(ensemble, dataset, pad_token_id=debug_tokenizer.pad_token_id,
                                 device="cpu", batch_size=2)
    assert len(cache) == len(dataset)

    base_collate_fn = make_cached_collate_fn(pad_token_id=debug_tokenizer.pad_token_id,
                                              teacher_logits_cache=cache)
    collate_fn = drop_keys_collate(base_collate_fn, keys=("idx",))
    student = _new_student()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ARCDTrainer(model=student, args=_training_args(tmpdir),
                               train_dataset=dataset, data_collator=collate_fn,
                               teacher_ensemble=None, temperature=2.0)  # pas d'ensemble : lit le cache
        trainer.train()


def test_arcd_trainer_eval_with_distinct_cache_does_not_crash(debug_tokenizer):
    """
    Reproduit le scénario ajouté contre le surapprentissage : un cache
    d'ENTRAÎNEMENT et un cache de VALIDATION distincts (deux datasets
    séparés, indices non interchangeables), avec eval_data_collator
    différent de data_collator sur le même Trainer. Vérifie que
    get_eval_dataloader (surchargé dans ARCDTrainer) bascule bien sur le
    bon collator, sans mélanger les deux caches, et que l'entraînement +
    l'évaluation se déroulent sans erreur.
    """
    torch.manual_seed(0)

    val_examples = EXAMPLES[:2]  # volontairement plus petit que le train,
                                   # pour être sûr qu'on ne réutilise pas le cache train par erreur
    train_dataset = _dataset(debug_tokenizer)
    val_dataset = PromptResponseDataset(val_examples, debug_tokenizer, max_length=32)

    ensemble = DebugTeacherEnsemble(vocab_size=VOCAB_SIZE, num_teachers=2)
    train_cache = build_teacher_cache(ensemble, train_dataset, pad_token_id=debug_tokenizer.pad_token_id,
                                       device="cpu", batch_size=2)
    val_cache = build_teacher_cache(ensemble, val_dataset, pad_token_id=debug_tokenizer.pad_token_id,
                                     device="cpu", batch_size=2)

    train_collate = drop_keys_collate(
        make_cached_collate_fn(pad_token_id=debug_tokenizer.pad_token_id, teacher_logits_cache=train_cache),
        keys=("idx",))
    eval_collate = drop_keys_collate(
        make_cached_collate_fn(pad_token_id=debug_tokenizer.pad_token_id, teacher_logits_cache=val_cache),
        keys=("idx",))

    student = _new_student()

    with tempfile.TemporaryDirectory() as tmpdir:
        args = TrainingArguments(
            output_dir=tmpdir, per_device_train_batch_size=2, per_device_eval_batch_size=2,
            num_train_epochs=1, eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            report_to=[], logging_steps=100, remove_unused_columns=False, disable_tqdm=True,
        )
        trainer = ARCDTrainer(
            model=student, args=args, train_dataset=train_dataset, eval_dataset=val_dataset,
            data_collator=train_collate, eval_data_collator=eval_collate, temperature=2.0,
        )
        trainer.train()  # ne doit pas planter : confirme que get_eval_dataloader utilise bien val_cache

        metrics = trainer.evaluate()
        assert "eval_loss" in metrics
        assert metrics["eval_loss"] == metrics["eval_loss"]  # pas de NaN (NaN != NaN)

        # Les métriques détaillées doivent être moyennées et fusionnées DANS
        # la même entrée que eval_loss, avec le préfixe "eval_" — sinon aucun
        # moyen de comparer objectivement ARCD à Hinton/Baseline sur eval_L_CE.
        for key in ("eval_arcd/loss", "eval_arcd/L_KD", "eval_arcd/L_CE",
                    "eval_arcd/C", "eval_arcd/T", "eval_arcd/S", "eval_arcd/lambda"):
            assert key in metrics, f"Métrique manquante après evaluation_loop: {key}"
            assert metrics[key] == metrics[key], f"NaN détecté dans {key}"

        # Ces métriques ne doivent PAS avoir pollué log_history batch par
        # batch pendant l'éval : une seule entrée par ÉVÉNEMENT d'évaluation
        # (train() en déclenche un en interne à la fin de l'epoch, puis
        # l'appel explicite à evaluate() ci-dessus en déclenche un second).
        eval_entries = [e for e in trainer.state.log_history if "eval_loss" in e]
        assert len(eval_entries) == 2
        for entry in eval_entries:
            assert "arcd/loss" not in entry  # la clé SANS préfixe eval_ ne doit pas traîner ici
