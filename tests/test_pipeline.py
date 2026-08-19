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
from trainers.hf_trainer import ARCDTrainer, HintonTrainer, FixedWeightTrainer, drop_keys_collate

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


def test_fixedweight_trainer_live_teachers_runs_without_error(debug_tokenizer):
    """Régime de contrôle (multi_teacher_fixed) : mêmes Teachers qu'ARCD,
    même consensus robuste, mais poids CONSTANT. Vérifie que la classe
    tourne indépendamment d'ARCDTrainer, sans rien y toucher."""
    torch.manual_seed(0)
    dataset = _dataset(debug_tokenizer)
    collate_fn = drop_keys_collate(make_collate_fn(pad_token_id=debug_tokenizer.pad_token_id), keys=("idx",))

    ensemble = DebugTeacherEnsemble(vocab_size=VOCAB_SIZE, num_teachers=2)
    student = _new_student()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = FixedWeightTrainer(model=student, args=_training_args(tmpdir),
                                      train_dataset=dataset, data_collator=collate_fn,
                                      teacher_ensemble=ensemble, temperature=2.0, alpha=0.5)
        trainer.train()


def test_fixedweight_trainer_cached_teachers_runs_without_error(debug_tokenizer):
    """Même test en mode cache — celui réellement utilisé en production,
    puisque multi_teacher_fixed réutilise le cache déjà construit pour arcd."""
    torch.manual_seed(0)
    dataset = _dataset(debug_tokenizer)

    ensemble = DebugTeacherEnsemble(vocab_size=VOCAB_SIZE, num_teachers=2)
    cache = build_teacher_cache(ensemble, dataset, pad_token_id=debug_tokenizer.pad_token_id,
                                 device="cpu", batch_size=2)

    base_collate_fn = make_cached_collate_fn(pad_token_id=debug_tokenizer.pad_token_id,
                                              teacher_logits_cache=cache)
    collate_fn = drop_keys_collate(base_collate_fn, keys=("idx",))
    student = _new_student()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = FixedWeightTrainer(model=student, args=_training_args(tmpdir),
                                      train_dataset=dataset, data_collator=collate_fn,
                                      teacher_ensemble=None, temperature=2.0, alpha=0.5)
        trainer.train()


def test_fixedweight_alpha_stays_constant_across_steps(debug_tokenizer):
    """Vérifie la propriété qui définit ce régime de contrôle : alpha ne doit
    JAMAIS varier d'un batch à l'autre, contrairement à lambda(x) dans ARCD."""
    from arcd.losses import FixedWeightConsensusLoss

    torch.manual_seed(0)
    batch, seq_len, num_teachers, vocab_size = 2, 8, 2, 200
    student_logits = torch.randn(batch, seq_len, vocab_size, requires_grad=True)
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size)
    labels = torch.randint(0, vocab_size, (batch, seq_len))

    criterion = FixedWeightConsensusLoss(temperature=2.0, alpha=0.5)
    _, metrics1 = criterion(student_logits, teacher_logits, labels)

    # Un second batch complètement différent -> alpha doit rester identique
    teacher_logits_2 = torch.randn(batch, seq_len, num_teachers, vocab_size)
    _, metrics2 = criterion(student_logits, teacher_logits_2, labels)

    assert metrics1["alpha"] == metrics2["alpha"] == 0.5


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


def test_arcd_trainer_with_anti_copy_weight_runs_without_error(debug_tokenizer):
    """Vérifie que le régime ARCD s'entraîne sans erreur avec anti_copy_weight
    actif, et que la métrique anti_copy_loss est bien loggée (voir
    scripts/diagnose_generation.py -- diagnostic qui a motivé ce terme)."""
    torch.manual_seed(0)
    dataset = _dataset(debug_tokenizer)
    collate_fn = drop_keys_collate(make_collate_fn(pad_token_id=debug_tokenizer.pad_token_id), keys=("idx",))

    ensemble = DebugTeacherEnsemble(vocab_size=VOCAB_SIZE, num_teachers=2)
    student = _new_student()

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = ARCDTrainer(model=student, args=_training_args(tmpdir),
                               train_dataset=dataset, data_collator=collate_fn,
                               teacher_ensemble=ensemble, temperature=2.0, anti_copy_weight=1.0)
        trainer.train()


def test_anti_copy_penalty_detects_forced_copy_bias():
    """Construit un cas où le Student met un logit énorme sur 'répéter le
    dernier token' alors que ce n'est pas la bonne réponse -- vérifie que
    anti_copy_loss le détecte (proche de 1) et que le gradient pousse à le
    corriger. Régression directe du diagnostic ayant motivé ce terme."""
    from arcd.losses import ARCDLoss

    torch.manual_seed(0)
    batch, seq_len, num_teachers, vocab_size = 2, 6, 2, 100

    input_ids = torch.randint(5, vocab_size, (batch, seq_len))
    labels = torch.randint(5, vocab_size, (batch, seq_len))
    labels = torch.where(labels == input_ids, labels + 1, labels) % vocab_size

    base_logits = torch.randn(batch, seq_len, vocab_size)
    with torch.no_grad():
        for b in range(batch):
            for t in range(seq_len):
                base_logits[b, t, input_ids[b, t]] = 20.0  # force le réflexe de copie

    student_logits = base_logits.clone().requires_grad_(True)
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size)

    criterion = ARCDLoss(temperature=2.0, anti_copy_weight=5.0)
    loss, metrics = criterion(student_logits, teacher_logits, labels, input_ids=input_ids)

    assert metrics["anti_copy_loss"] > 0.9, "p_copy doit être détecté proche de 1 dans ce cas construit"

    loss.backward()
    grad_on_copy_logit = student_logits.grad[0, 0, input_ids[0, 0]].item()
    assert grad_on_copy_logit > 0, "le gradient doit pousser à réduire la probabilité de copie"


def test_anti_copy_penalty_stays_low_without_copy_bias():
    """Vérifie que la pénalité ne punit pas artificiellement un modèle qui
    n'a pas de réflexe de copie particulier (logits aléatoires normaux)."""
    from arcd.losses import ARCDLoss

    torch.manual_seed(1)
    batch, seq_len, num_teachers, vocab_size = 2, 6, 2, 100
    input_ids = torch.randint(5, vocab_size, (batch, seq_len))
    labels = torch.randint(5, vocab_size, (batch, seq_len))
    student_logits = torch.randn(batch, seq_len, vocab_size, requires_grad=True)
    teacher_logits = torch.randn(batch, seq_len, num_teachers, vocab_size)

    criterion = ARCDLoss(temperature=2.0, anti_copy_weight=5.0)
    _, metrics = criterion(student_logits, teacher_logits, labels, input_ids=input_ids)

    assert metrics["anti_copy_loss"] < 0.1
