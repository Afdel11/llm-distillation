# llm-distillation — ARCD pour la distillation de LLM

## Idée
lambda(x) = C(x) * T(x) * (1 - S(x)), calculé **par token**.
L = lambda * L_KD + (1 - lambda) * L_CE

- C : consensus robuste entre Teachers, à chaque position (médiane pondérée + MAD pondéré)
- T : confiance moyenne des Teachers, à chaque position (entropie de Gini)
- S : confiance du Student, à chaque position (entropie de Gini)

## Choix des modèles — ATTENTION AU TOKENIZER
Teachers et Student DOIVENT partager le même tokenizer, sinon la médiane
pondérée compare des positions de vocabulaire qui ne désignent pas le même
token d'un modèle à l'autre (bug silencieux, aucun crash).

- **Teacher large** : Qwen/Qwen2.5-1.5B-Instruct
- **Teacher small** : Qwen/Qwen2.5-0.5B-Instruct — chargés en `bfloat16` (poids natifs Qwen2.5)
- **Student "MiniQwen"** : même architecture que les Teachers (Qwen2 — RoPE,
  RMSNorm, Grouped Query Attention), mais réduite (4 couches, hidden_size=512),
  poids aléatoires, entraîné from scratch. Dimensionné sur
  `vocab_size = len(tokenizer)` du tokenizer partagé. La contrainte réelle et
  non négociable est le partage du tokenizer, pas de l'architecture — mais
  garder la même famille évite de mélanger deux inductive bias différents
  sans raison, et permet d'écrire "le Student reprend l'architecture des
  Teachers avec une capacité réduite" dans le mémoire.

## Coût computationnel — pourquoi le cache de logits Teachers existe
Le calcul ARCD lui-même (médiane, MAD, entropie) est négligeable — de l'ordre
de 1000x moins cher que le simple forward pass des Teachers. Le vrai coût est
ailleurs : les Teachers sont **gelés**, donc recalculer leur forward à chaque
epoch est un gaspillage pur. `scripts/build_teacher_cache.py` les fait tourner
UNE SEULE FOIS sur tout le dataset et sauvegarde leurs logits (float16) sur
disque ; `scripts/train.py` détecte automatiquement ce cache et l'utilise
si présent (sinon, il recalcule en direct — plus simple mais plus lent).

## Structure
- `arcd/confidence.py` — confiance via entropie de Gini
- `arcd/consensus.py`  — médiane/MAD pondérés, axe Teachers = avant-dernier (`dim=-2`)
- `arcd/losses.py`     — ARCDLoss token par token, masquage `-100` (prompt/padding)
- `arcd/metrics.py`    — accumulation/affichage des métriques
- `models/teacher.py`  — `TeacherEnsemble` (Qwen2.5, production) + `DebugTeacherEnsemble` (tests, offline)
- `models/student.py`  — MiniQwen (Qwen2Config réduit), from scratch
- `data_pipeline/tokenizer.py` — chargement du tokenizer partagé
- `data_pipeline/prompts.py`   — mise en forme prompt/réponse + masquage
- `data_pipeline/dataloader.py`— Dataset + collate_fn (direct ou avec cache de logits)
- `data_pipeline/cache.py`     — calcul et sauvegarde des logits Teachers
- `trainers/hf_trainer.py` — `ARCDTrainer`/`HintonTrainer` (sous-classes de `transformers.Trainer` ;
  `student_alone` utilise `Trainer` directement, sans sous-classe)
- `scripts/build_teacher_cache.py` — pré-calcule les logits Teachers (à lancer avant train.py)
- `scripts/train.py`   — point d'entrée de production (vrais Qwen2.5, GPU distant)
- `tests/`             — un fichier de test par module de production

Aucun module de production ne contient de code de test (pas de bloc
`if __name__ == "__main__":` sauf dans `scripts/`, qui sont de vrais points
d'entrée). Tous les tests vivent dans `tests/`, au format `pytest`.

## Lancer les tests en local (sandbox, sans GPU ni internet)
    pip install -r requirements.txt
    pytest tests/ -v

30 tests s'exécutent réellement (poids aléatoires, tokenizer factice,
`DebugTeacherEnsemble` — valident uniquement la mécanique, y compris le
chemin cache, pas les performances). 5 tests (`test_tokenizer.py`,
`test_prompts.py`) nécessitent le vrai tokenizer Qwen2.5 et se marquent
automatiquement `SKIPPED` faute d'internet ici — ils s'exécuteront pour de
vrai sur ton GPU distant.

## Lancer le vrai entraînement sur ton GPU distant
    pip install -r requirements.txt
    python scripts/build_teacher_cache.py --config configs/arcd.yaml   # une seule fois
    python scripts/train.py --config configs/arcd.yaml                 # méthode proposée
    python scripts/train.py --config configs/baseline.yaml             # Student seul (ou regime: hinton_kd)

Le premier lancement télécharge automatiquement le tokenizer et les poids
Qwen2.5 (~1.5B + 0.5B paramètres, en bf16) — nécessite un accès internet normal.

**Avant un run complet**, remplace `data_pipeline/prompts.py:TOY_EXAMPLES` (3
exemples de démonstration) par un vrai dataset :

    python scripts/prepare_dataset.py \
        --hf_dataset jpacifico/French-Alpaca-dataset-Instruct-55K \
        --n_samples 1000 \
        --max_length 256 \
        --tokenizer_name Qwen/Qwen2.5-0.5B-Instruct

Puis mets `data.examples_path: "outputs/data/train.json"` dans
`configs/arcd.yaml`, `hinton.yaml` et `baseline.yaml` (les trois régimes
doivent utiliser exactement le même dataset pour que la comparaison soit
valide). `--max_length` doit correspondre à `data.max_length` dans ces
mêmes configs.

## Pourquoi transformers.Trainer plutôt que des boucles maison
Les premières versions de ce projet réécrivaient la boucle d'entraînement à
la main (trainers/baseline.py, hinton.py, arcd.py). Choix reconsidéré :
`arcd/` (confidence, consensus, losses) est la vraie contribution de
recherche et devait être écrit à la main — mais le scheduler de learning
rate, l'accumulation de gradient, la reprise sur coupure (`--resume_from_checkpoint`)
et la précision mixte n'avaient aucune raison d'être réinventés.
`trainers/hf_trainer.py` branche `arcd/` directement dans `compute_loss()`
de `transformers.Trainer` ; tout le reste (scheduler, sauvegarde, logging)
vient de la librairie.

## Historique : collision de nom résolue par renommage
Ce package s'appelait `datasets/`, comme la librairie HuggingFace `datasets`
(celle de `load_dataset()`). En mode éditable (`pip install -e .`), ça
rendait `import datasets` systématiquement ambigu dans tout le venv —
y compris **à l'intérieur de `transformers.Trainer` lui-même**
(`isinstance(train_dataset, datasets.Dataset)`), ce qui aurait fait planter
tout entraînement basé sur `Trainer`. Renommé en `data_pipeline/` pour
lever l'ambiguïté définitivement. `scripts/prepare_dataset.py` continue de
passer par `requests`/`pandas` plutôt que par la librairie `datasets`, par
choix de rester sans dépendance supplémentaire — plus par nécessité.

## Point d'attention mémoire (GPU)
Le calcul de la médiane/MAD par token porte sur des tenseurs de forme
`(batch, seq_len, num_teachers, vocab_size)`. Avec un vocabulaire Qwen2.5
(~151 936 tokens), ça peut représenter plusieurs Go temporairement pour un
batch de taille normale. Si tu rencontres un OOM sur le GPU distant, réduis
`data.batch_size` et/ou `data.max_length` dans la config avant de réduire
la taille des modèles. Le passage en bf16 pour les Teachers divise déjà la
mémoire des poids par ~2 par rapport à float32.

## Tester chaque module isolément
    pytest tests/test_confidence.py -v
    pytest tests/test_consensus.py -v
    pytest tests/test_losses.py -v
    pytest tests/test_metrics.py -v
    pytest tests/test_teacher.py -v      # DebugTeacherEnsemble, offline
    pytest tests/test_student.py -v      # MiniQwen, offline
    pytest tests/test_dataloader.py -v   # tokenizer factice, offline
    pytest tests/test_tokenizer.py -v    # nécessite internet (vrai tokenizer Qwen), sinon SKIPPED
    pytest tests/test_prompts.py -v      # nécessite internet (vrai tokenizer Qwen), sinon SKIPPED
    pytest tests/test_pipeline.py -v     # intégration complète (modes direct ET cache), offline

## Prochaines étapes
1. [x] arcd/ généralisé au cas token-par-token (confidence, consensus, losses, metrics)
2. [x] TeacherEnsemble (Qwen2.5, bf16) + Student MiniQwen from scratch
3. [x] Cache des logits Teachers (évite de recalculer un forward gelé à chaque epoch)
4. [x] Pipeline de données (tokenisation, masquage, padding dynamique, mode caché)
5. [x] 3 régimes d'entraînement (baseline, hinton, arcd), testés en intégration locale (direct + cache)
6. [ ] Remplacer TOY_EXAMPLES par un vrai dataset instructif
7. [ ] Premier vrai run sur le GPU distant (télécharger Qwen2.5, vérifier le OOM)
8. [ ] Étude d'ablation (retrait successif de C, T, S — voir chapitre méthodologie)
