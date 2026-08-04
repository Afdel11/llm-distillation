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
- **Teacher small** : Qwen/Qwen2.5-0.5B-Instruct
- **Student** : GPT-2 miniature, entraîné from scratch, dimensionné sur le
  vocabulaire du tokenizer Qwen2.5 (`vocab_size = len(tokenizer)`).
  Son architecture n'a pas besoin de ressembler à celle des Teachers —
  seul le partage du tokenizer compte.

## Structure
- `arcd/confidence.py` — confiance via entropie de Gini (inchangée vs. la v1 vision)
- `arcd/consensus.py`  — médiane/MAD pondérés, axe Teachers = avant-dernier (`dim=-2`)
- `arcd/losses.py`     — ARCDLoss token par token, masquage `-100` (prompt/padding)
- `arcd/metrics.py`    — accumulation/affichage des métriques
- `models/teacher.py`  — Qwen2.5 (production) + Teachers factices (`build_debug_teachers`, pour les tests)
- `models/student.py`  — GPT-2 miniature from scratch
- `datasets/tokenizer.py` — chargement du tokenizer partagé
- `datasets/prompts.py`   — mise en forme prompt/réponse + masquage
- `datasets/dataloader.py`— Dataset + collate_fn avec padding dynamique
- `trainers/`          — baseline.py / hinton.py / arcd.py (3 régimes comparables)
- `scripts/train.py`   — point d'entrée de production (vrais Qwen2.5, GPU distant)
- `tests/`             — un fichier de test par module de production (voir plus bas)

Aucun module de production ne contient de code de test (pas de bloc
`if __name__ == "__main__":` sauf dans `scripts/train.py`, qui est un vrai
point d'entrée, pas un test). Tous les tests vivent dans `tests/`, au format
`pytest`, un fichier par module (`arcd/confidence.py` → `tests/test_confidence.py`, etc.).

## Lancer les tests en local (sandbox, sans GPU ni internet)
    pip install -r requirements.txt
    pytest tests/ -v

27 tests s'exécutent réellement (poids aléatoires, tokenizer factice —
valident uniquement la mécanique, pas les performances). 5 tests
(`test_tokenizer.py`, `test_prompts.py`) nécessitent le vrai tokenizer Qwen2.5
et se marquent automatiquement `SKIPPED` faute d'internet ici — ils
s'exécuteront pour de vrai sur ton GPU distant.

## Lancer le vrai entraînement sur ton GPU distant
    pip install -r requirements.txt
    python scripts/train.py --config configs/arcd.yaml       # méthode proposée
    python scripts/train.py --config configs/baseline.yaml   # Student seul (ou change regime: hinton_kd)

Le premier lancement télécharge automatiquement le tokenizer et les poids
Qwen2.5 (~1.5B + 0.5B paramètres) — nécessite un accès internet normal.

**Avant un run complet**, remplace `datasets/prompts.py:TOY_EXAMPLES` (3
exemples de démonstration) par un vrai dataset, via `data.examples_path`
dans la config (fichier `.json`, liste de `{"prompt":..., "response":...}`).

## Point d'attention mémoire (GPU)
Le calcul de la médiane/MAD par token porte sur des tenseurs de forme
`(batch, seq_len, num_teachers, vocab_size)`. Avec un vocabulaire Qwen2.5
(~151 936 tokens), ça peut représenter plusieurs Go temporairement pour un
batch de taille normale. Si tu rencontres un OOM sur le GPU distant, réduis
`data.batch_size` et/ou `data.max_length` dans la config avant de réduire
la taille des modèles.

## Tester chaque module isolément
    pytest tests/test_confidence.py -v
    pytest tests/test_consensus.py -v
    pytest tests/test_losses.py -v
    pytest tests/test_metrics.py -v
    pytest tests/test_teacher.py -v      # mode debug, offline
    pytest tests/test_student.py -v      # mode debug, offline
    pytest tests/test_dataloader.py -v   # tokenizer factice, offline
    pytest tests/test_tokenizer.py -v    # nécessite internet (vrai tokenizer Qwen), sinon SKIPPED
    pytest tests/test_prompts.py -v      # nécessite internet (vrai tokenizer Qwen), sinon SKIPPED
    pytest tests/test_pipeline.py -v     # intégration complète, offline

## Prochaines étapes
1. [x] arcd/ généralisé au cas token-par-token (confidence, consensus, losses, metrics)
2. [x] Teachers Qwen2.5 (même tokenizer) + Student GPT-2 from scratch
3. [x] Pipeline de données (tokenisation, masquage, padding dynamique)
4. [x] 3 régimes d'entraînement (baseline, hinton, arcd), testés en intégration locale
5. [ ] Remplacer TOY_EXAMPLES par un vrai dataset instructif
6. [ ] Premier vrai run sur le GPU distant (télécharger Qwen2.5, vérifier le OOM)
7. [ ] Étude d'ablation (retrait successif de C, T, S — voir chapitre méthodologie)
