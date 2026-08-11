"""
scripts/smoke_test_teacher.py
===============================
Vérification manuelle rapide : charge le tokenizer + un Teacher réel sur
GPU et génère une réponse. Nécessite internet + cuda — CE N'EST PAS UN TEST
PYTEST (pas d'assertion, chargement de modèle au niveau module), donc ce
fichier vit dans scripts/ et pas dans tests/. Un fichier placé dans tests/
avec du code d'import qui plante casse la collection de TOUTE la suite
pytest, même pour les tests qui n'en ont pas besoin.

Usage : python scripts/smoke_test_teacher.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

print("Chargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Chargement du modèle...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

print("GPU :", next(model.parameters()).device)

prompt = "What is Knowledge Distillation?"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=40)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))