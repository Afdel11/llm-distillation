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