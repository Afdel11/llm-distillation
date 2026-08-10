from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen2.5-0.5B-Instruct"

tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

print("tokenizer.vocab_size :", tok.vocab_size)
print("len(tokenizer)       :", len(tok))
print("model vocab_size     :", model.config.vocab_size)
