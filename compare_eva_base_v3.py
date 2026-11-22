import unsloth
from unsloth import FastLanguageModel
import torch

BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
ADAPTER_DIR = r"D:\Eva_Ai_brain\eva-lora-adapter-v3"

questions = [
    "Explain EVA AI in simple English.",
    "RAG म्हणजे काय? मराठीत सोप्या भाषेत सांग."
]

print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

def ask(question):
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
        )

    return tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    ).strip()

print("\n========== BASE MODEL (NO V3 ADAPTER) ==========")
for q in questions:
    print("\nQUESTION:", q)
    print("ANSWER:", ask(q))

print("\nLoading V3 adapter...")
model.load_adapter(ADAPTER_DIR)
FastLanguageModel.for_inference(model)

print("\n========== V3 ADAPTER ENABLED ==========")
for q in questions:
    print("\nQUESTION:", q)
    print("ANSWER:", ask(q))

print("\nComparison complete.")
