from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from config import LLM_MODEL

def load_llm():
    print(f"Cargando modelo {LLM_MODEL}...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL,
        trust_remote_code=True
    )

    return tokenizer, model
