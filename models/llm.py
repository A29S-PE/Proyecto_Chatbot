from transformers import pipeline, BitsAndBytesConfig
from config import LLM_MODEL
import torch

def load_llm():
    print(f"Cargando modelo {LLM_MODEL}...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    pipe = pipeline(
        "text-generation",
        model=LLM_MODEL,
        quantization_config=quant_config,
        device_map="auto"
    )

    return pipe
