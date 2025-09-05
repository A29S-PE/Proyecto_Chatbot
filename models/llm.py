from transformers import pipeline
from config import LLM_MODEL
import torch

def load_llm():
    print(f"Cargando modelo {LLM_MODEL}...")

    pipe = pipeline(
        "text-generation",
        model=LLM_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    return pipe
