from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from config import LLM_MODEL

def load_llm():
    print(f"Cargando modelo {LLM_MODEL}...")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto", 
        torch_dtype="auto",
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm