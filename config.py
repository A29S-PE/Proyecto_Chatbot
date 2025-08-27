# Configuración de LLM
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False

# Configuración de clasificadores
EMOTION_MODEL = "bert_emociones.h5"
MENTAL_MODEL = "bert_estado_mental.h5"
TOKENIZER_MODEL = "dccuchile/bert-base-spanish-wwm-uncased"

# URL de RASA
RASA_URL = "http://localhost:5005/model/parse"
