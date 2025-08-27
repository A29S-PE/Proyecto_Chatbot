from fastapi import FastAPI
from schemas.chat import ChatRequest
from models.classifier import ClassifierManager
from models.intent import IntentClassifier
from models.llm import load_llm
from chains.policy import build_policy_chain
from chains.response import build_response_chain
from memory.memory import MemoryManager
from pydantic import BaseModel

# =======================
# Inicialización
# =======================

app = FastAPI(
    title="Chatbot Conversacional",
    description="API para interactuar con el chatbot en 3 pasos (clasificación, política de diálogo, generación de respuesta).",
    version="1.0.0"
)

# Modelo de entrada
class ChatRequest(BaseModel):
    user_id: str
    message: str

# Modelo de salida
class ChatResponse(BaseModel):
    response: str

classifier_manager = ClassifierManager()
intent_classifier = IntentClassifier()
llm = load_llm()
policy_chain = build_policy_chain(llm)
response_chain = build_response_chain(llm)
memory_manager = MemoryManager()

# =======================
# Endpoint principal
# =======================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    message = request.message

    # Paso 1: Clasificación
    emotion = classifier_manager.classify_emotion(message)
    print(emotion)
    mental_state = classifier_manager.classify_mental_state(message)
    print(mental_state)
    intent = intent_classifier.classify_intent(message)
    print(intent)

    # Paso 2: Política
    memory = memory_manager.get_memory(user_id)
    history_str = memory.load_memory_variables({})["history"]
    print('obtuvo historia')
    action = policy_chain.run(
        emotion=emotion,
        mental_state=mental_state,
        intent=intent,
        history=history_str,
        message=message
    ).strip()
    print(action)
    # Paso 3: Generación
    response = response_chain.run(
        action=action,
        history=history_str,
        message=message
    ).strip()

    # Actualizar memoria
    memory.save_context({"input": message}, {"output": response})

    return {
        "user_id": user_id,
        "emotion": emotion,
        "mental_state": mental_state,
        "intent": intent,
        "action": action,
        "response": response
    }

# Ejecutar con:
# uvicorn app:app --reload --host 0.0.0.0 --port 8000
