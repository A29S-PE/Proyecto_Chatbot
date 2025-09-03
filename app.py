from fastapi import FastAPI
from schemas.chat import ChatRequest
from models.classifier import ClassifierManager
from models.intent import IntentClassifier
from models.llm import load_llm
from chains.policy import get_conversation_action
from chains.response import get_conversation_response
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
pipe = load_llm()
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
    print('La emocion es: ', emotion)
    mental_state = classifier_manager.classify_mental_state(message)
    print('El estado mental es: ',mental_state)
    intent = intent_classifier.classify_intent(message)
    print('La intencion es: ',intent)

    # Paso 2: Política
    memory = memory_manager.get_memory(user_id)
    history_str = memory.load_memory_variables({})["history"]
    print('El historial es: ', history_str[0:30])
    action = get_conversation_action(pipe,emotion,mental_state,intent,message,history_str).strip()
    print('La acción es: ', action)
    # Paso 3: Generación
    response = get_conversation_response(pipe,action,message,history_str).strip()
    print('Respuesta: ', response)
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
