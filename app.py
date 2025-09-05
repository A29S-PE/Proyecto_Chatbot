from fastapi import FastAPI
from models.classifier import ClassifierManager
from models.intent import IntentClassifier
from models.llm import load_llm
from chains.policy_and_response import get_conversation_response
from memory.memory import MemoryManager, format_history
from pydantic import BaseModel
import time

app = FastAPI(
    title="Chatbot Conversacional",
    description="API Chatbot",
    version="1.0.0"
)

# Modelo de entrada
class ChatRequest(BaseModel):
    user_id: str
    message: str

# Modelo de salida
class ChatResponse(BaseModel):
    user_id: str
    emotion: str
    mental_state: str
    intent: str
    response: str

classifier_manager = ClassifierManager()
intent_classifier = IntentClassifier()
pipe = load_llm()
memory_manager = MemoryManager()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    message = request.message

    start_time = time.time()
    
    # Paso 1: Clasificación
    emotion = classifier_manager.classify_emotion(message)
    print('La emocion es: ', emotion)
    mental_state = classifier_manager.classify_mental_state(message)
    print('El estado mental es: ',mental_state)
    intent = intent_classifier.classify_intent(message)
    print('La intencion es: ',intent)

    # Paso 2: Política y Generación
    memory = memory_manager.get_memory(user_id)
    history_msgs = memory.load_memory_variables({})["history"]
    history_text = format_history(history_msgs)
    print('El historial es: ', history_text[0:100])
    response = get_conversation_response(pipe,emotion,mental_state,intent,message,history_text).strip()
    print('Respuesta: ', response)
    # Actualizar memoria
    memory.save_context({"input": message}, {"output": response})

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tiempo de respuesta: {elapsed_time:.2f} segundos")

    return {
        "user_id": user_id,
        "emotion": emotion,
        "mental_state": mental_state,
        "intent": intent,
        "response": response
    }

# uvicorn app:app --reload --host 0.0.0.0 --port 8000
