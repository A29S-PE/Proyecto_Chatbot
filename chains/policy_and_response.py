import torch
from transformers import Pipeline

def get_conversation_response(pipe: Pipeline, emotion, mental_state, intent, message, history):

    def generate_assistant_response(messages, return_full_text=False, num_return_sequences=1):
        torch.manual_seed(0)
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(
            prompt, max_new_tokens=256, num_return_sequences=num_return_sequences, do_sample=True,
            temperature=0.7, top_k=50, top_p=0.95, return_full_text=return_full_text,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        return outputs
    
    messages = [
        {
            "role": "system",
            "content": f"""
            Eres un asistente conversacional en español especializado en apoyo emocional.  
            
            Recibiras siguiente información del usuario:
              - Emoción
              - Estado mental
              - Intención
              - Historial
              - Mensaje actual

            Tu tarea es:
                1. Analizar la emoción, el estado mental, la intención, el historial y el mensaje del usuario.  
                2. Decidir internamente la acción más adecuada de la siguiente lista:  
                    - ResponderEmpaticamente(): responder de forma cálida y comprensiva.  
                    - DarInformacion(): dar información breve y útil sobre apoyo emocional o técnicas de manejo del ánimo.  
                    - Autocuidado(): sugerir una técnica breve de autocuidado (ejemplo: respiración, escribir, caminar) o motivación.  
                    - RedirigirProfesional(): recomendar con cuidado que busque ayuda profesional si el estado es “suicida” o “depresión grave”.  
                    - Aclarar(): pedir más detalles si no está claro lo que siente o necesita.  
                    - FueraDeDominio(): explicar educadamente que solo apoyas en temas emocionales.  

            Reglas clave:
            - Si el estado mental es "suicida" o "depresión grave" → usar RedirigirProfesional().  
            - Si el usuario muestra estrés, depresión leve o soledad → usar Autocuidado() o ResponderEmpaticamente() según corresponda.
            - Si saluda, agradece o busca interacción ligera → usar ResponderEmpaticamente().  
            - Si busca motivación explícita → usar Autocuidado().  
            - Si la intención es desconocida o contradictoria → usar Aclarar().  
            - Si pregunta algo fuera de salud mental → usar FueraDeDominio().  

            Decide la acción internamente, pero no la menciones en tu salida. Solo devuelve una respuesta natural, coherente y empática para el usuario.
            """
        },
        {
            "role": "user",
            "content": f"""
                Emoción: {emotion}  
                Estado mental: {mental_state}  
                Intención: {intent}  
                Historial: {history}  
                Mensaje actual: {message}  
                Respuesta:
            """
        }
    ]
