import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

def get_conversation_response(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, emotion, mental_state, intent, message, history):

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

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    prompt_length = inputs['input_ids'].shape[1]
    print(prompt_length)
    if prompt_length > 7500:
        print(f"Prompt demasiado largo: {prompt_length} tokens")
    
    torch.manual_seed(0)
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    assistant_response = generated_text.split('Respuesta:')[-1].strip()

    return assistant_response
