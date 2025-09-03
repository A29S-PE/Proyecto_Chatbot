from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def build_response_chain(llm):
    response_prompt = PromptTemplate(
        input_variables=["action", "history", "message"],
        template="""
Eres un asistente conversacional en español especializado en apoyo emocional. 
Debes seguir exactamente la acción indicada por el planificador de diálogo, sin inventar nuevas acciones.

Acción: {action}
Historial de la conversación: {history}
Mensaje del usuario: {message}

Instrucciones para generar la respuesta:
- Si la acción es ResponderEmpaticamente(): responde de forma cálida, comprensiva y validando la emoción del usuario.
- Si la acción es DarInformacion(): ofrece información clara, breve y relevante sobre apoyo emocional o técnicas de manejo del estado de ánimo.
- Si la acción es Autocuidado(): sugiere una técnica breve de autocuidado (ejemplo: respiración, escribir pensamientos, caminar) o un mensaje motivacional adaptado al contexto.
- Si la acción es RedirigirProfesional(): recomienda amablemente y con cuidado que el usuario busque ayuda profesional o hable con un especialista.
- Si la acción es Aclarar(): pide al usuario, de manera respetuosa, que comparta más detalles para poder entender mejor cómo se siente o qué necesita.
- Si la acción es FueraDeDominio(): explica de manera educada y empática que el chatbot solo brinda apoyo emocional y no responde preguntas de otros temas.

Reglas importantes:
1. Nunca ignores la acción recibida.  
2. La respuesta debe sonar natural, humana y empática.  
3. No incluyas explicaciones sobre qué acción estás usando, solo genera la respuesta directamente para el usuario.  

Respuesta:"""
    )
    return LLMChain(llm=llm, prompt=response_prompt)



import torch
from transformers import Pipeline

def get_conversation_response(pipe: Pipeline, action, message, history):

    def generate_assistant_response(messages, return_full_text=False, num_return_sequences=4):
        torch.manual_seed(0)
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(
            prompt, max_new_tokens=512, num_return_sequences=num_return_sequences, do_sample=True,
            temperature=0.7, top_k=50, top_p=0.95, return_full_text=return_full_text,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        return outputs, prompt

    messages = [
        {
            "role": "system",
            "content": f"""
                Eres un asistente conversacional en español especializado en apoyo emocional. 
                Debes seguir exactamente la acción indicada por el planificador de diálogo, sin inventar nuevas acciones.

                Acción: {action}
                Historial de la conversación: {history}
                Mensaje del usuario: {message}

                Instrucciones para generar la respuesta:
                - Si la acción es ResponderEmpaticamente(): responde de forma cálida, comprensiva y validando la emoción del usuario.
                - Si la acción es DarInformacion(): ofrece información clara, breve y relevante sobre apoyo emocional o técnicas de manejo del estado de ánimo.
                - Si la acción es Autocuidado(): sugiere una técnica breve de autocuidado (ejemplo: respiración, escribir pensamientos, caminar) o un mensaje motivacional adaptado al contexto.
                - Si la acción es RedirigirProfesional(): recomienda amablemente y con cuidado que el usuario busque ayuda profesional o hable con un especialista.
                - Si la acción es Aclarar(): pide al usuario, de manera respetuosa, que comparta más detalles para poder entender mejor cómo se siente o qué necesita.
                - Si la acción es FueraDeDominio(): explica de manera educada y empática que el chatbot solo brinda apoyo emocional y no responde preguntas de otros temas.

                Reglas importantes:
                1. Nunca ignores la acción recibida.  
                2. La respuesta debe sonar natural, humana y empática.  
                3. No incluyas explicaciones sobre qué acción estás usando, solo genera la respuesta directamente para el usuario.  

                Respuesta:"""
        }
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs, prompt = generate_assistant_response(messages, return_full_text=False, num_return_sequences=1)

    assistant_response = outputs[0]["generated_text"]

    return assistant_response
