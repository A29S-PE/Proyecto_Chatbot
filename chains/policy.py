from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def build_policy_chain(llm):
    policy_prompt = PromptTemplate(
        input_variables=["emotion", "mental_state", "intent", "history", "message"],
        template="""
Eres un planificador de diálogo en español que decide la acción más adecuada basándose en la emoción, el estado mental, la intención del usuario y su historial.

Dispones de la siguiente información:
- Emoción: {emotion}
- Estado mental: {mental_state}
- Intención: {intent}
- Historial: {history}
- Mensaje actual: {message}

Las acciones disponibles son:
1. ResponderEmpaticamente() → Generar una respuesta empática, reconociendo la emoción del usuario.
2. DarInformacion() → Responder con información útil o recursos de apoyo emocional.
3. Autocuidado() → Recomendar técnicas breves de autocuidado o motivación cuando los síntomas sean leves.
4. RedirigirProfesional() → Recomendar explícitamente buscar ayuda profesional si se detectan signos de depresión severa o riesgo alto (ej. estado mental "suicida" o "depresión grave").
5. Aclarar() → Consultar por más detalles si la intención o el estado emocional del usuario no son claros.
6. FueraDeDominio() → Responder educadamente que el chatbot solo brinda apoyo emocional y no responde preguntas fuera de este tema.

Reglas de decisión:
1. Si el estado mental es suicida o muestra signos de depresión severa, la acción debe ser RedirigirProfesional().
2. Si el estado mental es normal pero la intención es una de estas:  
   - mostrar_estres, mostrar_depresion, mostrar_sentimiento_de_soledad → usar Autocuidado() o ResponderEmpaticamente() según corresponda.  
3. Si el usuario expresa gratitud, saluda o busca interacción ligera → ResponderEmpaticamente().  
4. Si el usuario busca motivación explícita → Autocuidado().  
5. Si la intención es desconocido o contradictoria con la emoción/estado mental → usar Aclarar().  
6. Si el usuario hace preguntas que no están relacionadas con salud mental o emociones → FueraDeDominio().  

Devuelve solo el nombre de la acción exacta (ejemplo: `ResponderEmpaticamente()`) sin explicaciones adicionales."""
    )
    return LLMChain(llm=llm, prompt=policy_prompt)
