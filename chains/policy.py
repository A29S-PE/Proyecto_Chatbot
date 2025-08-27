from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def build_policy_chain(llm):
    policy_prompt = PromptTemplate(
        input_variables=["emotion", "mental_state", "intent", "history", "message"],
        template="""
Eres un planificador de diálogo en español.
Entrada:
- Emoción: {emotion}
- Estado mental: {mental_state}
- Intención: {intent}
- Historial: {history}
- Mensaje actual: {message}

Decide la acción más adecuada. Ejemplos:
- responder_empaticamente
- dar_informacion
- redirigir_profesional
- smalltalk

Acción:"""
    )
    return LLMChain(llm=llm, prompt=policy_prompt)
