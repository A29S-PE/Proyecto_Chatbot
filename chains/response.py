from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def build_response_chain(llm):
    response_prompt = PromptTemplate(
        input_variables=["action", "history", "message"],
        template="""
Eres un asistente conversacional en español.
Acción a realizar: {action}
Historial de la conversación: {history}
Mensaje del usuario: {message}

Genera una respuesta natural, coherente y empática para el usuario.

Respuesta:"""
    )
    return LLMChain(llm=llm, prompt=response_prompt)
