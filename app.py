import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import os

VECTOR_DIR = "vector_store"
OLLAMA_URL = "http://localhost:11434/api/generate"

print("🔄 Cargando sistema...")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    encode_kwargs={"normalize_embeddings": True}
)
db = FAISS.load_local(
    VECTOR_DIR,
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = db.as_retriever(search_kwargs={"k": 3})
print("✅ Sistema listo")

_session = requests.Session()


def _retrieve_documents(query: str):
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    raise AttributeError(
        "Retriever no implementa invoke() ni get_relevant_documents()"
    )


def query_ollama(prompt, timeout=160):
    """Consulta directa a Ollama con timeout"""
    try:
        response = _session.post(
            OLLAMA_URL,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,  # CRÍTICO: deshabilitar streaming
                "options": {
                    "temperature": 0,
                    "num_predict": 800,
                },
            },
            timeout=timeout
        )

        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error {response.status_code}: {response.text}"

    except requests.exceptions.Timeout:
        return "⏱️ Timeout: El modelo tardó más de 60 segundos"
    except requests.exceptions.ConnectionError:
        return (
            "❌ No se puede conectar a Ollama. ¿Está corriendo 'ollama serve'?"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"


def chat_fn(message, history):
    if not message.strip():
        return "Por favor escribe una pregunta."

    try:
        print(f"🔍 Buscando contexto para: {message}")
        docs = _retrieve_documents(message)

        if not docs:
            return "No encontré información relevante en los documentos."

        context_blocks = []
        source_lines = []

        for idx, doc in enumerate(docs[:3], start=1):
            content_snippet = doc.page_content.strip()
            if len(content_snippet) > 900:
                content_snippet = content_snippet[:900] + "..."

            source = doc.metadata.get("source") or "desconocido"
            page = doc.metadata.get("page")

            if isinstance(page, int):
                # PyPDFLoader pages are 0-indexed
                page_label = f"pág. {page + 1}"
            else:
                page_label = "pág. ?"

            source_name = os.path.basename(source)
            block_tag = f"[S{idx}]"

            context_blocks.append(
                f"{block_tag} ({source_name}, {page_label}):\n"
                f"{content_snippet}"
            )
            source_lines.append(
                f"{block_tag} {source_name} ({page_label})"
            )

        context = "\n\n".join(context_blocks)

        prompt = (
            "Eres un asistente experto que responde únicamente con base en "
            "los fragmentos proporcionados.\n\n"
            "Contexto:\n"
            f"{context}\n\n"
            "Instrucciones:\n"
            "- Responde en español.\n"
            "- Usa un tono claro y directo.\n"
            "- Si no hay información suficiente, responde literalmente: "
            '"No tengo esa información en los documentos".\n'
            "- Tu respuesta unicamnete deberá ser la opcion correcta y una breve explicacion.Ejemplo: B) xxxx, porque xxxx"
            "- Cuando cites información, referencia el identificador del "
            "fragmento correspondiente (por ejemplo, [S1]).\n\n"
            "Pregunta del usuario:\n"
            f"{message}\n\n"
            "Respuesta:"
        )

        print("📤 Enviando a Ollama...")
        answer = query_ollama(prompt, timeout=60)
        print("✅ Respuesta recibida")

        answer_clean = answer.strip()
        if source_lines:
            sources_text = "\n".join(source_lines)
            answer_clean = f"{answer_clean}\n\nFuentes:\n{sources_text}"

        return answer_clean

    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Error: {str(e)}"


# Interfaz con ejemplos
demo = gr.ChatInterface(
    fn=chat_fn,
    title="📄 Chat sobre tus Documentos",
    description=(
        "Pregunta sobre el contenido de los documentos en la carpeta 'docs'"
    ),
    examples=[
        "¿De qué tratan los documentos?",
        "Resume el contenido principal"
    ],

)

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1")
