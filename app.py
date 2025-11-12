# app.py (Gemini + PDFs completos, logging robusto y fuentes fiables)
import os
import re
import time
import gradio as gr
from datetime import datetime
from google import genai
from google.genai import types

# ------------------------------
# Configuración
# ------------------------------
VECTOR_DIR = "vector_store"  # opcional, ya no se usa FAISS
PDF_DIR = "docs"
LOG_FILE = "logs_app.txt"

# ------------------------------
# Cliente Google Gemini
# ------------------------------
client = genai.Client(
    vertexai=True,
    project="innate-diode-478014-c1",
    location="us-central1",
)

# ------------------------------
# Logging
# ------------------------------


def log_event(event: str):
    timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {event}"
    print(timestamp)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(timestamp + "\n")

# ------------------------------
# Función principal del chat
# ------------------------------


def chat_fn(message, history):
    if not message.strip():
        return "Por favor escribe una pregunta."

    try:
        log_event(f"💬 Pregunta recibida: {message}")
        start_time = time.time()

        # ------------------------------
        # Recuperar PDFs de la carpeta docs
        # ------------------------------
        pdf_files = [
            f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")
        ]
        pdf_paths = [os.path.join(PDF_DIR, f) for f in pdf_files]
        pdf_map = {f"doc_{i+1}": pdf_paths[i] for i in range(len(pdf_paths))}
        log_event(
            f"Se van a enviar {len(pdf_paths)} PDFs a Gemini: "
            f"{list(pdf_map.values())}"
        )

        # ------------------------------
        # Construir prompt
        # ------------------------------
        prompt_text = (
            "Eres un asistente experto que responde únicamente con base en los documentos proporcionados.\n\n"
            "Instrucciones:\n"
            "- Responde en español.\n"
            "- Si no hay información suficiente, responde literalmente: "
            '"No tengo esa información en los documentos".\n'
            "- Solo responde con la opción correcta y la fuente EXACTA usando los identificadores de documento proporcionados.\n"
            "- Usa el formato: [doc_X, página Y].\n"
            f"Identificadores disponibles: {', '.join(pdf_map.keys())}\n\n"
            f"Pregunta del usuario:\n{message}\n\nRespuesta:"
        )

        # ------------------------------
        # Preparar contenido para Gemini
        # ------------------------------
        contents = [prompt_text]
        for pdf_path in pdf_paths:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            contents.append(
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf"
                )
            )
            log_event(
                f"Adjuntado PDF al prompt: {pdf_path} "
                f"(tamaño {len(pdf_bytes)} bytes)"
            )

        # ------------------------------
        # Llamada a Gemini
        # ------------------------------
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=contents,
        )
        elapsed = round(time.time() - start_time, 2)
        answer_clean = response.text.strip()
        log_event(
            f"✅ Respuesta recibida de Gemini "
            f"(longitud {len(answer_clean)}) en {elapsed}s:\n{answer_clean}"
        )

        # ------------------------------
        # Extraer doc y página usando los identificadores
        # ------------------------------
        matches = re.findall(
            r"(doc_\d+).*?(\d+)",
            answer_clean,
            flags=re.IGNORECASE
        )
        log_event(f"Matches doc-página encontrados: {matches}")

        # Mapear a PDFs reales
        sources_list = [f"{os.path.basename(pdf_map[doc_id])} - pág {page}"
                        for doc_id, page in matches if doc_id in pdf_map]
        log_event(f"Fuentes mapeadas: {sources_list}")

        # Añadir fuentes al final de la respuesta
        if sources_list:
            answer_clean = f"{answer_clean}\n\nFuentes:\n" + "\n".join(
                sources_list
            )

        return answer_clean

    except Exception as e:
        log_event(f"❌ Error en chat_fn: {str(e)}")
        return f"Error: {str(e)}"

# ------------------------------
# Interfaz Gradio
# ------------------------------


demo = gr.ChatInterface(
    fn=chat_fn,
    title="📄 Chat sobre tus Documentos (Gemini)",
    description=(
        """Pregunta sobre el contenido completo de los PDFs
        cargados en la carpeta docs"""
    ),
    examples=[
        "¿De qué tratan los documentos?",
        "Resume el contenido principal",
    ],
)

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1")
