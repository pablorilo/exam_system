# app.py (Gemini + PDFs para Cloud Run)
import os
import re
import time
import gradio as gr
from textwrap import dedent
from datetime import datetime
from google import genai
from google.genai import types
from google.auth import default

# ------------------------------
# Configuración
# ------------------------------
PDF_DIR = "docs"
LOG_FILE = "logs_app.txt"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# ------------------------------
# Logging
# ------------------------------
def log_event(event: str):
    timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {event}"
    print(timestamp, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(timestamp + "\n")

# ------------------------------
# Autenticación (automática en Cloud Run)
# ------------------------------
def authenticate_user():
    creds, project = default(scopes=SCOPES)
    log_event(f"✅ Usando credenciales predeterminadas de Google Cloud (proyecto: {project})")
    return creds

# ------------------------------
# Cliente Gemini autenticado
# ------------------------------
def get_gemini_client():
    creds = authenticate_user()
    client = genai.Client(
        vertexai=True,
        project="innate-diode-478014-c1",  # Tu ID de proyecto
        location="us-central1",
        credentials=creds,
    )
    return client

client = get_gemini_client()

# ------------------------------
# Función principal del chat
# ------------------------------
def chat_fn(message, history):
    if not message.strip():
        return "Por favor escribe una pregunta."

    try:
        log_event(f"💬 Pregunta recibida: {message}")
        start_time = time.time()

        # Recuperar PDFs
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        pdf_paths = [os.path.join(PDF_DIR, f) for f in pdf_files]
        pdf_map = {f"doc_{i+1}": pdf_paths[i] for i in range(len(pdf_paths))}
        log_event(f"📄 Enviando {len(pdf_paths)} PDFs: {list(pdf_map.values())}")

        documentos_disponibles = ", ".join(pdf_map.keys()) if pdf_map else "Ninguno"
        prompt_text = dedent(
            f"""\
            Eres un asistente experto que responde preguntas únicamente usando la información de los documentos proporcionados.

            Responde siempre en español.

            Si la información no está en los documentos, responde exactamente:
            "No tengo esa información en los documentos".

            Cita siempre la fuente con el formato: [doc_X, página Y].

            Documentos disponibles: {documentos_disponibles}

            Pregunta:
            {message}

            Respuesta:"""
        )

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
            log_event(f"📤 Adjuntado PDF: {pdf_path} ({len(pdf_bytes)} bytes)")

        # Llamada a Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=contents,
        )

        elapsed = round(time.time() - start_time, 2)
        answer_clean = response.text.strip()
        log_event(f"✅ Respuesta recibida ({len(answer_clean)} chars, {elapsed}s):\n{answer_clean}")

        # Buscar referencias
        matches = re.findall(r"(doc_\d+).*?(\d+)", answer_clean, flags=re.IGNORECASE)
        sources_list = [f"{os.path.basename(pdf_map[doc_id])} - pág {page}"
                        for doc_id, page in matches if doc_id in pdf_map]
        if sources_list:
            answer_clean = f"{answer_clean}\n\nFuentes:\n" + "\n".join(sources_list)

        return answer_clean

    except Exception as e:
        log_event(f"❌ Error en chat_fn: {str(e)}")
        return f"Error: {str(e)}"

# ------------------------------
# Interfaz Gradio
# ------------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    title="📄 Chat sobre curso Controller",
    description="Pregunta sobre los PDFs cargados usando Gemini.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
