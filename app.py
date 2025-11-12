# app.py (Gemini + PDFs + OAuth2 local sin gcloud)
import os
import re
import time
import gradio as gr
from datetime import datetime
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# ------------------------------
# Configuración
# ------------------------------
PDF_DIR = "docs"
LOG_FILE = "logs_app.txt"
TOKEN_FILE = "token.json"
CLIENT_SECRET_FILE = "client_secret.json"  # Este lo descargas de Google Cloud Console
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# ------------------------------
# Logging
# ------------------------------
def log_event(event: str):
    timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {event}"
    print(timestamp)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(timestamp + "\n")

# ------------------------------
# Autenticación OAuth2
# ------------------------------
def authenticate_user():
    creds = None
    if os.path.exists(TOKEN_FILE):
        log_event("🔑 Cargando token guardado...")
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # Si no hay token o es inválido, iniciar login
    if not creds or not creds.valid:
        log_event("🌐 No hay credenciales válidas, iniciando login en navegador...")
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
        log_event("✅ Autenticación completada y token guardado.")
    else:
        log_event("✅ Credenciales válidas cargadas.")

    return creds

# ------------------------------
# Cliente Gemini autenticado
# ------------------------------
def get_gemini_client():
    creds = authenticate_user()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = TOKEN_FILE  # Reutilizar token
    client = genai.Client(
        vertexai=True,
        project="innate-diode-478014-c1",  # <-- Cambia por tu ID de proyecto
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

        # Prompt
        prompt_text = (
            "Eres un asistente experto que responde únicamente con base en los documentos proporcionados.\n\n"
            "Instrucciones:\n"
            "- Responde en español.\n"
            "- Si no hay información suficiente, responde literalmente: "
            '"No tengo esa información en los documentos".\n'
            "- Usa el formato: [doc_X, página Y] para indicar fuentes.\n\n"
            f"Identificadores disponibles: {', '.join(pdf_map.keys())}\n\n"
            f"Pregunta:\n{message}\n\nRespuesta:"
        )

        # Construir contenido
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
        log_event(f"🔎 Matches doc-página: {matches}")

        sources_list = [f"{os.path.basename(pdf_map[doc_id])} - pág {page}"
                        for doc_id, page in matches if doc_id in pdf_map]
        log_event(f"📚 Fuentes: {sources_list}")

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
    title="📄 Chat sobre tus Documentos (Gemini OAuth2)",
    description="Inicia sesión con Google y pregunta sobre tus PDFs.",
)

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1")
