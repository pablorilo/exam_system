# app.py (Gemini + PDFs desde GCS para Cloud Run)
import os
import re
import time
import gradio as gr
from textwrap import dedent
from datetime import datetime
from google import genai
from google.genai import types
from google.auth import default
from google.cloud import storage

# ------------------------------
# Configuración
# ------------------------------
LOG_FILE = "logs_app.txt"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
BUCKET_NAME = "controller_docs"

# ------------------------------
# Logging
# ------------------------------
def log_event(event: str):
    timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {event}"
    print(timestamp, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(timestamp + "\n")

# ------------------------------
# Autenticación en Cloud Run
# ------------------------------
def authenticate_user():
    creds, project = default(scopes=SCOPES)
    log_event(f"✅ Credenciales predeterminadas de Google Cloud (proyecto: {project})")
    return creds

# ------------------------------
# Cliente Gemini autenticado
# ------------------------------
def get_gemini_client():
    creds = authenticate_user()
    client = genai.Client(
        vertexai=True,
        project="innate-diode-478014-c1",
        location="us-central1",
        credentials=creds,
    )
    return client

client = get_gemini_client()

# ------------------------------
# Cargar PDFs desde GCS
# ------------------------------
def load_pdfs_from_gcs(bucket_name: str, prefix: str = ""):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        pdf_blobs = [b for b in blobs if b.name.lower().endswith(".pdf")]

        pdf_map = {}
        for i, blob in enumerate(pdf_blobs):
            pdf_bytes = blob.download_as_bytes()
            pdf_map[f"doc_{i+1}"] = {
                "name": blob.name,
                "bytes": pdf_bytes
            }
            log_event(f"📄 PDF cargado: {blob.name} ({len(pdf_bytes)} bytes)")
        return pdf_map
    except Exception as e:
        log_event(f"❌ Error cargando PDFs desde GCS: {str(e)}")
        return {}

# ------------------------------
# Chat principal
# ------------------------------
def chat_fn(message, history):
    if not message.strip():
        return "Por favor escribe una pregunta."

    try:
        log_event(f"💬 Pregunta recibida: {message}")
        start_time = time.time()

        # Cargar PDFs desde GCS
        pdf_map = load_pdfs_from_gcs(BUCKET_NAME)
        log_event(f"📄 PDFs detectados: {list(v['name'] for v in pdf_map.values())}")

        # ------------------------------
        # Prompt reforzado
        # ------------------------------
        prompt_text = dedent(
                """
                Eres un asistente experto que responde preguntas tipo test y de cálculos **únicamente usando la información contenida en los documentos PDF adjuntos**.
                Tu única fuente de información son los PDFs. No puedes usar información externa.

                Reglas estrictas:

                1. Antes de responder, realiza una búsqueda **exhaustiva en todos los documentos y todas las páginas** para encontrar toda la información relevante.
                2. Interpreta cuidadosamente la pregunta y todas sus opciones. Puede ser:
                - Pregunta de teoría
                - Supuesto práctico
                - Cálculo numérico
                3. Responde **solo con la respuesta correcta**:
                - Para teoría y supuestos, devuelve la opción exacta (a, b, c, ...) o "Todas son correctas" si aplica.
                - Para cálculos, realiza los cálculos usando únicamente datos y fórmulas encontradas en los PDFs, y devuelve la respuesta numérica junto con la opción correcta (a, b, c, ...).
                4. Cita siempre las fuentes encontradas en los PDFs usando el formato [doc_X, página Y]. Si hay varias fuentes, cítalas todas.
                5. **Si después de revisar exhaustivamente todos los documentos no encuentras información**, responde exactamente:
                "No tengo esa información en los documentos".
                6. No agregues explicaciones, texto adicional ni supuestos fuera de los PDFs.
                7. Nunca asumas información que no esté explícitamente en los documentos.
                """
            )



        # ------------------------------
        # Construcción de Parts compatible
        # ------------------------------
        contents = [
            types.Part(text=prompt_text),   # Instrucciones
            types.Part(text=message)        # Pregunta concreta
        ]

        for doc_id, data in pdf_map.items():
            contents.append(
                types.Part.from_bytes(
                    data=data["bytes"],
                    mime_type="application/pdf"
                )
            )
            log_event(f"📤 PDF enviado a Gemini: {data['name']}")

        # ------------------------------
        # Llamada a Gemini
        # ------------------------------
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=contents
        )

        elapsed = round(time.time() - start_time, 2)
        answer = response.text.strip()
        log_event(f"✅ Respuesta recibida ({len(answer)} chars, {elapsed}s)\n{answer}")

        # ------------------------------
        # Extraer citas y mostrar fuentes
        # ------------------------------
        matches = re.findall(r"(doc_\d+).*?(\d+)", answer, flags=re.IGNORECASE)
        sources = [
            f"{pdf_map[doc_id]['name']} - pág {page}"
            for doc_id, page in matches if doc_id in pdf_map
        ]

        if sources:
            answer += "\n\nFuentes:\n" + "\n".join(sources)

        return answer

    except Exception as e:
        log_event(f"❌ Error en chat_fn: {str(e)}")
        return f"Error: {str(e)}"

# ------------------------------
# Interfaz Gradio
# ------------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    title="📄 Chat sobre curso Controller v5.0",
    description="Pregunta sobre los PDFs cargados desde Cloud Storage usando Gemini.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
