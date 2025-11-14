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
from google.cloud import storage  # <-- GCS

# ------------------------------
# Configuración
# ------------------------------
LOG_FILE = "logs_app.txt"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
BUCKET_NAME = "controller_docs"  # <-- TU BUCKET

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

        return pdf_map
    except Exception as e:
        log_event(f"❌ Error cargando PDFs desde GCS: {str(e)}")
        return {}

# ------------------------------
# Función principal del chat
# ------------------------------
def chat_fn(message, history):
    if not message.strip():
        return "Por favor escribe una pregunta."

    try:
        log_event(f"💬 Pregunta recibida: {message}")
        start_time = time.time()

        # Cargar PDFs desde GCS
        log_event(f"📥 Cargando PDFs desde GCS: {BUCKET_NAME}")
        pdf_map = load_pdfs_from_gcs(BUCKET_NAME)

        log_event(f"📄 PDFs detectados: {list(v['name'] for v in pdf_map.values())}")

        documentos_disponibles = ", ".join(pdf_map.keys()) if pdf_map else "Ninguno"

        # Construir prompt
        prompt_text = dedent(
            f"""\
            Eres un asistente especializado cuya única tarea es encontrar información en los documentos proporcionados.

Reglas estrictas de comportamiento:

1. Debes realizar una búsqueda exhaustiva y minuciosa en todos los documentos PDF proporcionados antes de responder.
2. Solo puedes responder utilizando información que esté explícitamente en los documentos.
3. Si no puedes confirmar la respuesta directamente y sin ambigüedad en los documentos, responde exactamente:
   "No tengo esa información en los documentos".
4. No inventes ni interpretes contenido. No completes, no rellenes, no extrapoles.
5. No añadas explicaciones, contexto adicional, ni texto innecesario.  
   Tu salida debe ser solo la respuesta correcta y nada más.
6. Siempre cita la fuente en formato: [doc_X, página Y].  
   Si existen varias fuentes, cítalas todas.
7. Si la pregunta no tiene soporte literal en los documentos, usa estrictamente:
   "No tengo esa información en los documentos".

Regla especial para preguntas de opciones múltiples:

8. Si la pregunta lista opciones del estilo:
      • Afirmación 1  
      • Afirmación 2  
      • Todas son correctas  
   Y los documentos indican que todas las opciones son correctas o verdaderas, debes responder exactamente:
   "Todas son correctas"
   junto con las citas asociadas.
9. Si no puedes confirmar que todas son correctas según los documentos, entonces responde la opción correcta correspondiente; y si no puedes confirmarla, responde:
   "No tengo esa información en los documentos".

Tu respuesta final debe incluir únicamente:
- La respuesta correcta basada en los documentos (sin explicaciones)
- Las citas correspondientes en el formato indicado
ta siempre la fuente con el formato: [doc_X, página Y].

            Documentos disponibles: {documentos_disponibles}

            Pregunta:
            {message}

            Respuesta:"""
        )

        contents = [prompt_text]

        # Adjuntar PDFs
        for doc_id, data in pdf_map.items():
            contents.append(
                types.Part.from_bytes(
                    data=data["bytes"],
                    mime_type="application/pdf"
                )
            )
            log_event(f"📤 Adjuntado PDF desde GCS: {data['name']} ({len(data['bytes'])} bytes)")

        # Llamada a Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=contents,
        )

        elapsed = round(time.time() - start_time, 2)
        answer_clean = response.text.strip()

        log_event(f"✅ Respuesta recibida ({len(answer_clean)} chars, {elapsed}s)")

        # Reconocer referencias tipo [doc_X, página Y]
        matches = re.findall(r"(doc_\d+).*?(\d+)", answer_clean, flags=re.IGNORECASE)

        # Crear enlaces clickeables públicos
        sources_list = []
        for doc_id, page in matches:
            if doc_id in pdf_map:
                filename = pdf_map[doc_id]["name"]

                # Enlace público (porque tus PDFs son públicos)
                url = f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"

                sources_list.append(
                    f"[{filename} - pág {page}]({url})"
                )

        # Agregar sección de fuentes clickeables
        if sources_list:
            answer_clean += "\n\n### Fuentes:\n" + "\n".join(sources_list)

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
    description="Pregunta sobre los PDFs cargados desde Cloud Storage usando Gemini.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))
