# Sistema de Preguntas sobre Documentos

Aplicación Gradio que permite consultar documentos PDF/DOCX mediante un flujo
RAG respaldado por embeddings en `FAISS` y respuestas servidas por Ollama.

## Requisitos

- Python 3.10 (usar `exam_env/` para aislar dependencias).
- Ollama en ejecución local con el modelo `mistral` descargado.
- Documentos en formato PDF o DOCX dentro de la carpeta `docs/`.

## Ingesta de documentos

```bash
source exam_env/bin/activate
python ingest.py
```

El script:

- Carga y divide los documentos en fragmentos (`chunk_size=1200`, solapamiento de
  `200` caracteres).
- Genera embeddings con `intfloat/multilingual-e5-base` normalizados.
- Guarda el índice FAISS en `vector_store/`.

## Ejecución de la app

```bash
source exam_env/bin/activate
python app.py
```

La interfaz (`gr.ChatInterface`) muestra las respuestas del modelo junto con una
lista de fragmentos citados (`[S1]`, `[S2]`, …) indicando el archivo y la página
de origen.

## Notas de mantenimiento

- Asegúrate de regenerar el índice tras añadir o actualizar documentos.
- El cliente HTTP reutiliza una sesión para reducir la latencia en Ollama y eleva
  el límite de tokens (`num_predict=800`) para evitar respuestas truncadas.
- Ante un error de conexión, verifica que `ollama serve` esté en ejecución.


