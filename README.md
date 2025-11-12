# Exam System – Chat con Gemini

Aplicación local que permite hacer preguntas en español sobre un conjunto de PDFs y obtener respuestas citadas usando Google Gemini vía Vertex AI. La autenticación se realiza con OAuth2 para cuentas de Google Cloud y los archivos adjuntos se cargan dinámicamente desde la carpeta `docs`.

## Requisitos previos

- Python 3.10 (se recomienda usar un entorno virtual).
- Cuenta de Google Cloud con acceso a Vertex AI habilitado.
- Archivo de credenciales OAuth (`client_secret.json`) descargado desde Google Cloud Console (tipo Desktop App).
- PDFs o materiales de referencia que se colocarán dentro de la carpeta `docs`.

## Instalación rápida

```bash
python3.10 -m venv exam_env
source exam_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Si ya existe el entorno `exam_env`, basta con activarlo y actualizar dependencias en caso necesario.

## Configuración de credenciales

1. Copia `client_secret.json` al directorio raíz del proyecto (`/home/pablo/cicerai/exam_system/`).
2. En el primer arranque, la aplicación abrirá un navegador para completar el flujo de OAuth.  
   - El token autorizado se guardará en `token.json`.
   - Si quieres renovar el token, elimina `token.json` y vuelve a lanzar la app.
3. Verifica que la variable `project` en `app.py` (`get_gemini_client()`) use el ID real de tu proyecto de Google Cloud.

## Cargar documentación

- Guarda tus PDFs dentro del directorio `docs/`.  
- Cada archivo se enviará como contexto al modelo y las respuestas citarán la fuente con el formato `[doc_X, página Y]`.

## Ejecución

```bash
source exam_env/bin/activate
python app.py
```

Esto abre una interfaz de `gradio` en `http://127.0.0.1:7860`. Si necesitas exponerla a la red local, cambia `server_name` en `app.py`.

## Registro y depuración

- Los eventos y errores se escriben en `logs_app.txt`.  
- Cada ejecución numera los PDFs enviados y registra las respuestas, lo que facilita auditar el comportamiento del modelo.

## Empaquetado opcional

En el repositorio hay especificaciones de PyInstaller (`exam_responses.spec`, `ExamResponses.spec`) y artefactos compilados en `build/` y `dist/`. Si deseas regenerar un ejecutable autónomo:

```bash
source exam_env/bin/activate
pyinstaller exam_responses.spec
```

Asegúrate de empaquetar `docs/`, `client_secret.json` y cualquier otro recurso necesario junto con el ejecutable.

## Estructura del proyecto

- `app.py`: lógica principal del chat y autenticación.
- `docs/`: repositorio de PDFs y cuestionarios.
- `logs_app.txt`: bitácora de ejecución.
- `requirements.txt`: dependencias mínimas.

## Próximos pasos sugeridos

- Ajustar el prompt o el modelo en `app.py` según tus necesidades.
- Añadir pruebas automáticas que validen la detección de referencias en las respuestas.
- Considerar una caché o indexación de PDFs para acelerar las consultas si el volumen crece.

