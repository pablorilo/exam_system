# Imagen base
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Copiar archivos
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto
EXPOSE 8080

# Comando de inicio
CMD ["python", "app.py"]
