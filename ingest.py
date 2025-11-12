# ingest.py (optimizado)
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

DOCS_DIR = "docs"
VECTOR_DIR = "vector_store"


def load_documents():
    docs = []
    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs


def main():
    documents = load_documents()
    if not documents:
        print("⚠️ No se encontraron documentos en la carpeta 'docs'")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )
    texts = splitter.split_documents(documents)
    print(f"📄 {len(texts)} fragmentos creados")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        encode_kwargs={"normalize_embeddings": True}
    )

    db = FAISS.from_documents(texts, embeddings)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    db.save_local(VECTOR_DIR)
    print(f"✅ Vector store guardado en {VECTOR_DIR}")


if __name__ == "__main__":
    main()
