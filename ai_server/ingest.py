import os
import uuid
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ai_server.app.config import RAG_DB_DIR, RAG_SOURCE_DIR

# --- НАСТРОЙКИ ---
BASE_DIR = Path(__file__).resolve().parent

# Путь к вашей базе данных (должен совпадать с тем, что в main.py)
DB_PATH = str(RAG_DB_DIR)
COLLECTION_NAME = "codebase"

# Папка, которую будем сканировать (ваш проект)
# IMPORTANT: scan ONLY the RAG source folder (static by default via config)
SOURCE_DIRECTORY = str(RAG_SOURCE_DIR)

# Какие файлы читать (добавьте нужные расширения)
ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".html", ".css", ".md", ".txt", ".json", ".rs"}

# Ignored folders inside SOURCE_DIRECTORY only
IGNORE_DIRS = {".git", "__pycache__", ".idea", ".vscode"}

# Настройки нарезки (Chunking)
CHUNK_SIZE = 1000  # Символов в одном куске
CHUNK_OVERLAP = 200  # Перекрытие (чтобы не терять смысл на стыках)

# -----------------


def load_documents(source_dir):
    docs = []
    print(f"Сканирование папки: {source_dir}...")

    for root, dirs, files in os.walk(source_dir):
        # Удаляем игнорируемые папки из списка обхода
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in ALLOWED_EXTENSIONS:
                file_path = os.path.join(root, file)
                try:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                    except UnicodeDecodeError:
                        # Fallback for mixed encodings (common on Windows)
                        with open(file_path, "r", encoding="cp1251", errors="ignore") as f:
                            text = f.read()

                    if not text.strip():
                        continue  # Пропускаем пустые файлы

                    # Store paths relative to the scanned root for cleaner citations
                    rel_source = os.path.relpath(file_path, source_dir)
                    docs.append({"text": text, "source": rel_source, "filename": file})
                except Exception as e:
                    print(f"Ошибка чтения {file_path}: {e}")
    return docs


def split_text(text, chunk_size, overlap):
    """Простая функция нарезки текста с перекрытием"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Сдвигаем окно назад на overlap, чтобы захватить контекст
        start += chunk_size - overlap
    return chunks


def main():
    # 1. Инициализация БД и модели (СТРОГО CPU для экономии VRAM)
    print("Инициализация модели Embeddings (CPU)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    client = chromadb.PersistentClient(path=DB_PATH)

    # Удаляем старую коллекцию и создаем новую (полная перезапись для чистоты)
    # Если хотите ДОБАВЛЯТЬ, уберите delete_collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Старая база очищена.")
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    # 2. Чтение файлов
    raw_docs = load_documents(SOURCE_DIRECTORY)
    print(f"Найдено файлов: {len(raw_docs)}")

    # 3. Нарезка и подготовка к загрузке
    chunked_texts = []
    chunked_metadatas = []
    chunked_ids = []

    print("Нарезка текста на чанки...")
    for doc in raw_docs:
        chunks = split_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            chunked_texts.append(chunk)
            chunked_metadatas.append(
                {"source": doc["source"], "filename": doc["filename"], "chunk_id": i}
            )
            chunked_ids.append(str(uuid.uuid4()))

    if not chunked_texts:
        print("Нет данных для добавления.")
        return

    # 4. Векторизация и сохранение (батчами, чтобы не зависло)
    batch_size = 100  # Обрабатываем по 100 кусочков за раз
    total_chunks = len(chunked_texts)
    print(f"Всего чанков: {total_chunks}. Начало векторизации...")

    for i in range(0, total_chunks, batch_size):
        batch_texts = chunked_texts[i : i + batch_size]
        batch_metadatas = chunked_metadatas[i : i + batch_size]
        batch_ids = chunked_ids[i : i + batch_size]

        # Генерируем эмбеддинги
        embeddings = embedder.encode(batch_texts).tolist()

        # Добавляем в базу
        collection.add(
            documents=batch_texts, embeddings=embeddings, metadatas=batch_metadatas, ids=batch_ids
        )
        print(f"Обработано {min(i + batch_size, total_chunks)} / {total_chunks}")

    print("✅ База данных успешно обновлена!")


if __name__ == "__main__":
    main()
