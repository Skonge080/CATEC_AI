import os
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

# 1. Настройка путей
DATA_DIR = "./knowledge_base"  # Папка с документами колледжа (pdf, txt, docx)
DB_PATH = "./qdrant_local_db"  # Сюда сохранится база для продакшена
COLLECTION_NAME = "college_knowledge"

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"[*] Создана папка '{DATA_DIR}'. Положите туда документы для базы знаний и перезапустите скрипт.")
        return

    print("[*] Загрузка документов из папки...")
    documents = SimpleDirectoryReader(DATA_DIR,recursive=True).load_data()
    print(f"[+] Загружено документов: {len(documents)}")

    # 2. Настройка модели эмбеддингов (FastEmbed)
    # Префиксы важны для моделей семейства E5
    embed_model = FastEmbedEmbedding(
        model_name="intfloat/multilingual-e5-large",
        max_length=512
    )

    # 3. Настройка чанкера (разбиение текста)
    # chunk_size 512 отлично подходит под max_length модели e5-large
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

    # 4. Инициализация локального Qdrant (On-disk хранилище)
    client = qdrant_client.QdrantClient(path=DB_PATH)
    
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=COLLECTION_NAME
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. Сборка индекса (эмбеддинг и сохранение)
    print("[*] Индексация данных и генерация эмбеддингов (это может занять время)...")
    
    # Передаем глобальные настройки через создание индекса
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[node_parser],
        show_progress=True
    )

    print(f"[+] Успешно! База данных сохранена в папку: {DB_PATH}")
    print("[*] Вы можете запаковать эту папку и перенести на Windows Server в прод.")
    
    client.close()

if __name__ == "__main__":
    main()
