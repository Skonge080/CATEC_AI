import os
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores.types import VectorStoreQueryMode
import qdrant_client

IS_DEVELOPMENT = True 
DB_PATH = "./qdrant_local_db"
COLLECTION_NAME = "college_knowledge"

QA_PROMPT_TMPL = """Ты — официальный ИИ-ассистент колледжа. Твоя задача — давать точные, вежливые и структурированные ответы на основе предоставленного текста.

Строгие правила:
1. Отвечай ТОЛЬКО на русском языке.
2. Используй ТОЛЬКО контекст внутри тегов <context>. Если информации нет — пиши "Нет данных".
3. Данные разбиты по тегам <document name="имя_файла">. Обращай внимание на атрибут name, чтобы понимать, к какому файлу относится текст.
4. Если тебя просят перечислить специальности или квалификации, выведи их строгим списком на основе имён документов и текста внутри них. Не перефразируй названия, не объединяй пункты и выводи ВСЕ найденные уникальные направления.

<context>
{context_str}
</context>

Вопрос пользователя: {query_str}
Ответ:"""

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

def init_assistant():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    embed_model = FastEmbedEmbedding(model_name="intfloat/multilingual-e5-large", max_length=512)
    Settings.embed_model = embed_model

    if IS_DEVELOPMENT:
        print("[*] Режим разработки: подключаем локальную Ollama (qwen2.5:7b)...")
        llm = Ollama(model="qwen2.5:7b", request_timeout=120.0, temperature=0.0)
    else:
        print("[*] Режим продакшена: подключаем внешнее API...")
        from llama_index.llms.openai import OpenAI
        llm = OpenAI(model="your-prod-model-name", api_key="your_api_key", api_base="https://api.yourassistant.com/v1", temperature=0.0)
        
    Settings.llm = llm

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"База данных не найдена по пути {DB_PATH}.")
        
    client = qdrant_client.QdrantClient(path=DB_PATH, read_only=True)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    query_engine = index.as_query_engine(
        similarity_top_k=15, 
        vector_store_query_mode=VectorStoreQueryMode.MMR,
        vector_store_kwargs={"mmr_prefetch_k": 30},
        text_qa_template=QA_PROMPT
    )
    
    return query_engine

if __name__ == "__main__":
    try:
        assistant = init_assistant()
        print("[+] ИИ Ассистент готов к работе!")
        print("-" * 50)
        
        while True:
            user_query = input("\nЗапрос: ")
            if user_query.lower() in ['exit', 'quit', 'выход']:
                break
                
            if not user_query.strip():
                continue
                
            print("[*] Поиск...")
            
            retriever = assistant._retriever
            nodes = retriever.retrieve(user_query)
            
            print("\n" + "="*20 + " ДИАГНОСТИКА ПОИСКА " + "="*20)
            
            # 1. Группируем чанки по имени файла, чтобы убрать дубликаты тегов
            from collections import defaultdict
            file_chunks = defaultdict(list)
            
            for idx, node_with_score in enumerate(nodes, 1):
                node = node_with_score.node
                score = node_with_score.score
                file_name = node.metadata.get('file_name', 'Неизвестный файл')
                text_content = node.get_content(metadata_mode=MetadataMode.NONE).strip()
                
                # Добавляем чанк, только если такого текста для этого файла еще не было
                if text_content not in file_chunks[file_name]:
                    file_chunks[file_name].append(text_content)
                
                print(f"[{idx}] Файл: {file_name} | Score: {score:.4f}")
            print("="*80 + "\n")
            
            # 2. Собираем чистый контекст: один файл — один тег <document>
            custom_context_list = []
            for file_name, chunks in file_chunks.items():
                merged_text = "\n\n".join(chunks)
                formatted_chunk = f'<document name="{file_name}">\n{merged_text}\n</document>'
                custom_context_list.append(formatted_chunk)
            
            if not custom_context_list:
                context_str = "Нет данных"
            else:
                context_str = "\n".join(custom_context_list)
            
            # Прямой запрос к модели с вашим шаблоном
            response = Settings.llm.predict(
                QA_PROMPT,
                context_str=context_str,
                query_str=user_query
            )
            
            print(f"[Ассистент]:\n{response}")
            
    except Exception as e:
        print(f"[-] Ошибка: {e}")
