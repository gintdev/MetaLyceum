import logging
import os
from typing import List, Dict, Optional
import numpy as np

from app.services.embedders.openai_embedder import OpenAIEmbedder
from app.services.pdf_processor import PDFProcessor
from app.services.text_chunker import TextChunker
from app.services.vector_store import QdrantVectorStore
from app.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG (Retrieval Augmented Generation) сервис
    Интегрирует: embedder, PDF processor, text chunker, vector store
    """

    async def translate_to_english(self, text: str) -> str:
        """
        Перевести текст на английский язык через OpenAI Chat API
        """
        try:
            logger.info(f"🌐 Перевод запроса на английский: {text[:50]}...")
            system_prompt = "You are a translation assistant. Translate the following text to English. Return only the translation, no explanations."
            user_message = text
            response = await self.embedder.openai_client.chat.completions.create(
                model=settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,
                max_tokens=512
            )
            if response and hasattr(response, 'choices') and response.choices:
                translation = response.choices[0].message.content.strip()
                logger.info(f"✓ Перевод выполнен: {translation[:50]}")
                return translation
            else:
                logger.warning("⚠️ Не удалось получить перевод через OpenAI")
                return text
        except Exception as e:
            logger.error(f"✗ Ошибка при переводе: {str(e)}")
            return text

    async def hybrid_search(
        self,
        query: str,
        limit: int = settings.SEARCH_LIMIT,
        translation_limit: int = 5,
        final_limit: int = None
    ) -> List[Dict]:
        """
        Гибридный поиск: ищет по оригинальному и переведённому запросу, объединяет и сортирует результаты по score
        Args:
            query: исходный запрос
            limit: сколько результатов брать по оригиналу
            translation_limit: сколько брать по переводу
            final_limit: сколько вернуть в итоге (если None, то limit+translation_limit)
        Returns:
            Список уникальных чанков, отсортированных по score
        """
        logger.info(f"🔍 Гибридный поиск: оригинал + перевод для '{query[:50]}'")
        # 1. Поиск по оригиналу
        orig_results = await self.search(query, limit)
        # 2. Перевести запрос
        translated_query = await self.translate_to_english(query)
        print(f"🔍 DEBUG: Translated query: {translated_query}")
        # 3. Поиск по переводу (если перевод отличается)
        if translated_query.strip().lower() != query.strip().lower():
            trans_results = await self.search(translated_query, translation_limit)
        else:
            trans_results = []
        # 4. Объединить и дедуплицировать по (article_id, chunk_text)
        seen = set()
        all_results = []
        for chunk in orig_results + trans_results:
            key = (chunk.get('article_id'), chunk.get('chunk_text'))
            if key not in seen:
                seen.add(key)
                all_results.append(chunk)
        # 5. Отсортировать по score (по убыванию)
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        # 6. Ограничить итоговое количество
        if final_limit is None:
            final_limit = limit + translation_limit
        return all_results[:final_limit]
    
    def __init__(
        self,
        embedder: Optional[OpenAIEmbedder] = None,
        pdf_processor: Optional[PDFProcessor] = None,
        text_chunker: Optional[TextChunker] = None,
        vector_store: Optional[QdrantVectorStore] = None,
    ):
        """
        Инициализация RAG сервиса
        
        Args:
            embedder: OpenAI embedder инстанс
            pdf_processor: PDF processor инстанс
            text_chunker: Text chunker инстанс
            vector_store: Qdrant vector store инстанс
        """
        self.embedder = embedder or OpenAIEmbedder(
            model_name=settings.OPENAI_MODEL,
            dimensions=settings.OPENAI_EMBEDDING_DIMENSIONS,
            quantization=settings.OPENAI_QUANTIZATION,
        )
        self.pdf_processor = pdf_processor or PDFProcessor()
        self.text_chunker = text_chunker or TextChunker()
        self.vector_store = vector_store or QdrantVectorStore()
    
    async def initialize(self):
        """Инициализировать RAG сервис (создать коллекцию Qdrant)"""
        try:
            await self.vector_store.initialize_collection()
            logger.info("✓ RAG сервис инициализирован")
        except Exception as e:
            logger.error(f"✗ Ошибка при инициализации RAG сервиса: {str(e)}")
            raise
    
    async def process_article(
        self,
        article_id: int,
        filename: str,
    ) -> Dict:
        """
        Полная обработка статьи: скачивание -> парсинг -> chunking -> embedding -> сохранение
        
        Args:
            article_id: ID статьи в БД
            filename: Имя файла на Яндекс Диске
            
        Returns:
            Словарь с результатами обработки
        """
        try:
            logger.info(f"🚀 Начало обработки статьи {article_id}: {filename}")
            
            # 1. Скачать PDF и извлечь текст
            logger.info("1️⃣ Скачивание и извлечение текста из PDF...")
            text = await self.pdf_processor.process_pdf(filename)
            logger.info(f"   ✓ Текст извлечен: {len(text)} символов")
            
            # 2. Разбить текст на чанки
            logger.info("2️⃣ Разбиение текста на чанки...")
            chunks = self.text_chunker.chunk_text_with_metadata(
                text=text,
                article_id=article_id,
                filename=filename
            )
            logger.info(f"   ✓ Создано {len(chunks)} чанков")
            
            # 3. Получить embeddings для каждого чанка
            logger.info("3️⃣ Генерирование embeddings...")
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]
            embeddings = await self.embedder.embed_texts(chunk_texts)
            logger.info(f"   ✓ Embeddings созданы: {len(embeddings)} векторов")
            
            # 4. Сохранить в Qdrant
            logger.info("4️⃣ Сохранение в векторной БД...")
            point_ids = await self.vector_store.add_vectors(embeddings, chunks)
            logger.info(f"   ✓ Сохранено {len(point_ids)} точек в Qdrant")
            
            logger.info(f"✅ Статья {article_id} успешно обработана")
            
            return {
                "article_id": article_id,
                "filename": filename,
                "text_length": len(text),
                "chunks_count": len(chunks),
                "embeddings_count": len(embeddings),
                "stored_points": len(point_ids),
                "point_ids": point_ids,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"✗ Ошибка при обработке статьи {article_id}: {str(e)}")
            raise
    
    async def reprocess_article(
        self,
        article_id: int,
        filename: str,
    ) -> Dict:
        """
        Переобработка статьи: удалить старые данные и обработать заново
        
        Args:
            article_id: ID статьи в БД
            filename: Имя файла на Яндекс Диске
            
        Returns:
            Словарь с результатами обработки
        """
        try:
            logger.info(f"🔄 Переобработка статьи {article_id}...")
            
            # Удалить старые векторы
            await self.vector_store.delete_by_article_id(article_id)
            
            # Обработать заново
            result = await self.process_article(article_id, filename)
            logger.info(f"✅ Статья {article_id} переобработана")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Ошибка при переобработке статьи {article_id}: {str(e)}")
            raise
    
    async def search(
        self,
        query: str,
        limit: int = settings.SEARCH_LIMIT,
    ) -> List[Dict]:
        """
        Поиск по запросу: embedding -> поиск в Qdrant
        
        Args:
            query: Текстовый запрос
            limit: Количество результатов
            
        Returns:
            Список найденных документов
        """
        try:
            logger.info(f"🔍 Поиск по запросу: {query[:50]}...")
            
            # 1. Получить embedding для запроса
            query_embedding = await self.embedder.get_embedding(query)
            
            # 2. Найти похожие документы
            results = await self.vector_store.search(query_embedding, limit)
            
            logger.info(f"✓ Найдено {len(results)} релевантных документов")
            return results
            
        except Exception as e:
            logger.error(f"✗ Ошибка при поиске: {str(e)}")
            raise
    
    async def search_in_article(
        self,
        query: str,
        article_id: int,
        limit: int = settings.SEARCH_LIMIT,
    ) -> List[Dict]:
        """
        Поиск по запросу в конкретной статье
        
        Args:
            query: Текстовый запрос
            article_id: ID статьи для фильтрации
            limit: Количество результатов
            
        Returns:
            Список найденных документов в статье
        """
        try:
            logger.info(f"🔍 Поиск по запросу в статье {article_id}: {query[:50]}...")
            
            # 1. Получить embedding для запроса
            query_embedding = await self.embedder.get_embedding(query)
            
            # 2. Найти похожие документы в статье
            results = await self.vector_store.search_by_article_id(
                query_embedding,
                article_id,
                limit
            )
            
            logger.info(f"✓ Найдено {len(results)} документов в статье {article_id}")
            return results
            
        except Exception as e:
            logger.error(f"✗ Ошибка при поиске в статье {article_id}: {str(e)}")
            raise
    
    async def get_vector_store_info(self) -> Dict:
        """Получить информацию о векторной БД"""
        try:
            info = await self.vector_store.get_collection_info()
            return info
        except Exception as e:
            logger.error(f"✗ Ошибка при получении информации: {str(e)}")
            raise
    
    async def generate_answer(
        self,
        query: str,
        limit: int = settings.SEARCH_LIMIT,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Полный RAG цикл: гибридный поиск чанков -> генерация ответа
        """
        try:
            logger.info(f"🔄 RAG запрос (гибрид): {query[:50]}...")
            # 1. Гибридный поиск (оригинал + перевод)
            search_results = await self.hybrid_search(query, limit=limit, translation_limit=limit, final_limit=limit)
            if not search_results:
                logger.warning("⚠️ Релевантные чанки не найдены")
                return {
                    "answer": "К сожалению, в базе знаний не найдено релевантной информации.",
                    "chunks": [],
                    "status": "no_results"
                }
            logger.info(f"✓ Найдено {len(search_results)} чанков для контекста (гибрид)")
            # 2. Собрать контекст из чанков (сократить до 2000 символов)
            context_parts = []
            total_length = 0
            max_context_length = 2000
            for i, chunk in enumerate(search_results, 1):
                text = chunk.get('chunk_text') or chunk.get('text') or ''
                text = text[:300]
                if total_length + len(text) > max_context_length:
                    break
                context_parts.append(f"[Источник {i}] {text}")
                total_length += len(text)
            context = "\n\n".join(context_parts)
            # 3. Создать prompt для OpenAI
            if system_prompt is None:
                try:
                    app_dir = os.path.dirname(os.path.dirname(__file__))
                    prompt_path = os.path.join(app_dir, "prompts", "basic_request_prompt.txt")
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        system_prompt = f.read()
                    logger.info("✓ Загрузили system_prompt из файла basic_request_prompt.txt")
                except Exception as e:
                    logger.error(f"✗ Не удалось загрузить system_prompt из файла: {str(e)}")
                    system_prompt = """Вы — помощник по философским вопросам. 
Используйте предоставленный контекст для ответа на вопросы пользователя.
Если информация не в контексте, скажите об этом.
Будьте точны и ссылайтесь на источники."""
            user_message = f"""Контекст из философских статей:

{context}

---

Вопрос: {query}

Ответьте на вопрос, используя информацию из контекста выше."""
            # 4. Отправить в OpenAI
            print(f"🔍 DEBUG: Начинаем запрос к OpenAI...")
            print(f"🔍 DEBUG: Model: {settings.OPENAI_CHAT_MODEL}")
            print(f"🔍 DEBUG: Query: {query[:100]}")
            logger.info("4️⃣ Генерирование ответа через OpenAI...")
            try:
                print(f"🔍 DEBUG: Отправляем запрос в OpenAI...")
                logger.debug(f"OpenAI Chat Model: {settings.OPENAI_CHAT_MODEL}")
                logger.debug(f"User message length: {len(user_message)}")
                response = await self.embedder.openai_client.chat.completions.create(
                    model=settings.OPENAI_CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                print(f"🔍 DEBUG: Получен ответ от OpenAI")
                print(f"🔍 DEBUG: Response type: {type(response).__name__}")
                logger.debug(f"OpenAI Response type: {type(response)}")
                logger.debug(f"OpenAI Response: {response}")
                if response is None:
                    print(f"❌ DEBUG: Response is None")
                    logger.error("❌ Response is None")
                    answer = "Ошибка: ответ от OpenAI пустой"
                elif not hasattr(response, 'choices'):
                    print(f"❌ DEBUG: Response has no 'choices' attribute")
                    logger.error(f"❌ Response has no 'choices' attribute: {dir(response)}")
                    answer = f"Ошибка: неожиданный формат ответа"
                elif response.choices is None or len(response.choices) == 0:
                    print(f"❌ DEBUG: No choices in response")
                    logger.error(f"❌ No choices in response: {response.choices}")
                    answer = "Ошибка: нет выбора в ответе OpenAI"
                else:
                    answer = response.choices[0].message.content
                    print(f"✅ DEBUG: Ответ получен, длина: {len(answer)}")
                    logger.info("✓ Ответ получен от OpenAI")
            except Exception as api_error:
                print(f"❌ DEBUG: Ошибка в OpenAI API: {str(api_error)}")
                logger.error(f"❌ Ошибка OpenAI API: {str(api_error)}", exc_info=True)
                raise
            print(f"🔍 DEBUG: Формируем ответ...")
            logger.info("✓ Ответ сгенерирован")
            try:
                print(f"🔍 DEBUG: search_results type: {type(search_results)}")
                print(f"🔍 DEBUG: search_results length: {len(search_results)}")
                if search_results:
                    print(f"🔍 DEBUG: First result keys: {search_results[0].keys() if isinstance(search_results[0], dict) else 'not a dict'}")
                sources = []
                for i, chunk in enumerate(search_results):
                    text = chunk.get("chunk_text") or chunk.get("text") or "N/A"
                    source = {
                        "score": chunk.get("score"),
                        "article_id": chunk.get("article_id"),
                        "filename": chunk.get("filename"),
                        "text": (text[:200] + "...") if text != "N/A" else "N/A"
                    }
                    sources.append(source)
                result = {
                    "query": query,
                    "answer": answer,
                    "sources": sources,
                    "chunks_count": len(search_results),
                    "status": "success"
                }
                print(f"✅ DEBUG: Результат сформирован успешно")
                return result
            except Exception as format_error:
                print(f"❌ DEBUG: Ошибка при формировании результата: {str(format_error)}")
                logger.error(f"❌ Ошибка при формировании результата: {str(format_error)}", exc_info=True)
                raise
        except Exception as e:
            logger.error(f"✗ Ошибка при генерации ответа: {str(e)}")
            raise




# Глобальный инстанс RAG сервиса
_rag_service: Optional[RAGService] = None


async def get_rag_service() -> RAGService:
    """Получить инстанс RAGService (singleton pattern)"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
        await _rag_service.initialize()
    return _rag_service
