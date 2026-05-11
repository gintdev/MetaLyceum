import logging
import os
import re
import json
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Optional
import numpy as np

from app.services.embedders.openai_embedder import OpenAIEmbedder
from app.services.pdf_processor import PDFProcessor
from app.services.text_chunker import TextChunker
from app.services.vector_store import QdrantVectorStore
from app.config import settings

logger = logging.getLogger(__name__)


def _build_requests_logger() -> logging.Logger:
    requests_logger = logging.getLogger("rag_requests")
    if requests_logger.handlers:
        return requests_logger

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(logs_dir, "requests.log"), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    requests_logger.setLevel(logging.INFO)
    requests_logger.addHandler(file_handler)
    requests_logger.propagate = False
    return requests_logger


requests_logger = _build_requests_logger()


class RAGService:
    """
    RAG (Retrieval Augmented Generation) сервис
    Интегрирует: embedder, PDF processor, text chunker, vector store
    """

    async def translate_to_english(self, text: str) -> str:
        try:
            logger.info(f"Перевод запроса на английский: {text[:50]}...")
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
                logger.info(f"Перевод выполнен: {translation[:50]}")
                return translation
            else:
                logger.warning("Не удалось получить перевод через OpenAI")
                return text
        except Exception as e:
            logger.error(f"Ошибка при переводе: {str(e)}")
            return text

    async def hybrid_search(
        self,
        query: str,
        limit: int = settings.SEARCH_LIMIT,
        translation_limit: int = 5,
        final_limit: int = None,
        article_ids: Optional[List[int]] = None,
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
        # 1. Поиск по оригиналу
        orig_results = await self.search(query, limit, article_ids=article_ids)
        # 2. Перевести запрос
        translated_query = await self.translate_to_english(query)
        print(f"🔍 DEBUG: Translated query: {translated_query}")
        # 3. Поиск по переводу (если перевод отличается)
        if translated_query.strip().lower() != query.strip().lower():
            trans_results = await self.search(translated_query, translation_limit, article_ids=article_ids)
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
        self.paraphrase_count = 5
        self.rrf_k = 60
        self.min_candidate_score = 0.5
        self.per_query_pool_size = 60

    async def generate_paraphrases(self, query: str, count: int = 5) -> List[str]:
        """Generate diverse paraphrases in the same language as the original query."""
        system_prompt = (
            "You generate retrieval paraphrases. Return strictly JSON array of strings. "
            "No explanations."
        )
        user_prompt = (
            f"Original query: {query}\n"
            f"Return {count} diverse paraphrases in the same language."
        )
        try:
            response = await self.embedder.openai_client.chat.completions.create(
                model=settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
                max_tokens=300,
            )
            if not response or not response.choices:
                return []
            content = (response.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                return []
            cleaned = [str(item).strip() for item in parsed if str(item).strip()]
            return cleaned[:count]
        except Exception as exc:
            logger.warning("Не удалось сгенерировать перефразировки: %s", exc)
            return []

    @staticmethod
    def _chunk_key(chunk: Dict) -> str:
        if chunk.get("id") is not None:
            return f"id:{chunk.get('id')}"
        return f"meta:{chunk.get('article_id')}:{chunk.get('chunk_index')}:{chunk.get('filename')}"

    def _rrf_fuse_ranked_lists(self, ranked_lists: List[List[Dict]], final_limit: int) -> List[Dict]:
        """Fuse multiple ranked result lists with classic RRF: 1 / (k + rank)."""
        by_key: Dict[str, Dict] = {}

        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list, start=1):
                key = self._chunk_key(item)
                contribution = 1.0 / (self.rrf_k + rank)

                if key not in by_key:
                    by_key[key] = {
                        **item,
                        "rrf_score": 0.0,
                        "rrf_details": [],
                    }

                by_key[key]["rrf_score"] += contribution
                by_key[key]["rrf_details"].append(
                    {
                        "retriever_query": item.get("retriever_query"),
                        "rank": rank,
                        "contribution": contribution,
                    }
                )

        fused = sorted(by_key.values(), key=lambda x: x["rrf_score"], reverse=True)
        for rank, item in enumerate(fused, start=1):
            item["rank"] = rank
        return fused[:final_limit]

    @staticmethod
    def _serialize_chunk_for_log(item: Dict) -> Dict:
        text = item.get("chunk_text") or item.get("text") or ""
        return {
            "rank": item.get("rank"),
            "id": item.get("id"),
            "score": item.get("score"),
            "article_id": item.get("article_id"),
            "filename": item.get("filename"),
            "chunk_index": item.get("chunk_index"),
            "retriever_query": item.get("retriever_query"),
            "rrf_score": item.get("rrf_score"),
            "rrf_details": item.get("rrf_details"),
            "chunk_text": text[:500],
        }

    async def retrieve_with_paraphrase_rrf(
        self,
        query: str,
        limit: int,
        article_ids: Optional[List[int]] = None,
    ) -> tuple[List[Dict], Dict]:
        """Run hybrid search for original query + paraphrases and fuse with RRF."""
        paraphrases = await self.generate_paraphrases(query, count=self.paraphrase_count)
        all_queries = [query, *paraphrases]
        per_query_limit = max(self.per_query_pool_size, limit)

        tasks = [
            self.hybrid_search(
                sub_query,
                limit=per_query_limit,
                translation_limit=per_query_limit,
                final_limit=per_query_limit * 2,
                article_ids=article_ids,
            )
            for sub_query in all_queries
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        ranked_lists: List[List[Dict]] = []
        per_query_trace: Dict[str, List[Dict]] = {}
        for sub_query, result in zip(all_queries, raw_results):
            if isinstance(result, Exception):
                logger.warning("Ошибка поиска по перефразировке '%s': %s", sub_query[:80], result)
                per_query_trace[sub_query] = []
                continue
            filtered_result = [
                item for item in result if float(item.get("score") or 0.0) > self.min_candidate_score
            ]
            ranked = [
                {
                    **item,
                    "rank": rank,
                    "retriever_query": sub_query,
                }
                for rank, item in enumerate(filtered_result, start=1)
            ]
            if ranked:
                ranked_lists.append(ranked)
            per_query_trace[sub_query] = [self._serialize_chunk_for_log(item) for item in ranked]

        if not ranked_lists:
            return [], {
                "original_query": query,
                "paraphrases": paraphrases,
                "all_queries": all_queries,
                "candidate_filter": {
                    "min_score_gt": self.min_candidate_score,
                    "per_query_limit": per_query_limit,
                },
                "per_query_results": per_query_trace,
                "rrf": {
                    "k": self.rrf_k,
                    "top_k": [],
                },
            }

        fused_top = self._rrf_fuse_ranked_lists(ranked_lists, final_limit=limit)
        retrieval_trace = {
            "original_query": query,
            "paraphrases": paraphrases,
            "all_queries": all_queries,
            "candidate_filter": {
                "min_score_gt": self.min_candidate_score,
                "per_query_limit": per_query_limit,
            },
            "per_query_results": per_query_trace,
            "rrf": {
                "k": self.rrf_k,
                "top_k": [self._serialize_chunk_for_log(item) for item in fused_top],
            },
        }
        return fused_top, retrieval_trace
    
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
            logger.info(f"Статья {article_id} переобработана")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Ошибка при переобработке статьи {article_id}: {str(e)}")
            raise
    
    async def search(
        self,
        query: str,
        limit: int = settings.SEARCH_LIMIT,
        article_ids: Optional[List[int]] = None,
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
            if article_ids is not None:
                results = await self.vector_store.search_by_article_ids(query_embedding, article_ids, limit)
            else:
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
    
    # Ответ на вопрос
    async def generate_answer(
        self,
        query: str,
        limit: int = settings.SEARCH_LIMIT,
        query_type: int = 0,
        article_ids: Optional[List[int]] = None,
    ) -> Dict:
        try:
            logger.info(f"RAG запрос (перефразировки + RRF): {query[:50]}...")
            # 1. Один retriever: перефразировки + hybrid-search для каждой + RRF
            search_results, retrieval_trace = await self.retrieve_with_paraphrase_rrf(
                query=query,
                limit=limit,
                article_ids=article_ids,
            )
            if not search_results:
                logger.warning("Релевантные чанки не найдены")
                return {
                    "answer": "К сожалению, в базе знаний не найдено релевантной информации.",
                    "chunks": [],
                    "status": "no_results"
                }
            logger.info(f"Найдено {len(search_results)} чанков после RRF")
            # 2. Собрать контекст из чанков (сократить до 2000 символов)
            context_parts = []
            used_chunks = []
            source_ref_map: dict[tuple[object, object], int] = {}
            total_length = 0
            max_context_length = 2000
            for chunk in search_results:
                text = chunk.get('chunk_text') or chunk.get('text') or ''
                text = text[:300]
                if total_length + len(text) > max_context_length:
                    break

                article_id = chunk.get("article_id")
                filename = chunk.get("filename")
                source_key = (article_id, filename)
                if source_key not in source_ref_map:
                    source_ref_map[source_key] = len(source_ref_map) + 1
                ref_index = source_ref_map[source_key]

                context_parts.append(f"[{ref_index}] {text}")
                used_chunks.append({
                    **chunk,
                    "ref_index": ref_index,
                })
                total_length += len(text)
            context = "\n\n".join(context_parts)

            system_prompt = ""
            try:
                app_dir = os.path.dirname(os.path.dirname(__file__))
                prompt_map = {
                    0: "basic_request_prompt.txt",
                    1: "essay_prompt.txt",
                    2: "literature_recommendation_prompt.txt",
                }
                prompt_filename = prompt_map.get(query_type, "basic_request_prompt.txt")
                prompt_path = os.path.join(app_dir, "prompts", prompt_filename)
                with open(prompt_path, "r", encoding="utf-8") as f:
                    system_prompt = f.read()
                logger.info(f"Загрузили system_prompt из файла {prompt_filename}")
            except Exception as e:
                logger.error(f"Не удалось загрузить system_prompt из файла: {str(e)}")
                system_prompt = ""

            user_message = f"""
            Контекст из философских статей: {context}
                ---
            Вопрос: {query}"""

            request_trace = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "top_k": len(search_results),
                "retrieval": retrieval_trace,
                "llm_input": {
                    "model": settings.OPENAI_CHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                },
            }
            requests_logger.info(json.dumps(request_trace, ensure_ascii=False))
            
            print(f"DEBUG: Начинаем запрос к OpenAI...")
            print(f"DEBUG: Model: {settings.OPENAI_CHAT_MODEL}")
            print(f"DEBUG: Query: {query[:100]}")
            logger.info("Генерирование ответа через OpenAI...")
            try:
                print(f"DEBUG: Отправляем запрос в OpenAI...")
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
                print(f"DEBUG: Получен ответ от OpenAI")
                print(f"DEBUG: Response type: {type(response).__name__}")
                logger.debug(f"OpenAI Response type: {type(response)}")
                logger.debug(f"OpenAI Response: {response}")
                if response is None:
                    print(f"DEBUG: Response is None")
                    logger.error("Response is None")
                    answer = "Ошибка: ответ от OpenAI пустой"
                elif not hasattr(response, 'choices'):
                    print(f"DEBUG: Response has no 'choices' attribute")
                    logger.error(f"Response has no 'choices' attribute: {dir(response)}")
                    answer = f"Ошибка: неожиданный формат ответа"
                elif response.choices is None or len(response.choices) == 0:
                    print(f"DEBUG: No choices in response")
                    logger.error(f"No choices in response: {response.choices}")
                    answer = "Ошибка: нет выбора в ответе OpenAI"
                else:
                    answer = response.choices[0].message.content
                    # Normalize citation style if model outputs legacy pattern like [Источник 3].
                    answer = re.sub(r"\[(?:Источник|источник|source)\s+(\d+)\]", r"[\1]", answer)
                    print(f"DEBUG: Ответ получен, длина: {len(answer)}")
                    logger.info("✓ Ответ получен от OpenAI")
            except Exception as api_error:
                print(f"DEBUG: Ошибка в OpenAI API: {str(api_error)}")
                logger.error(f"Ошибка OpenAI API: {str(api_error)}", exc_info=True)
                raise
            print(f"🔍 DEBUG: Формируем ответ...")
            logger.info("✓ Ответ сгенерирован")
            try:
                print(f"🔍 DEBUG: search_results type: {type(search_results)}")
                print(f"🔍 DEBUG: search_results length: {len(search_results)}")
                if search_results:
                    print(f"🔍 DEBUG: First result keys: {search_results[0].keys() if isinstance(search_results[0], dict) else 'not a dict'}")
                sources = []
                pdf_files = []
                seen_file_keys = set()
                for chunk in used_chunks:
                    text = chunk.get("chunk_text") or chunk.get("text") or "N/A"
                    filename = chunk.get("filename")
                    article_id = chunk.get("article_id")
                    ref_index = chunk.get("ref_index")
                    source = {
                        "score": chunk.get("score"),
                        "article_id": article_id,
                        "filename": filename,
                        "ref_index": ref_index,
                        "text": (text[:200] + "...") if text != "N/A" else "N/A"
                    }
                    sources.append(source)

                    file_key = (article_id, filename)
                    if filename and file_key not in seen_file_keys:
                        seen_file_keys.add(file_key)
                        pdf_files.append({
                            "filename": filename,
                            "article_id": article_id,
                            "ref_index": ref_index,
                        })
                result = {
                    "query": query,
                    "answer": answer,
                    "sources": sources,
                    "pdf_files": pdf_files,
                    "chunks_count": len(used_chunks),
                    "status": "success"
                }
                print(f"DEBUG: Результат сформирован успешно")
                return result
            except Exception as format_error:
                print(f"DEBUG: Ошибка при формировании результата: {str(format_error)}")
                logger.error(f"Ошибка при формировании результата: {str(format_error)}", exc_info=True)
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
