import logging
from typing import List, Tuple
from app.config import settings

logger = logging.getLogger(__name__)


class TextChunker:
    """Сервис для разбиения текста на чанки"""
    
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        """
        Инициализация TextChunker
        
        Args:
            chunk_size: Размер чанка в символах
            chunk_overlap: Перекрытие между чанками в символах
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def _split_text_by_sentences(self, text: str) -> List[str]:
        """
        Разбить текст на предложения
        
        Args:
            text: Исходный текст
            
        Returns:
            Список предложений
        """
        # Простое разбиение по знакам препинания
        # Для более сложного парсинга можно использовать NLTK
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            
            if char in '.!?':
                # Пропустить если это аббревиатура
                if len(current_sentence) > 3:
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # Добавить остаток
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Разбить текст на чанки с перекрытием
        
        Args:
            text: Исходный текст
            
        Returns:
            Список чанков
        """
        try:
            # Очистить текст от лишних пробелов и переносов
            text = " ".join(text.split())
            
            if len(text) <= self.chunk_size:
                logger.info(f"Текст меньше размера чанка ({len(text)} символов)")
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(text):
                # Определить конец чанка
                end = start + self.chunk_size
                
                # Если это не последний чанк, найти ближайший пробел
                if end < len(text):
                    # Найти последний пробел перед концом
                    last_space = text.rfind(" ", start, end)
                    if last_space > start:
                        end = last_space
                
                # Добавить чанк
                chunk = text[start:end].strip()
                if chunk:  # Не добавлять пустые чанки
                    chunks.append(chunk)
                
                # Переместиться на следующий чанк
                start = end - self.chunk_overlap
            
            logger.info(f"✓ Текст разбит на {len(chunks)} чанков")
            return chunks
            
        except Exception as e:
            logger.error(f"✗ Ошибка при разбиении текста: {str(e)}")
            raise
    
    def chunk_text_with_metadata(
        self,
        text: str,
        article_id: int,
        filename: str
    ) -> List[dict]:
        """
        Разбить текст на чанки с метаданными
        
        Args:
            text: Исходный текст
            article_id: ID статьи
            filename: Имя файла
            
        Returns:
            Список словарей с чанками и метаданными
        """
        chunks = self.chunk_text(text)
        
        chunked_data = []
        for idx, chunk in enumerate(chunks):
            chunked_data.append({
                "article_id": article_id,
                "filename": filename,
                "chunk_index": idx,
                "chunk_text": chunk,
                "chunk_size": len(chunk)
            })
        
        return chunked_data


# Глобальный инстанс chunker
_text_chunker = None


def get_text_chunker() -> TextChunker:
    """Получить инстанс TextChunker (singleton pattern)"""
    global _text_chunker
    if _text_chunker is None:
        _text_chunker = TextChunker()
    return _text_chunker
