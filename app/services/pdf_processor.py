import os
import logging
import asyncio
import aiofiles
from typing import Optional, List
import yadisk
import PyPDF2
from app.config import settings

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Сервис для загрузки и обработки PDF файлов"""
    
    def __init__(self, yadisk_token: Optional[str] = None):
        """
        Инициализация PDF processor
        
        Args:
            yadisk_token: Токен для Яндекс Диска
        """
        self.yadisk_token = yadisk_token or settings.YADISK_TOKEN
        self.temp_dir = settings.PDF_TEMP_DIR
        self._ensure_temp_dir()
        
    def _ensure_temp_dir(self):
        """Создать директорию для временных файлов если её нет"""
        os.makedirs(self.temp_dir, exist_ok=True)
    
    async def download_pdf_from_yadisk(self, filename: str) -> str:
        """
        Скачать PDF с Яндекс Диска
        
        Args:
            filename: Имя файла на диске (может включать папки: "MetaLyceum/Cyberleninka/file.pdf")
            
        Returns:
            Путь к скачанному файлу
            
        Raises:
            Exception: Если скачивание не удалось
        """
        if not self.yadisk_token:
            raise ValueError("YADISK_TOKEN не установлен в конфигурации")
        
        try:
            # Инициализировать клиент Яндекс Диска
            y = yadisk.Client(token=self.yadisk_token)
            
            # Определить путь на Яндекс Диске
            # Если filename уже начинается с YADISK_FOLDER, использовать как есть
            # Иначе добавить YADISK_FOLDER в начало
            if filename.startswith(settings.YADISK_FOLDER):
                yadisk_path = f"/{filename}"
            else:
                yadisk_path = f"/{settings.YADISK_FOLDER}/{filename}"
            
            # Локальный путь для сохранения (извлечь только имя файла)
            local_filename = os.path.basename(filename)
            local_path = os.path.join(self.temp_dir, local_filename)
            
            # Создать папку если нужна
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            logger.info(f"📥 Скачивание: {yadisk_path}")
            logger.info(f"   → {local_path}")
            
            # Скачивание файла (асинхронный запрос в отдельном потоке)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: y.download(yadisk_path, local_path)
            )
            
            logger.info(f"✓ PDF скачан успешно")
            return local_path
            
        except Exception as e:
            logger.error(f"✗ Ошибка при скачивании PDF {filename}: {str(e)}")
            raise
    
    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Извлечь текст из PDF файла
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Извлеченный текст
            
        Raises:
            Exception: Если извлечение текста не удалось
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _extract():
                text = []
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text.append(page.extract_text())
                return "\n".join(text)
            
            text = await loop.run_in_executor(None, _extract)
            logger.info(f"✓ Текст извлечен из PDF: {len(text)} символов")
            return text
            
        except Exception as e:
            logger.error(f"✗ Ошибка при извлечении текста из PDF {pdf_path}: {str(e)}")
            raise
    
    async def process_pdf(self, filename: str) -> str:
        """
        Полная обработка PDF: скачивание + извлечение текста
        
        Args:
            filename: Имя файла на Яндекс Диске
            
        Returns:
            Извлеченный текст из PDF
        """
        try:
            # Скачать PDF
            pdf_path = await self.download_pdf_from_yadisk(filename)
            
            # Извлечь текст
            text = await self.extract_text_from_pdf(pdf_path)
            
            return text
            
        except Exception as e:
            logger.error(f"✗ Ошибка при обработке PDF {filename}: {str(e)}")
            raise
    
    def cleanup_temp_file(self, pdf_path: str):
        """
        Удалить временный файл
        
        Args:
            pdf_path: Путь к файлу для удаления
        """
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                logger.info(f"✓ Временный файл удален: {pdf_path}")
        except Exception as e:
            logger.warning(f"⚠ Ошибка при удалении файла {pdf_path}: {str(e)}")


# Глобальный инстанс processor
_pdf_processor: Optional[PDFProcessor] = None


def get_pdf_processor() -> PDFProcessor:
    """Получить инстанс PDF processor (singleton pattern)"""
    global _pdf_processor
    if _pdf_processor is None:
        _pdf_processor = PDFProcessor()
    return _pdf_processor
