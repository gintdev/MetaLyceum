FROM python:3.11-slim

WORKDIR /app

# Установить системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Копировать файл требований
COPY requirements.txt .

# Установить зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копировать приложение
COPY . .

# Exposer порт
EXPOSE 8000

# Команда для запуска
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
