# MetaLyceum

«Среда интерактивного обучения философии на основе Retrieval-Augmented Generation».

## Сравнительный анализ ответов (исходный вариант)
[Google Таблицы](https://docs.google.com/spreadsheets/d/19BmGm9xRtjaRuERp0eTRWx6xFLF7UMoTq0b42uBsutQ/edit?usp=sharing)

## Запуск через Docker Compose

Запускает одновременно:

- PostgreSQL
- Qdrant
- FastAPI backend (`http://localhost:8000`)
- React + TypeScript frontend (`http://localhost:5173`)

### 1) Подготовить `.env`

Убедитесь, что в `.env` есть минимум:

```env
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=metalyceum
OPENAI_API_KEY=your_openai_key
GOOGLE_CLIENT_ID=your_google_oauth_web_client_id
VITE_GOOGLE_CLIENT_ID=your_google_oauth_web_client_id
```

### 2) Запуск

```bash
docker compose up --build
```

### 3) Интерфейс

Откройте `http://localhost:5173` — веб-интерфейс. После авторизации доступен весь функионал приложения.

## Авторизация

- Реализована авторизация только через Google Identity Services.
- Фронтенд получает `credential` (Google ID token) и отправляет его в `POST /auth/google`.
- Бэкенд валидирует токен у Google, создает или обновляет пользователя в таблице `users`, затем выдает собственный JWT для работы с API.
