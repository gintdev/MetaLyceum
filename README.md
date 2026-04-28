# MetaLyceum

«Среда интерактивного обучения философии на основе RAG» — это веб-приложение, предназначенное для упрощения и оптимизации процесса обучения философии.

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

Откройте `http://localhost:5173` — там доступен чат, который отправляет сообщения в FastAPI endpoint `POST /articles/ask`.

## Авторизация

- В проекте включена авторизация только через Google Identity Services.
- Фронтенд получает `credential` (Google ID token) и отправляет его в `POST /auth/google`.
- Бэкенд валидирует токен у Google, создает или обновляет пользователя в таблице `users`, затем выдает собственный JWT для работы с API (`/chats/*`, `/auth/me`).
