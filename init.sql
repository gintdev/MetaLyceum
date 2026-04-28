-- Статья
CREATE TABLE IF NOT EXISTS articles (
    id              SERIAL PRIMARY KEY,
    title           TEXT        NOT NULL,      -- Название статьи
    source          VARCHAR(20),               -- JSTOR, Киберленинка, PhilPapers
    abstract        TEXT,                      -- Аннотация
    authors         TEXT[],                    -- Авторы
    keywords        TEXT[],                    -- Ключевые слова
    published_year  VARCHAR(10),               -- Год публикации
    views_count     INT,                       -- Количество просмотров
    downloads_count INT,                       -- Количество скачиваний
    file_name       TEXT,                      -- Имя файла на диске
    download_url    TEXT,                      -- Ссылка на скачивание
    parsed_at       TIMESTAMPTZ DEFAULT now()  -- Дата парсинга
);

CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    google_sub      VARCHAR(255) NOT NULL UNIQUE,
    email           VARCHAR(320) NOT NULL UNIQUE,
    display_name    VARCHAR(120),
    avatar_url      VARCHAR(2048),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chats (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title           VARCHAR(255) NOT NULL DEFAULT 'New chat',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    chat_id         INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    sources         JSONB NOT NULL DEFAULT '[]'::jsonb,
    query_type      INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_messages_role CHECK (role IN ('user', 'assistant', 'system'))
);

CREATE INDEX IF NOT EXISTS idx_chats_user_updated_at ON chats(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_chat_created_at ON messages(chat_id, created_at);