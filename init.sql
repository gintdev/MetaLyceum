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

-- Реестр ссылок
CREATE TABLE IF NOT EXISTS article_registry (
    id              SERIAL PRIMARY KEY,
    article_url     TEXT,
    source          VARCHAR(10), -- JSTOR, Киберленинка, PhilPapers
    download_status VARCHAR(10)
)
-- Индексы для оптимизации
CREATE INDEX IF NOT EXISTS idx_articles_title ON articles USING GIN(to_tsvector('russian', title));
CREATE INDEX IF NOT EXISTS idx_articles_keywords ON articles USING GIN(keywords);