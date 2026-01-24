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