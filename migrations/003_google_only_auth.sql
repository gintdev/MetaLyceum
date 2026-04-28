-- Convert existing password-based auth schema to Google-only auth.
ALTER TABLE users ADD COLUMN IF NOT EXISTS google_sub VARCHAR(255);

-- Backfill deterministic placeholder for existing accounts before setting NOT NULL/UNIQUE.
UPDATE users
SET google_sub = CONCAT('legacy-user-', id)
WHERE google_sub IS NULL;

ALTER TABLE users ALTER COLUMN google_sub SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname = 'users_google_sub_key'
    ) THEN
        ALTER TABLE users ADD CONSTRAINT users_google_sub_key UNIQUE (google_sub);
    END IF;
END $$;

ALTER TABLE users DROP COLUMN IF EXISTS password_hash;
