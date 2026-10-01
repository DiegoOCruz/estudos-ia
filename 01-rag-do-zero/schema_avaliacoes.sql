CREATE TABLE
    IF NOT EXISTS avaliacoes (
        id SERIAL PRIMARY KEY,
        score SMALLINT CHECK (
            score >= 1
            AND score <= 5
        ) NOT NULL,
        review_id TEXT NOT NULL UNIQUE,
        comment TEXT NOT NULL,
        embedding vector (1024) NOT NULL
    );