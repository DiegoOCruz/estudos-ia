CREATE TABLE IF NOT EXISTS documentos (
    id            SERIAL PRIMARY KEY,
    tipo          TEXT NOT NULL,
    sku           TEXT,
    codigo_barras TEXT,
    personagem    TEXT,
    preco         NUMERIC(10, 2),
    conteudo      TEXT NOT NULL,
    embedding     vector(1024) NOT NULL
);
