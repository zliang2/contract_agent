-- Clausewitz schema (docs/SPEC.md §4). Idempotent: safe to apply repeatedly.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS contracts (
  id          SERIAL PRIMARY KEY,
  title       TEXT NOT NULL,
  split       TEXT NOT NULL CHECK (split IN ('dev', 'test')),
  full_text   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  id            SERIAL PRIMARY KEY,
  contract_id   INT NOT NULL REFERENCES contracts(id),
  section_path  TEXT,                 -- e.g. "12.3 Termination"
  start_char    INT NOT NULL,
  end_char      INT NOT NULL,
  text          TEXT NOT NULL,
  embedding     vector(384) NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_contract_id ON chunks (contract_id);

CREATE TABLE IF NOT EXISTS annotations (            -- CUAD gold spans
  id           SERIAL PRIMARY KEY,
  contract_id  INT NOT NULL REFERENCES contracts(id),
  clause_type  TEXT NOT NULL,
  start_char   INT NOT NULL,
  end_char     INT NOT NULL
);
