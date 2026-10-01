import pandas as pd
import psycopg
from pgvector.psycopg import register_vector

from comum import CONEXAO, embed
from dados_olist import limpar_avaliacoes
import time

if __name__ == "__main__":
    df = pd.read_csv("dados/olist_order_reviews_dataset.csv")
    df = limpar_avaliacoes(df)
    comeco = time.time()
    textos = df["texto_limpo"].tolist()
    vetores = embed(textos)

    with psycopg.connect(CONEXAO) as conn:
        register_vector(conn)
        conn.execute("TRUNCATE avaliacoes RESTART IDENTITY")

        for nota, review_id, texto, vetor in zip(
            df["review_score"], df["review_id"], df["texto_limpo"], vetores
        ):
            conn.execute(
                "INSERT INTO avaliacoes (score, review_id, comment, embedding) VALUES (%s, %s, %s, %s)",
                (int(nota), review_id, texto, vetor),
            )

    fim = time.time()
    print("Gravadas:", len(df))
    print(f"Tempo total: {fim - comeco:.2f} segundos")