import psycopg
from pgvector.psycopg import register_vector
import time

from comum import CONEXAO, embed



def buscar_avaliacoes(conn, pergunta, k=3, nota_max=None):
    q = embed([pergunta])[0]
    params = {"q": q, "k": k}
    where = ""

    if nota_max is not None:
        where = "WHERE nota <= %(nota_max)s"
        params["nota_max"] = nota_max

    return conn.execute(
        f"""SELECT 1 - (embedding <=> %(q)s) AS similaridade, nota, comment
            FROM avaliacoes
            {where}
            ORDER BY embedding <=> %(q)s
            LIMIT %(k)s""",
        params,
    ).fetchall()
    

if __name__ == "__main__":
    inicio = time.time()
    perguntas = [
        "a mercadoria veio danificada",
        "demorou uma eternidade pra chegar",
        "o vendedor nunca respondeu minhas mensagens",
    ]

    with psycopg.connect(CONEXAO) as conn:
        conn.execute("SET enable_seqscan = off")
        register_vector(conn)
        for p in perguntas:
            print(f"\n> {p}")
            for similaridade, nota, texto in buscar_avaliacoes(conn, p): #buscar_avaliacoes(conn, "problema com a entrega", nota_max=2)  # só notas 1 e 2
                print(f"  {similaridade:.2f}  nota {nota}  {texto[:80]}")
    
    fim = time.time()
    print(f"\nTempo total: {fim - inicio:.2f} segundos")
    # q = embed(["a mercadoria veio danificada"])[0]
    # with psycopg.connect(CONEXAO) as conn:
    #     register_vector(conn)
    #     conn.execute("SET enable_seqscan = off")
    #     print(conn.execute("SHOW enable_seqscan").fetchone())
    #     plano = conn.execute(
    #         """EXPLAIN ANALYZE
    #             SELECT 1 - (embedding <=> %(q)s) AS similaridade, nota, comment
    #             FROM avaliacoes
    #             ORDER BY embedding <=> %(q)s
    #             LIMIT 3""",
    #         {"q": q},
    #     ).fetchall()

    # for (linha,) in plano:
    #     print(linha)