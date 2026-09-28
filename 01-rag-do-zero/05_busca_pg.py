import ollama
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from decimal import Decimal

MODELO = "bge-m3"
CONEXAO = "postgresql://rag:rag@localhost:5432/rag"

def embed(textos):
    resp = ollama.embed(model=MODELO, input=textos)
    v = np.array(resp["embeddings"])
    return v / np.linalg.norm(v, axis=1, keepdims=True)

# def buscar(conn, pergunta, k=3):
#     q = embed([pergunta])[0]
#     return conn.execute(
#         """SELECT 1 - (embedding <=> %(q)s) AS score, conteudo
#            FROM documentos
#            ORDER BY embedding <=> %(q)s
#            LIMIT %(k)s""",
#         {"q": q, "k": k},
#     ).fetchall()
def buscar(conn, pergunta, k=3, personagem=None, tipo=None, preco_max=None):
    q = embed([pergunta])[0]
    params = {"q": q, "k": k}
    filtros = []

    if personagem:
        filtros.append("personagem = %(personagem)s")
        params["personagem"] = personagem
    if tipo:
        filtros.append("tipo = %(tipo)s")
        params["tipo"] = tipo
    if preco_max is not None:
        filtros.append("preco <= %(preco_max)s")
        params["preco_max"] = preco_max

    where = "WHERE " + " AND ".join(filtros) if filtros else ""

    return conn.execute(
        f"""SELECT 1 - (embedding <=> %(q)s) AS score, conteudo, preco
            FROM documentos
            {where}
            ORDER BY embedding <=> %(q)s
            LIMIT %(k)s""",
        params,
    ).fetchall()
    
# perguntas = [
#     "Vocês têm Funko do Homem-Aranha?",
#     "Quanto tempo leva pra imprimir uma peça minha?",
#     "Posso devolver um boneco que comprei?",
#     "Queria um presente de super-herói pra uma criança",
#     "Vocês vendem PlayStation 5?",
# ]

# with psycopg.connect(CONEXAO) as conn:
#     register_vector(conn)
#     for p in perguntas:
#         print(f"\n> {p}")
#         for score, conteudo in buscar(conn, p):
#             print(f"  {score:.2f}  {conteudo[:70]}")



testes = [
    ("Vocês têm Funko do Homem-Aranha?", {}),
    ("Vocês têm Funko do Homem-Aranha?", {"personagem": "Homem-Aranha"}),
    ("Posso devolver um boneco que comprei?", {"tipo": "politica"}),
    ("Queria um presente de super-herói pra uma criança", {"preco_max": Decimal("150")}),
]

with psycopg.connect(CONEXAO) as conn:
    register_vector(conn)
    for pergunta, filtros in testes:
        print(f"\n> {pergunta}  {filtros}")
        for score, conteudo, preco in buscar(conn, pergunta, **filtros):
            print(f"  {score:.2f}  {preco or '':>7}  {conteudo[:60]}")