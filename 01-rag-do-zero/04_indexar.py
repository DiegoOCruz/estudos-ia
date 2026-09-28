from decimal import Decimal

import ollama
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

MODELO = "bge-m3"
CONEXAO = "postgresql://rag:rag@localhost:5432/rag"

def embed(textos):
    resp = ollama.embed(model=MODELO, input=textos)
    v = np.array(resp["embeddings"])
    return v / np.linalg.norm(v, axis=1, keepdims=True)

catalogo = [
    {"tipo": "produto", "sku": "FUN-001", "personagem": "Homem-Aranha", "preco": Decimal("149.90"),
     "conteudo": "Funko Pop Marvel Homem-Aranha, figura de vinil de 10 cm do Spider-Man em pose clássica, caixa com janela."},
    {"tipo": "produto", "sku": "FUN-002", "personagem": "Batman", "preco": Decimal("139.90"),
     "conteudo": "Funko Pop DC Batman, figura de vinil de 10 cm do Cavaleiro das Trevas com capa preta."},
    {"tipo": "produto", "sku": "FUN-003", "personagem": "Venom", "preco": Decimal("159.90"),
     "conteudo": "Funko Pop Marvel Venom, o vilão simbionte inimigo do Homem-Aranha, figura de vinil de 10 cm."},
    {"tipo": "produto", "sku": "ACT-001", "personagem": "Homem-Aranha", "preco": Decimal("299.90"),
     "conteudo": "Action figure articulada do Spider-Man, 30 cm, com 20 pontos de articulação e acessórios."},
    {"tipo": "servico",
     "conteudo": "Impressão 3D sob encomenda: peças em PLA ou PETG a partir do seu arquivo STL. Prazo médio de 3 a 5 dias úteis."},
    {"tipo": "produto", "sku": "FIL-001", "preco": Decimal("119.90"),
     "conteudo": "Filamento PLA preto 1 kg, diâmetro 1,75 mm, para impressoras 3D."},
    {"tipo": "politica",
     "conteudo": "Política de trocas: produtos lacrados podem ser trocados em até 7 dias após o recebimento."},
    {"tipo": "politica",
     "conteudo": "Frete grátis para compras acima de R$ 200 para todo o Brasil."},
]

with psycopg.connect(CONEXAO) as conn:
    register_vector(conn)
    conn.execute("TRUNCATE documentos RESTART IDENTITY")

    vetores = embed([d["conteudo"] for d in catalogo])

    for doc, vetor in zip(catalogo, vetores):
        conn.execute(
            """INSERT INTO documentos (tipo, sku, personagem, preco, conteudo, embedding)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (doc["tipo"], doc.get("sku"), doc.get("personagem"),
             doc.get("preco"), doc["conteudo"], vetor),
        )

print(f"{len(catalogo)} documentos indexados.")