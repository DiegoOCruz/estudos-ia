import ollama
import numpy as np
from typing import Literal, Optional
from pydantic import BaseModel

MODELO = "bge-m3"
LLM = "qwen2.5:14b"
CONEXAO = "postgresql://rag:rag@localhost:5432/rag"

def embed(textos):
    resp = ollama.embed(model=MODELO, input=textos)
    v = np.array(resp["embeddings"])
    return v / np.linalg.norm(v, axis=1, keepdims=True)

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
    
class Filtros(BaseModel):
    preco_max: Optional[float] = None
    tipo: Optional[Literal["produto", "servico", "politica"]] = None
    
INSTRUCOES_FILTROS = (
    "Você extrai filtros de busca da pergunta de um cliente de uma loja de Funkos e impressão 3D. "
    "preco_max: preencha só se o cliente mencionar um valor máximo. "
    "tipo: 'politica' para trocas, devoluções e frete; 'servico' para impressão sob encomenda; "
    "'produto' para itens à venda. Use null quando não estiver claro. "
    "Nunca invente filtros que o cliente não pediu."
)

def extrair_filtros(pergunta):
    resp = ollama.chat(
        model=LLM,
        messages=[
            {"role": "system", "content": INSTRUCOES_FILTROS},
            {"role": "user", "content": pergunta},
        ],
        format=Filtros.model_json_schema(),
        options={"temperature": 0},
    )
    return Filtros.model_validate_json(resp["message"]["content"])