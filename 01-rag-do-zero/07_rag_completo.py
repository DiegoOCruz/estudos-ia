from decimal import Decimal

import ollama
import psycopg
from pgvector.psycopg import register_vector

from comum import LLM, CONEXAO, buscar, extrair_filtros

def para_kwargs(filtros):
    kwargs = {}
    if filtros.preco_max is not None:
        kwargs["preco_max"] = Decimal(str(filtros.preco_max))
    if filtros.tipo:
        kwargs["tipo"] = filtros.tipo
    return kwargs

INSTRUCOES_RESPOSTA = (
    "Você é o atendente de uma loja de Funkos e impressão 3D. "
    "Responda usando APENAS as informações fornecidas. "
    "Se elas não responderem à pergunta, diga que não temos essa informação ou esse produto. "
    "Nunca invente produtos, preços ou prazos."
)

def responder(conn, pergunta, k=3):
    filtros = extrair_filtros(pergunta)
    resultados = buscar(conn, pergunta, k, **para_kwargs(filtros))

    contexto = "\n".join(
        f"- {conteudo} (preço: R$ {preco})" if preco else f"- {conteudo}"
        for _, conteudo, preco in resultados
    )

    resp = ollama.chat(model=LLM, messages=[
        {"role": "system", "content": INSTRUCOES_RESPOSTA},
        {"role": "user", "content": f"Informações:\n{contexto}\n\nPergunta: {pergunta}"},
    ])
    return filtros, resp["message"]["content"]

perguntas = [
    "Vocês têm Funko do Homem-Aranha até 150 reais?",
    "Quanto custa o Funko do Batman?",
    "Posso trocar um produto?",
    "Vocês vendem PlayStation 5?",
]

with psycopg.connect(CONEXAO) as conn:
    register_vector(conn)
    for p in perguntas:
        filtros, resposta = responder(conn, p)
        print(f"\n> {p}")
        print(f"  filtros: {filtros}")
        print(f"  {resposta}")