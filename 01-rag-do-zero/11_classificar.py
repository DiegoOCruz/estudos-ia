from typing import Literal
from comum import LLM
from pydantic import BaseModel, Field
import ollama

Problema = Literal[
    "atraso_ou_nao_recebido",
    "problema_na_entrega",
    "pedido_incompleto",
    "produto_errado_ou_diferente",
    "defeito_ou_qualidade",
    "atendimento_ou_financeiro",
    "outro",
]

class Classificacao(BaseModel):
    problemas: list[Problema] = Field(
        description="Problemas citados no comentário. Lista vazia se nenhum problema for identificado."
    )
    sentimento: Literal["positivo", "negativo", "neutro", "misto"]
    
PROMPT_SISTEMA = """Você classifica avaliações de clientes de uma loja online.

Problemas possíveis:
- atraso_ou_nao_recebido: o produto atrasou ou ainda não chegou
- problema_na_entrega: entregue no endereço errado, ou o cliente teve que retirar no correio
- pedido_incompleto: faltou algum item do pedido
- produto_errado_ou_diferente: veio outro produto, falsificado ou diferente do anúncio
- defeito_ou_qualidade: produto com defeito, danificado ou de má qualidade
- atendimento_ou_financeiro: falta de resposta, nota fiscal, cobrança, reembolso
- outro: problema que não se encaixa em nenhuma categoria acima

Sentimentos:
- positivo: o cliente só elogia
- negativo: o cliente está insatisfeito, mesmo sem dizer o motivo
- misto: elogia uma coisa e reclama de outra
- neutro: não dá para saber se está satisfeito

Classifique apenas pelo que está escrito. Não deduza problemas que o cliente não citou."""

EXEMPLOS = [
    (
        "Não chegou na minha casa, precisei ir na agência dos Correios pegar o pacote.",
        Classificacao(problemas=["problema_na_entrega"], sentimento="negativo"),
    ),
    (
        "O pedido ainda está a caminho, dentro do prazo. Depois eu avalio.",
        Classificacao(problemas=[], sentimento="neutro"),
    ),
    (
        "O prazo de entrega é muito longo, mas chegou na data prometida.",
        Classificacao(problemas=["outro"], sentimento="negativo"),
    ),
    (
        "Já passou uma semana da data prevista e nada do produto.",
        Classificacao(problemas=["atraso_ou_nao_recebido"], sentimento="negativo"),
    ),
]

def montar_mensagens(texto: str) -> list[dict]:
    mensagens = [{"role": "system", "content": PROMPT_SISTEMA}]

    for exemplo_texto, exemplo_resposta in EXEMPLOS:
        mensagens.append({"role": "user", "content": exemplo_texto})
        mensagens.append({"role": "assistant", "content": exemplo_resposta.model_dump_json()})

    mensagens.append({"role": "user", "content": texto})
    return mensagens

def classificar(texto: str) -> Classificacao:
    resp = ollama.chat(
        model=LLM,
        messages=montar_mensagens(texto),
        format=Classificacao.model_json_schema(),
        options={"temperature": 0},
    )
    return Classificacao.model_validate_json(resp["message"]["content"])

if __name__ == "__main__":
    if __name__ == "__main__":
        exemplos = [
        "Entrega super rápida , os produtos são bons porém vem todos iguais achei disnecessario vou ficar com muitos pincéis sem usar pelo fato de serem iguais.",
        "PRODUTO ENTREGUE NO PRAZO COM QUALIDADE",
        "Não gostei",
        "Ainda não posso avaliar nem o estado do produto nem a loja,Quando chegar farei uma avaliação justa .",
        "Não recebi, tive que buscar no correio, por alegação de tentativa de entrega sem ninguém em casa, ma é mentira na data alegada tinha gente em casa sim.",
        "boa noite, havia feito 3 encomendas, 3 Cortinas Voil Cinza 3,00x2,80 Forro Microfibra Ilhós Cromado e 1 Cortina Microfibra Cinza 3,00x2,80 Com Ilhós Cromados, e veio apenas de microfibra.",
        "Muuuuito demorado, apesar de entregue no prazo!",
    ]

    for texto in exemplos:
        c = classificar(texto)
        print(f"{c.sentimento:8} {c.problemas}")
        print(f"         {texto[:80]}\n")