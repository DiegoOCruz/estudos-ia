from typing import Literal

# from comum import LLM
from pydantic import BaseModel, Field
import ollama

Problema = Literal[
    "atraso_ou_nao_recebido",
    "problema_na_entrega",
    "pedido_incompleto",
    "produto_errado_ou_diferente",
    "defeito_ou_qualidade",
    "atendimento_ou_financeiro",
    "frete_caro",
    "prazo_longo",
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
- atendimento_ou_financeiro: o cliente reclama do atendimento ou da parte financeira: ficou sem resposta, foi mal atendido, nota fiscal, cobrança indevida, reembolso. Pedir uma troca ou dizer que abriu uma reclamação não é, por si só, problema de atendimento.
- frete_caro: o cliente acha o valor do frete alto. Uma cobrança errada de frete é atendimento_ou_financeiro.
- prazo_longo: o produto chegou dentro do prazo, mas o cliente acha o prazo de entrega longo ou queria uma entrega mais rápida
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
        Classificacao(problemas=["prazo_longo"], sentimento="negativo"),
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
        mensagens.append(
            {"role": "assistant", "content": exemplo_resposta.model_dump_json()}
        )

    mensagens.append({"role": "user", "content": texto})
    return mensagens


def classificar(texto: str, LLM: str) -> Classificacao:
    resp = ollama.chat(
        model=LLM,
        messages=montar_mensagens(texto),
        format=Classificacao.model_json_schema(),
        options={"temperature": 0},
        think=False,
    )
    return Classificacao.model_validate_json(resp["message"]["content"])


if __name__ == "__main__":
    LLMs = [
        "qwen3.8:27b",
        "qwen2.5:14b",
    ]
    for modelo in LLMs:
        print("LLM:", modelo)
        casos = [
            (
                "Entrega super rápida , os produtos são bons porém vem todos iguais achei disnecessario vou ficar com muitos pincéis sem usar pelo fato de serem iguais.",
                ["produto_errado_ou_diferente"],
                "misto",
            ),
            ("PRODUTO ENTREGUE NO PRAZO COM QUALIDADE", [], "positivo"),
            ("Não gostei", [], "negativo"),
            (
                "Ainda não posso avaliar nem o estado do produto nem a loja,Quando chegar farei uma avaliação justa .",
                [],
                "neutro",
            ),
            (
                "Não recebi, tive que buscar no correio, por alegação de tentativa de entrega sem ninguém em casa, ma é mentira na data alegada tinha gente em casa sim.",
                ["problema_na_entrega"],
                "negativo",
            ),
            (
                "NAO TIVE RETORNO E NAO RECEBI A MERCADORIA COM MAIS DE 20 DIAS DE PEDIDO APROVADO",
                ["atraso_ou_nao_recebido", "atendimento_ou_financeiro"],
                "negativo",
            ),
            ("Produto ce qualidade", [], "positivo"),
            (
                "Muuuuito demorado, apesar de entregue no prazo!",
                ["prazo_longo"],
                "negativo",
            ),
            (
                "Recebi o carregador, mas ele veio com defeito. Não está carregando o celular. Como realizo a troca?",
                ["defeito_ou_qualidade"],
                "negativo",
            ),
            (
                "Comprei um produto, recebi outro completamente diferente, ja abri uma reclamação, estou aguardando retorno.",
                ["produto_errado_ou_diferente"],
                "negativo",
            ),
            ("So achei o preco do frete muito alto!", ["frete_caro"], "negativo"),
            ("Adorei o produto, porém o frete muito caro.", ["frete_caro"], "misto"),
            (
                "Cobraram 2 fretes para mesmo produto pois tinha comprado 2 unidades achei isso muita sacanagem !!!",
                ["atendimento_ou_financeiro"],
                "negativo",
            ),
            (
                "Sempre compro no lannister.com, porém dessa vez tive uma experiência ruim, já passou o prazo de entrega e eu não recebi minhas poltronas além de ter pago um frete abusivo",
                ["atraso_ou_nao_recebido", "frete_caro"],
                "negativo",
            ),
            (
                "Como eu estava precisando das tintas fiz o pedido só achei o prazo da entrega um pouco longo tirando isso esta tudo ok muito bom",
                ["prazo_longo"],
                "misto",
            ),
            (
                "O produto é bom, mas no anúncio não constam as medidas o que é muito importante pra se fazer um investimento dessa ordem! Arrisquei e é comprei mesmo sem ter as medidas mas foi um risco desnecessário",
                ["outro"],
                "misto",
            ),
        ]

        acertos = 0
        for texto, problemas_esperados, sentimento_esperado in casos:
            c = classificar(texto, modelo)
            ok = (
                set(c.problemas) == set(problemas_esperados)
                and c.sentimento == sentimento_esperado
            )
            acertos += ok

            print(f"{'✓' if ok else '✗'} {c.sentimento:8} {c.problemas}")
            if not ok:
                print(f"  esperado: {sentimento_esperado:8} {problemas_esperados}")
            print(f"  {texto[:80]}\n")

        print(f"{acertos}/{len(casos)} como esperado")
