import ollama
import numpy as np

LLM = "qwen2.5:14b"
MODELO = "bge-m3"

def embed(textos):
    resp = ollama.embed(model=MODELO, input=textos)
    v = np.array(resp["embeddings"])
    return v / np.linalg.norm(v, axis=1, keepdims=True)

catalogo = [
    "Funko Pop Marvel Homem-Aranha, figura de vinil de 10 cm do Spider-Man em pose clássica, caixa com janela.",
    "Funko Pop DC Batman, figura de vinil de 10 cm do Cavaleiro das Trevas com capa preta.",
    "Funko Pop Marvel Venom, o vilão simbionte inimigo do Homem-Aranha, figura de vinil de 10 cm.",
    "Action figure articulada do Spider-Man, 30 cm, com 20 pontos de articulação e acessórios.",
    "Impressão 3D sob encomenda: peças em PLA ou PETG a partir do seu arquivo STL. Prazo médio de 3 a 5 dias úteis.",
    "Filamento PLA preto 1 kg, diâmetro 1,75 mm, para impressoras 3D.",
    "Política de trocas: produtos lacrados podem ser trocados em até 7 dias após o recebimento.",
    "Frete grátis para compras acima de R$ 200 para todo o Brasil.",
]

docs = embed(catalogo)

def buscar(pergunta, k=3):
    q = embed([pergunta])[0]
    scores = docs @ q
    melhores = np.argsort(scores)[::-1][:k]
    return [(scores[i], catalogo[i]) for i in melhores]

def responder(pergunta, k=3):
    resultados = buscar(pergunta, k)
    contexto = "\n".join(f"- {doc}" for _, doc in resultados)

    instrucoes = (
        "Você é o atendente de uma loja de Funkos e impressão 3D. "
        "Responda usando APENAS as informações fornecidas. "
        "Se elas não responderem à pergunta, diga que não temos essa informação ou esse produto. "
        "Nunca invente produtos, preços ou prazos."
    )

    resp = ollama.chat(model=LLM, messages=[
        {"role": "system", "content": instrucoes},
        {"role": "user", "content": f"Informações:\n{contexto}\n\nPergunta: {pergunta}"},
    ])
    return resp["message"]["content"]

perguntas = [
    "Vocês têm Funko do Homem-Aranha?",
    "Quanto tempo leva pra imprimir uma peça minha?",
    "Posso devolver um boneco que comprei?",
    "Queria um presente de super-herói pra uma criança",
    "Vocês vendem PlayStation 5?",
]

for p in perguntas:
    print(f"\n> {p}")
    print(responder(p))