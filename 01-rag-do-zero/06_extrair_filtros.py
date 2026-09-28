from comum import extrair_filtros

perguntas = [
    "Vocês têm Funko do Homem-Aranha até 150 reais?",
    "Qual o prazo pra imprimir uma peça?",
    "Posso trocar um produto?",
    "Tem algo legal pra dar de presente?",
    "Quanto custa o frete?",
]

for p in perguntas:
    print(f"> {p}")
    print(f"  {extrair_filtros(p)}")