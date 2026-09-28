import ollama
import numpy as np

frases = [
    "Funko Pop do Homem-Aranha",
    "Boneco colecionável do Spider-Man",
    "Funko Pop do Batman",
    "Filamento PLA preto de 1kg",
    "Receita de bolo de cenoura",
    "Spider-Man Funko Pop",
    "Spider-Man collectible figure",
]

resp = ollama.embed(model="bge-m3", input=frases)
vetores = np.array(resp["embeddings"])

# def cosseno(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# for i, frase in enumerate(frases):
#     print(f"{cosseno(vetores[0], vetores[i]):.3f}  {frase}")
normas = np.linalg.norm(vetores, axis=1, keepdims=True)
unitarios = vetores / normas
sim = unitarios @ unitarios.T

for i, frase in enumerate(frases):
    print(i, frase)
print(np.round(sim, 2))