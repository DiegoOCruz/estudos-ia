# 01 — RAG do zero

Um sistema de RAG (*Retrieval-Augmented Generation*) construído sem frameworks,
usando apenas **numpy** e modelos locais servidos pelo **Ollama**. O objetivo não
era só fazer funcionar, mas entender cada peça: o que é um embedding, como a
similaridade é calculada e onde o sistema falha.

O cenário de teste é o atendimento de uma loja de Funkos e impressão 3D.

## Arquivos

| Arquivo | O que faz |
|---------|-----------|
| `01_similaridade.py` | Gera embeddings de frases e monta a matriz de similaridade de todas contra todas |
| `02_busca.py` | Busca os documentos mais relevantes (top-k) de um catálogo para cada pergunta |
| `03_rag.py` | RAG completo: recupera os documentos e gera a resposta com um LLM |

Cada arquivo acrescenta uma camada ao anterior.

## Como funciona

1. **Indexação:** cada documento do catálogo vira um vetor (embedding) normalizado.
2. **Consulta:** a pergunta vira um vetor da mesma forma.
3. **Recuperação:** a similaridade de cosseno entre a pergunta e todos os documentos
   é calculada numa única multiplicação de matriz por vetor (`docs @ q`), e os `k`
   maiores scores são selecionados.
4. **Geração:** os documentos recuperados são colados no prompt, e o LLM responde
   com base apenas neles.

Com vetores normalizados (norma 1), a similaridade de cosseno se reduz ao produto
escalar, e comparar todos contra todos vira `U @ U.T`.

## Experimentos e descobertas

### 1. O termo "Funko Pop" domina o vetor

Comparei dois modelos de embeddings com as mesmas frases:

| Par de frases | nomic-embed-text | bge-m3 |
|---------------|:----------------:|:------:|
| Funko Pop do Homem-Aranha × Boneco colecionável do Spider-Man | 0,66 | 0,46 |
| Funko Pop do Homem-Aranha × Funko Pop do Batman | 0,72 | 0,73 |
| Funko Pop do Homem-Aranha × Spider-Man Funko Pop | 0,71 | 0,75 |
| Boneco colecionável do Spider-Man × Spider-Man collectible figure | 0,69 | 0,81 |
| Receita de bolo × demais frases (faixa) | 0,30–0,52 | 0,26–0,35 |

- O `nomic-embed-text` funciona melhor em inglês do que em português.
- O `bge-m3`, multilíngue, aproxima muito bem traduções e separa melhor o que é
  irrelevante (veja a faixa do bolo).
- **Nos dois modelos, um Funko de outro personagem fica mais próximo do que um
  sinônimo do mesmo personagem.** Em textos curtos, o termo que se repete (a
  categoria) pesa mais do que o termo que diferencia (o personagem).

Conclusão: trocar de modelo não resolve esse problema. Ele é estrutural, e a
solução é tratar atributos como personagem e linha como **filtros estruturados**,
combinados com busca por palavra-chave e busca semântica (**busca híbrida**).

Modelo escolhido para os passos seguintes: `bge-m3`.

### 2. Busca semântica sem palavras em comum

A pergunta *"Posso devolver um boneco que comprei?"* recuperou em primeiro lugar a
política de **trocas**, sem nenhuma palavra relevante em comum. Uma busca por
palavras-chave teria falhado aqui.

### 3. A busca não sabe dizer "não sei"

Para *"Vocês vendem PlayStation 5?"*, a busca devolveu mesmo assim três documentos,
com a impressão 3D em primeiro lugar:

| Pergunta | Scores do top 3 |
|----------|-----------------|
| Tempo de impressão (acerto) | 0,71 · 0,48 · 0,48 |
| Devolução (acerto) | 0,47 · 0,38 · 0,38 |
| PlayStation 5 (não existe) | 0,36 · 0,32 · 0,32 |

Sinais de que não há resposta: score máximo baixo e scores achatados. Um limite
mínimo ajuda como filtro, mas é frágil (um acerto deu 0,47; um ruído deu 0,48).
Valores absolutos não são comparáveis entre perguntas diferentes, só a ordem e a
distância dentro da mesma pergunta.

### 4. O LLM extrapola de forma plausível

Com o `qwen2.5:14b` e a instrução de usar apenas o contexto fornecido, as respostas
foram corretas e a pergunta sobre o PlayStation foi recusada. Mas surgiram
afirmações que **não estavam nos documentos**:

- *"a política abrange trocas e não devoluções para reembolso"*: o documento não
  menciona reembolso.
- *"não comercializamos videogames"*: o modelo viu só 3 documentos e generalizou
  para a loja inteira.

Não é alucinação escancarada, é extrapolação plausível, e por isso passa
despercebida. Detectar isso em escala exige avaliação sistemática.

## Quando usar RAG

Com um catálogo de 8 itens, a busca nem seria necessária: caberia tudo no prompt.
A recuperação passa a ser indispensável quando a base cresce, por causa do limite
de contexto do LLM, do custo de processar tudo a cada pergunta e da perda de
qualidade com contexto em excesso.

## Próximos passos

- Guardar os embeddings num banco vetorial (PostgreSQL + pgvector)
- Busca híbrida: filtros estruturados + palavra-chave + semântica
- Avaliação sistemática das respostas

## Como rodar

Na raiz do repositório:

```bash
pip install -r requirements.txt
ollama pull bge-m3
ollama pull qwen2.5:14b
python 01-rag-do-zero/03_rag.py
```
