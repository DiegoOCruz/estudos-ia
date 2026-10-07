# 01 — RAG do zero

Um sistema de RAG (*Retrieval-Augmented Generation*) construído sem frameworks,
com modelos locais servidos pelo **Ollama**. O objetivo não era só fazer
funcionar, mas entender cada peça: o que é um embedding, como a similaridade é
calculada, como guardar e buscar vetores num banco de dados e onde o sistema
falha.

O módulo tem duas partes:

1. **Catálogo fictício** de uma loja de Funkos e impressão 3D: do numpy até um
   pipeline completo com PostgreSQL, filtros e LLM.
2. **Dados reais:** 41 mil avaliações de clientes do dataset público da Olist,
   com exploração, limpeza, indexação e índice HNSW.

## Arquivos

| Arquivo | O que faz |
|---------|-----------|
| `01_similaridade.py` | Gera embeddings de frases e monta a matriz de similaridade de todas contra todas |
| `02_busca.py` | Busca os documentos mais relevantes (top-k) de um catálogo, com numpy |
| `03_rag.py` | RAG completo em memória: recupera os documentos e gera a resposta com um LLM |
| `schema.sql` | Tabela `documentos` (catálogo) no PostgreSQL + pgvector |
| `04_indexar.py` | Vetoriza o catálogo e grava no banco, uma única vez |
| `05_busca_pg.py` | Busca no banco com o operador de distância de cosseno e filtros opcionais |
| `06_extrair_filtros.py` | Testa a extração de filtros da pergunta com o LLM |
| `07_rag_completo.py` | Pipeline completo: extrair filtros → buscar no banco → responder |
| `comum.py` | Código compartilhado: `embed`, `buscar`, `extrair_filtros` |
| `dados_olist.py` | Limpeza das avaliações da Olist (`limpar_avaliacoes`) |
| `08_explorar_avaliacoes.py` | Exploração do dataset com pandas |
| `schema_avaliacoes.sql` | Tabela `avaliacoes` com restrições de integridade |
| `09_indexar_avaliacoes.py` | Vetoriza e grava as avaliações em lotes |
| `10_buscar_avaliacoes.py` | Busca semântica nas avaliações, com filtro opcional por nota |

## Como funciona

1. **Indexação:** cada documento vira um vetor (embedding) normalizado e é
   gravado no banco, junto com colunas estruturadas. Feita uma vez.
2. **Consulta:** só a pergunta é vetorizada.
3. **Recuperação:** o banco calcula a distância de cosseno (`<=>`) entre a
   pergunta e os vetores guardados, aplica os filtros e devolve os `k` mais
   próximos.
4. **Geração:** os textos recuperados são colados no prompt, e o LLM responde
   com base apenas neles. O LLM nunca vê os vetores, só os textos.

Com vetores normalizados (norma 1), a similaridade de cosseno se reduz ao
produto escalar. Por isso a normalização é feita uma vez, na indexação.

---

# Parte 1: catálogo fictício

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
  irrelevante.
- **Nos dois modelos, um Funko de outro personagem fica mais próximo do que um
  sinônimo do mesmo personagem.** Em textos curtos, o termo que se repete (a
  categoria) pesa mais do que o termo que diferencia (o personagem).

Conclusão: o problema é estrutural, e não do modelo. Modelo escolhido: `bge-m3`.

### 2. Busca semântica sem palavras em comum

*"Posso devolver um boneco que comprei?"* recuperou em primeiro lugar a
política de **trocas**, sem nenhuma palavra relevante em comum.

### 3. A busca não sabe dizer "não sei"

| Pergunta | Scores do top 3 |
|----------|-----------------|
| Tempo de impressão (acerto) | 0,71 · 0,48 · 0,48 |
| Devolução (acerto) | 0,47 · 0,38 · 0,38 |
| PlayStation 5 (não existe) | 0,36 · 0,32 · 0,32 |

Sinais de que não há resposta: score máximo baixo e scores achatados. Valores
absolutos não são comparáveis entre perguntas diferentes.

### 4. O LLM extrapola de forma plausível

Com o `qwen2.5:14b` instruído a usar só o contexto, surgiram afirmações que não
estavam nos documentos: *"não comercializamos videogames"* (generalizou a loja
inteira a partir de 3 documentos) e, mais tarde, *"disponível"* (o banco nem tem
dado de estoque). Não é alucinação escancarada; soa certo, e por isso passa
despercebida.

### 5. Do numpy para o PostgreSQL + pgvector

- **Indexação separada da consulta:** os embeddings do catálogo são calculados
  uma vez e guardados; cada busca vetoriza só a pergunta.
- **A conta passou para o SQL:** `docs @ q` virou `1 - (embedding <=> q)`, e o
  `argsort` virou `ORDER BY ... LIMIT k`. Os scores bateram exatamente com os
  do numpy, o que validou a migração.
- **Colunas estruturadas junto com o vetor:** tipo, personagem, preço, SKU e
  código de barras. Preço e códigos ficam fora do texto do embedding: modelos
  representam números mal, e o preço muda com frequência.

### 6. Filtros: restrição rígida × preferência

- **Filtros restringem, a similaridade ordena.** Mas filtros dizem o que é
  *permitido*, não o que é *relevante*: com preço máximo de R$ 150, um filamento
  de impressora entrou no top 3 de "presente de super-herói", com score 0,32.
- Filtrar por personagem eliminou o **Venom** da busca por Homem-Aranha, mas ele
  é uma sugestão natural para esse cliente. Personagem é **preferência**, não
  restrição: fica com a busca semântica. Preço máximo é **restrição**: vira
  `WHERE`.

### 7. O LLM transformando a pergunta em filtros

Antes da busca, o LLM extrai as restrições da pergunta em formato estruturado
(Pydantic + JSON Schema, temperatura zero). O `Literal` impede valores
inventados.

| Pergunta | Filtros extraídos |
|----------|-------------------|
| Vocês têm Funko do Homem-Aranha até 150 reais? | `preco_max=150.0 tipo='produto'` |
| Posso trocar um produto? | `tipo='politica'` |
| Vocês vendem PlayStation 5? | nenhum |

"Posso **trocar um produto**?" foi classificada como política, apesar da palavra
"produto": o modelo entendeu a intenção, não a palavra.

---

# Parte 2: dados reais (avaliações da Olist)

Dataset: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
arquivo `olist_order_reviews_dataset.csv`. Os dados não estão no repositório;
baixe e coloque em `dados/`.

### 8. Exploração

- De **100.000** avaliações, só **41.753** têm comentário escrito (42%).
- Os comentários vêm dos extremos, um caso de **viés de seleção**:

| Nota | Comentários | Tamanho médio |
|:----:|------------:|--------------:|
| 5 | 20.646 | 52 caracteres |
| 4 | 6.034 | 62 |
| 3 | 3.665 | 85 |
| 2 | 2.229 | 98 |
| 1 | 9.179 | 102 |

- Hipótese testada e **refutada**: comentários curtos indicariam insatisfação. É
  o contrário: quanto pior a nota, mais longo o texto. Quem está satisfeito
  escreve "muito bom"; quem está insatisfeito explica o problema.

### 9. Limpeza

| Regra | Comentários afetados |
|-------|---------------------:|
| Espaços e quebras de linha (`\s+` → um espaço) | 4.003 |
| Códigos de rastreio dos Correios → `[RASTREIO]` | 17 |
| Duplicatas de `review_id` removidas | 322 |

- **Medir antes de investir:** o código de rastreio chamava atenção nos
  sorteios, mas afetava menos de 0,05% dos textos.
- **Marcar em vez de apagar:** apagar os códigos quebrava frases ("consta
  somente o , que recebi..."). A marca preserva a informação de que o cliente
  citou um rastreio.
- **Regex engana:** `\r|\n|\s+` transforma `\r\n` em dois espaços, porque o `|`
  fica com a primeira alternativa que funcionar. `\s+` sozinho resolve.
- O texto original foi preservado; a limpeza gera uma coluna nova.

**Resultado:** 41.431 avaliações limpas.

### 10. Indexação

- Tabela com restrições de integridade: `nota SMALLINT NOT NULL CHECK (nota
  BETWEEN 1 AND 5)`, `review_id UNIQUE`, `comment NOT NULL`.
- Vetorização em lotes de 100: 2.000 avaliações em **12 s**. Projeção por regra
  de três: ~4 minutos para todas. Indexadas as **41.431**.

### 11. Busca semântica com textos reais

| Consulta | Encontrou |
|----------|-----------|
| a mercadoria veio danificada | "infelizmente recebi o produto danificado", "O produto veio estragado" |
| demorou uma eternidade pra chegar | "Demorou muito pra chegar" |
| o vendedor nunca respondeu minhas mensagens | "Ninguém me dá um retorno", "ninguém entrou em contato para resolver nada" |

Uma busca por texto exato (`ILIKE`) não encontraria a maioria desses
resultados.

**A nota não conta a história toda:** reclamações claras aparecem com notas 3
e 4 ("Demorou muito pra chegar", nota 4). A nota resume a experiência inteira;
o comentário fala de uma parte dela. Um filtro `nota <= 2` traz reclamações com
segurança, mas perde as que estão escondidas nas notas medianas.

### 12. Índice HNSW e o planejador do PostgreSQL

| | Plano | Linhas lidas | Tempo no banco |
|---|---|---:|---:|
| Sem índice | `Seq Scan` | 41.431 | 238 ms |
| Com índice HNSW | `Index Scan` | só as necessárias | **2,96 ms** |

- Cerca de **80× mais rápido**, com os mesmos resultados.
- **Criar o índice não bastou:** o planejador continuou escolhendo a varredura
  completa, porque estimava que ela era mais barata. A estimativa estava errada:
  os vetores de 4 KB ficam guardados fora da linha principal (TOAST), e o
  planejador não contabiliza bem esse custo (`width=90` no plano).
- O `EXPLAIN ANALYZE` revelou o problema; o `SET enable_seqscan = off` na
  conexão de busca resolveu.
- Script inteiro, 3 buscas: **0,84 s → 0,13 s**. Agora o tempo é dominado pela
  vetorização das consultas: o gargalo mudou de lugar.

## Próximos passos

- Classificar cada comentário com o LLM (reclamação? sobre o quê?), para
  filtrar pelo conteúdo e não pela nota
- Busca híbrida: semântica + palavra-chave + filtros
- Avaliação sistemática das respostas
- Aplicar ao catálogo real da loja, com estoque

## Como rodar

```bash
pip install -r requirements.txt
ollama pull bge-m3
ollama pull qwen2.5:14b

docker run -d --name pgvector \
  -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag \
  -p 5432:5432 -v pgdata:/var/lib/postgresql/data \
  pgvector/pgvector:pg17

docker exec -it pgvector psql -U rag -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Catálogo fictício:

```bash
docker exec -i pgvector psql -U rag -d rag < schema.sql
python 04_indexar.py
python 07_rag_completo.py
```

Avaliações da Olist (com o CSV em `dados/`):

```bash
docker exec -i pgvector psql -U rag -d rag < schema_avaliacoes.sql
python 09_indexar_avaliacoes.py
docker exec -it pgvector psql -U rag -d rag -c "CREATE INDEX ON avaliacoes USING hnsw (embedding vector_cosine_ops);"
python 10_buscar_avaliacoes.py
```