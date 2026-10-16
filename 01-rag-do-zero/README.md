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
   com exploração, limpeza, indexação, índice HNSW e classificação de cada
   comentário com um LLM.

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
| `classificacao.py` | Módulo de classificação: categorias, prompt, few-shot, `classificar` e o teste com respostas esperadas |
| `11_medir_classificacao.py` | Mede o tempo de classificação numa amostra e projeta para o total |
| `schema_classificacao.sql` | Colunas `problemas`, `sentimento` e `modelo_classificacao` na tabela `avaliacoes` |
| `migracao_frete_prazo.sql` | Migração: acrescenta `frete_caro` e `prazo_longo` à restrição de `problemas` |
| `12_classificar_avaliacoes.py` | Classifica todas as avaliações em lotes, de forma retomável |
| `13_medir_paralelo.py` | Compara a classificação um por vez × pedidos simultâneos |

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
segurança, mas perde as que estão escondidas nas notas medianas. O item 14
mede quantas.

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

### 13. Classificação dos comentários com LLM

Cada comentário passou pelo LLM uma vez e ganhou duas colunas novas: a **lista
de problemas** citados e o **sentimento**. Assim dá para filtrar pelo conteúdo,
e não pela nota.

**O formato da resposta** (Pydantic, saída estruturada, temperatura zero):

```python
class Classificacao(BaseModel):
    problemas: list[Problema]   # lista vazia = nenhum problema
    sentimento: Literal["positivo", "negativo", "misto", "neutro"]
```

**As categorias saíram dos dados, não da imaginação.** Antes de escrever o
código, li 30 comentários sorteados (notas ≤ 3) e agrupei por assunto. A
primeira versão teve 7 categorias; a análise do resultado (item 14) revelou
mais duas.

| Categoria | Quando usar |
|---|---|
| `atraso_ou_nao_recebido` | atrasou ou ainda não chegou |
| `problema_na_entrega` | endereço errado, retirada no correio |
| `pedido_incompleto` | faltou item do pedido |
| `produto_errado_ou_diferente` | outro produto, falsificado, diferente do anúncio |
| `defeito_ou_qualidade` | defeito, danificado, má qualidade |
| `atendimento_ou_financeiro` | o cliente **reclama** do atendimento: sem resposta, nota fiscal, cobrança indevida, reembolso |
| `frete_caro` | acha o valor do frete alto |
| `prazo_longo` | chegou no prazo, mas o prazo é longo |
| `outro` | não se encaixa nas anteriores |

**Decisões de design:**

- **Lista, e não categoria única:** "veio apenas de microfibra" é pedido
  incompleto e produto errado ao mesmo tempo.
- **Atraso e não recebido juntos:** quando o cliente escreve, não dá para saber
  se o produto vai chegar atrasado ou nunca.
- **`problemas` antes de `sentimento`:** o LLM escreve da esquerda para a
  direita, e o sentimento sai coerente com os problemas já listados.
- **A nota não vai para o modelo:** para poder detectar quando o texto e a nota
  se contradizem.
- **`outro` existe de propósito:** sem ele, o modelo força o problema na
  categoria menos errada.

**Ajustando o prompt, um caso de cada vez:**

| Problema | O que não funcionou | O que funcionou |
|---|---|---|
| "Não recebi, tive que buscar no correio" virava atraso | Definição mais precisa ("ainda não está com ele"): nada mudou, e quebrou outro caso | **Few-shot:** 4 exemplos resolvidos, entregues como conversa (user → assistant) |
| "Como realizo a troca?" virava atendimento | — | Definição: o cliente precisa **reclamar** do atendimento; pedir troca não conta |

- **O modelo segue a definição ao pé da letra:** "ainda não está com ele" fez
  "Ainda não posso avaliar... quando chegar" virar atraso. O cliente não estava
  reclamando, mas a definição não dizia que precisava.
- **O prompt funciona como um todo:** mudar a definição de atendimento fez o
  14B acertar um caso de prazo que não tinha nada a ver com atendimento.
- **Exemplos do few-shot ≠ casos do teste:** senão o modelo copia a resposta e
  o teste não prova nada.
- **Exemplos dos dois lados da fronteira:** um atraso de verdade entre os
  exemplos, para o modelo não aprender a evitar a categoria.

**O teste:** 16 casos com resposta esperada, conferidos automaticamente (✓/✗ e
placar). Cada falha nova encontrada em dados reais virou um caso. Um caso
ambíguo (a cortina, que admite duas leituras) foi **retirado**: ele não separa
um modelo bom de um ruim, só gera ruído no placar.

**A escolha do modelo:**

| | `qwen2.5:14b` | `qwen3.8:27b` (`think=False`) |
|---|:-:|:-:|
| Teste final (16 casos) | 11/16 | **15/16** |
| Estabilidade entre versões do prompt | oscilou (perdia casos já acertados) | estável |
| Tempo por comentário | 0,25 s | 0,70–0,85 s |
| Tempo para 41 mil | ~3 h | ~8–9 h |

O 27B foi escolhido: a classificação é feita uma vez, e uma classificação pior
fica no banco para sempre.

**Execução:** lotes de 100, com `commit` por lote, processando só
`WHERE sentimento IS NULL`. O banco sabe o que falta, então o script é
retomável: uma queda de energia no meio da noite custou só o lote em
andamento. Antes de confiar nisso por 9 horas, testei de propósito: Ctrl+C
depois de 4 lotes deixou exatamente 400 linhas gravadas.

**NULL × `{}`:** `problemas` NULL = ainda não classificada; `{}` = classificada,
sem problemas. É essa distinção que torna o script retomável.

### 14. O que a classificação revelou

**Problemas citados** (41.431 comentários, 19.416 menções):

| Problema | Menções |
|---|---:|
| atraso_ou_nao_recebido | 5.454 |
| defeito_ou_qualidade | 3.831 |
| atendimento_ou_financeiro | 3.144 |
| produto_errado_ou_diferente | 2.981 |
| pedido_incompleto | 2.589 |
| problema_na_entrega | 842 |
| frete_caro | 264 |
| prazo_longo | 204 |
| outro | 107 |

**A nota esconde um terço das reclamações** (números da primeira classificação
completa):

| Nota | Comentários | Com reclamação (negativo + misto) |
|:-:|---:|---:|
| 1 | 9.071 | 96% |
| 2 | 2.200 | 93% |
| 3 | 3.635 | 74% |
| 4 | 5.996 | **32%** |
| 5 | 20.529 | 5,5% |

De 16.504 comentários com reclamação, um filtro `nota <= 2` pega 10.772 e
**perde 5.732 (35%)**. Quase um em cada três clientes que deram nota 4
reclamou de alguma coisa no texto.

**Divergências entre nota e texto:** as duas células parecem simétricas, mas
significam coisas diferentes.

| | Nota 5 + negativo (476) | Nota 1 + positivo (79) |
|---|---|---|
| Amostra de 15 | 15 reclamações reais | 14 notas erradas, 1 erro do modelo |
| Significado | o cliente reclamou apesar da nota alta (muitos escreveram enquanto esperavam uma solução) | o cliente errou a nota (talvez invertendo a escala) |

O erro do modelo foi uma crítica indireta: *"Quando os produtos eram vendidos
pela própria loja, chegavam sempre antes do prazo e sem problemas"* saiu
positivo. O modelo leu as palavras, não o que a frase implica.

**Uma categoria larga é um `outro` disfarçado.** Na primeira rodada, só 335
menções caíram em `outro`, o que parecia provar que as categorias cobriam bem
os dados. Lendo uma amostra de `atendimento_ou_financeiro`, apareceu o **frete
caro** escondido ali: a palavra "financeiro" absorvia reclamações de preço. E o
`outro` escondia o **prazo longo**.

As duas categorias foram criadas, testadas, acrescentadas ao `CHECK` por uma
migração, e só as ~3.800 linhas afetadas foram reclassificadas: bastou apagar a
classificação delas (`sentimento = NULL`) e rodar o mesmo script. O mecanismo
de retomada virou, de graça, um mecanismo de reclassificação seletiva.

| | Antes | Depois |
|---|---:|---:|
| atendimento_ou_financeiro | 3.491 | 3.144 |
| outro | 335 | 107 |
| frete_caro | — | 264 |
| prazo_longo | — | 204 |

**A amostra descobre; não mede.** A amostra de 15 sugeria uns 900 casos de
frete caro escondidos em atendimento; saíram 264. Com 15 casos, 4 em 15 é
compatível com uma taxa real entre menos de 10% e mais de 50%. A amostra
serviu para descobrir que a categoria existia; para medir quantas eram, foi
preciso classificar tudo.

**Limitações conhecidas:**

- Comentários muito curtos com erro de digitação podem sair invertidos:
  *"Produto ce qualidade"* (nota 5) sai negativo com defeito nos dois modelos.
- Ironia e críticas indiretas passam como elogio.
- A fronteira entre negativo e misto é difusa ("Chegou no prazo porém veio
  arranhado" saiu negativo).
- Temperatura zero não garante resultado idêntico: cerca de 1 em 60
  classificações muda entre duas execuções, sempre em casos de fronteira.
- A reclassificação seletiva só releu atendimento e `outro`: um frete caro que
  tenha ido para outra categoria não foi revisto.

### 15. Velocidade: raciocínio escondido e paralelismo

**O modelo de 27B estava "pensando" antes de responder.** Medido com a mesma
amostra, ele levava 3,10 s por comentário, contra 0,24 s do 14B: 13× mais
lento por menos do que o dobro do tamanho. O `ollama ps` mostrou `100% GPU`, o
que descartou falta de memória. A causa era o **modo de raciocínio** da família
Qwen3: um texto escondido gerado antes de cada JSON.

| | Por comentário | 41 mil |
|---|---:|---:|
| 27B com raciocínio | 3,10 s | 36 h |
| 27B com `think=False` | 0,70 s | 8 h |

A conta que fecha: a geração é limitada pela **largura de banda da memória** da
GPU (tokens/s ≈ banda ÷ tamanho do modelo). 17 GB contra 9 GB explicam ~2×, não
13×.

**Paralelismo: não funcionou, e o motivo estava no log.** A ideia era enviar
vários comentários ao mesmo tempo (`OLLAMA_NUM_PARALLEL=4` + threads no
script), aproveitando a força de cálculo que sobra enquanto a placa lê o
modelo.

- **27B: ganho de 1,0×.** O log do Ollama dizia: *"model architecture does not
  currently support parallel requests"* (arquitetura `qwen35`).
- **14B: ficou 6× mais lento.** Cada pedido simultâneo reserva o seu próprio
  contexto (32.768 tokens). Com 4, o modelo foi de 15 GB para 35 GB, e 35% foi
  para a CPU.
- **Revertido:** a variável vale para o servidor inteiro, e o modelo escolhido
  não se beneficia. Para um próximo uso, a receita é limitar o contexto por
  pedido (`num_ctx: 4096`; o prompt usa ~1.000 tokens) e conferir no
  `ollama ps` que tudo continua `100% GPU`.

## Próximos passos

- Combinar a busca semântica com o filtro por problema no `10_buscar_avaliacoes.py`
  (ex.: "reclamações de entrega" + `'problema_na_entrega' = ANY(problemas)`)
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

Classificação (banco novo: o `schema_classificacao.sql` já traz as nove
categorias; banco criado antes dele: rodar também a migração):

```bash
ollama pull qwen3.8:27b
docker exec -i pgvector psql -U rag -d rag < schema_classificacao.sql
python classificacao.py                  # teste: 15/16 esperado
python 11_medir_classificacao.py         # tempo numa amostra
nohup python -u 12_classificar_avaliacoes.py > classificacao.log 2>&1 &
tail -f classificacao.log
```