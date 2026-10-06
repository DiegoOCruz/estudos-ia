# Glossário

Conceitos estudados no repositório, cada um com uma explicação curta, um
exemplo pequeno e onde ele aparece no código.

- [Matemática](#matemática)
- [Embeddings](#embeddings)
- [RAG](#rag)
- [Banco de dados](#banco-de-dados)
- [LLM](#llm)
- [Dados](#dados)
- [Python, numpy e pandas](#python-numpy-e-pandas)

---

## Matemática

### Vetor

**O que é:** uma lista de números. Dá para imaginar como uma seta saindo do
ponto zero. Com 2 números, é uma seta no plano; com 1024, a matemática é a
mesma, só não dá para desenhar.

**Exemplo:** `[1, 2]` é uma seta que anda 1 para a direita e 2 para cima.

**No código:** cada embedding é um vetor. O `bge-m3` gera vetores de 1024 números.

### Produto escalar

**O que é:** multiplica posição por posição e soma tudo. O resultado é um
número só.

**Exemplo:** `[1, 2] · [2, 1] = 1×2 + 2×1 = 4`

**No código:** `np.dot(a, b)` ou `a @ b` (este último só com arrays do numpy).

### Norma

**O que é:** o tamanho do vetor. Multiplica cada número por ele mesmo (eleva
ao quadrado), soma tudo e tira a raiz quadrada. O resultado é um número só.

**Exemplo:** norma de `[3, 4]` = √(9 + 16) = √25 = 5. Norma de `[1, 2]` = √5 ≈ 2,24.

**No código:** `np.linalg.norm(v)`

### Normalização

**O que é:** dividir cada número do vetor pela norma dele. O resultado é um
vetor novo, com a mesma direção e tamanho 1. Dica para lembrar: "normalizar"
contém "norma". Primeiro mede a norma, depois divide por ela.

**Exemplo:** `[1, 2]` normalizado = `[1 ÷ 2,24; 2 ÷ 2,24]` ≈ `[0,45; 0,89]`

**No código:** `v / np.linalg.norm(v, axis=1, keepdims=True)`, na função `embed`.

### Similaridade de cosseno

**O que é:** mede se dois vetores apontam para a mesma direção. Vai de -1
(opostos) a 1 (mesma direção); 0 significa perpendiculares, "nada a ver".
Fórmula: produto escalar dividido pelo produto das normas. Dividir pelas
normas faz o tamanho do vetor não importar, só a direção.

**Exemplo:** `[1, 2]` e `[2, 1]` → 4 ÷ (√5 × √5) = 4 ÷ 5 = **0,8**.
`[3, 4]` e `[6, 8]` → **1,0** (mesma direção, tamanhos diferentes).

**Atalho:** com vetores normalizados, as normas valem 1, e o cosseno vira só o
produto escalar.

**No código:** `docs @ q` (com vetores já normalizados).

### Matriz e transposta

**O que é:** uma matriz é uma tabela de números; empilhando vetores como
linhas, cada linha é um vetor. A transposta vira as linhas em colunas.

**Exemplo:** 7 frases com vetores de 1024 números formam uma matriz 7 × 1024.
A transposta dela é 1024 × 7.

**No código:** `U.T`

### Multiplicação de matrizes

**O que é:** a posição `[i][j]` do resultado é o produto escalar da linha `i`
da primeira matriz com a coluna `j` da segunda. Regra dos formatos: as
dimensões do meio precisam ser iguais e "somem" no resultado.

**Exemplo:** (7 × **1024**) @ (**1024** × 7) = (7 × 7), a matriz de
similaridade de todas as frases contra todas. (8 × **1024**) @ (**1024**) =
(8), os scores de 8 documentos contra uma pergunta.

**Erro típico:** `shapes (1,1024) and (1,1024) not aligned` significa que as
dimensões do meio não batem (1024 contra 1).

**No código:** `unitarios @ unitarios.T` e `docs @ q`.

---

## Embeddings

### Embedding

**O que é:** um texto transformado num vetor de tamanho fixo, que representa o
significado dele. Funciona como uma coordenada num "mapa de significados":
textos parecidos ficam perto, diferentes ficam longe. Nenhum número
isoladamente significa uma palavra; o significado está na combinação de todos,
como latitude e longitude juntas indicam uma cidade.

**Exemplo:** "Posso devolver um boneco?" fica perto de "Política de trocas",
mesmo sem palavras em comum.

**No código:** `ollama.embed(model="bge-m3", input=textos)`

**Sinônimo:** vetorizar.

### Modelo de embeddings × LLM

**O que é:** o modelo de embeddings (como o `bge-m3`) transforma texto em
vetor e serve para **encontrar** textos. O LLM (como o `qwen2.5:14b`) lê texto
e **escreve** respostas. O LLM nunca recebe os vetores; recebe os textos que a
busca encontrou. Analogia: o embedding é o bibliotecário que acha as estantes
certas; o LLM é quem lê os livros e responde.

### Tokenização

**O que é:** o texto é picotado em pedaços (tokens) antes de entrar no modelo.
Palavras comuns viram um pedaço; palavras raras são quebradas em partes. Assim
o modelo lida com qualquer palavra, mesmo uma que nunca viu.

**Exemplo:** "O rato roeu a roupa do rei de Roma" (9 palavras) virou 16 tokens.

**No código:** o campo `prompt_eval_count` da resposta do `ollama.embed`.

### Aprendizado contrastivo

**O que é:** a forma como modelos de embeddings são treinados. Recebem
milhões de pares que deveriam ficar próximos (pergunta e resposta, texto e
tradução) e aprendem a puxar os vetores de cada par para perto e empurrar os
não relacionados para longe. É daí que vem a propriedade de o cosseno medir
semelhança de significado.

### Limitações dos embeddings

**O que é:** embeddings capturam bem **sobre o que** um texto fala, mas mal
**quem fez o quê**. Também são sensíveis a palavras em comum: em textos
curtos, o termo que se repete pesa mais do que o termo que diferencia.

**Exemplos:**
- "O rei de Roma roeu a roupa do rato" (sentido invertido) deu **0,84** com a
  frase original; a paráfrase "Um roedor estragou as vestes do monarca romano"
  deu só **0,71**.
- "Funko Pop do Batman" ficou mais perto de "Funko Pop do Homem-Aranha" do que
  "Boneco colecionável do Spider-Man".

**Consequência:** vetores de modelos diferentes não podem ser comparados entre
si. Trocar o modelo de embeddings exige reindexar tudo.

---

## RAG

### RAG (Retrieval-Augmented Generation)

**O que é:** geração aumentada por recuperação. Em vez de o LLM responder só
com o que sabe, o sistema primeiro **recupera** os textos relevantes de uma
base e os cola no prompt, e o LLM responde com base neles.

**Quando usar:** quando existe o problema de **encontrar** a informação numa
base grande demais para caber no prompt. Se o documento certo já está na mão
(como no corretor de trabalhos), não é preciso recuperar: basta entregá-lo ao
modelo.

### Indexação

**O que é:** a fase em que os documentos são vetorizados e guardados. Feita
uma vez, antes das perguntas, e pode demorar, porque ninguém está esperando.

**No código:** `04_indexar.py` e `09_indexar_avaliacoes.py`.

### Consulta (recuperação)

**O que é:** a fase em que chega uma pergunta, só ela é vetorizada, e os
documentos mais próximos são buscados. Precisa ser rápida.

**No código:** `05_busca_pg.py` e a função `buscar`.

### Top-k

**O que é:** os `k` documentos mais similares à pergunta. A busca sempre
devolve `k` resultados, mesmo quando nenhum é relevante: ela sabe ordenar,
mas não sabe dizer "não tenho".

**Sinais de que não há resposta:** score máximo baixo e scores achatados (como
o PlayStation 5: 0,36 · 0,32 · 0,32).

**Cuidado:** scores absolutos não se comparam entre perguntas diferentes, só a
ordem e a distância dentro da mesma pergunta.

### Extrapolação plausível

**O que é:** quando o LLM afirma algo que não está nos documentos, mas que
soa certo. Não é uma alucinação escancarada, e por isso passa despercebida.

**Exemplos:** "não comercializamos videogames" (generalizou para a loja
inteira a partir de 3 documentos); "disponível" (o banco nem tem estoque).

### Busca híbrida

**O que é:** combinar busca semântica (embeddings), busca por palavras exatas
e filtros estruturados. Cada uma cobre o ponto fraco da outra: a semântica
acha "devolver" ≈ "trocas"; a palavra exata acerta nomes próprios; os filtros
aplicam regras como preço máximo.

### Filtro rígido × preferência

**O que é:** um filtro rígido exclui o que não atende (preço máximo vira
`WHERE`). Uma preferência só prioriza, sem excluir: personagem fica com a
busca semântica, para o Venom continuar aparecendo para quem procura
Homem-Aranha. Filtros dizem o que é **permitido**, não o que é **relevante**.

### Self-query

**O que é:** usar o LLM para transformar a pergunta do cliente em filtros
estruturados antes da busca.

**Exemplo:** "Funko do Homem-Aranha até 150 reais?" → `preco_max=150.0 tipo='produto'`

**No código:** `extrair_filtros` no `comum.py`.

---

## Banco de dados

### pgvector

**O que é:** extensão do PostgreSQL que adiciona o tipo `vector` e operadores
de distância. Permite guardar vetores junto com colunas comuns e combinar
busca semântica com `WHERE`.

**No código:** `embedding vector(1024) NOT NULL`

### Distância de cosseno (`<=>`)

**O que é:** o operador do pgvector que calcula `1 - similaridade de cosseno`.
Quanto **menor**, mais parecido. Por isso o `ORDER BY` é crescente.

**No código:**
```sql
SELECT 1 - (embedding <=> %(q)s) AS score, conteudo
FROM documentos
ORDER BY embedding <=> %(q)s
LIMIT %(k)s
```

### Idempotência

**O que é:** uma operação que pode ser repetida quantas vezes for preciso, com
o mesmo resultado final.

**No código:** o `TRUNCATE ... RESTART IDENTITY` no começo dos scripts de
indexação. Rodar duas vezes não duplica nada.

### SQL injection e parâmetros

**O que é:** se um valor externo for colado no texto do SQL, um texto com
aspas quebra a consulta, e um texto malicioso pode executar comandos no
banco. A proteção é enviar os valores como **parâmetros** (`%s`), separados do
SQL.

**Regra:** nunca coloque um valor externo dentro do texto do SQL. Montar a
estrutura da consulta com trechos escritos por você (como o `WHERE` dos
filtros) é seguro.

### Restrições (NOT NULL, UNIQUE, CHECK)

**O que é:** regras que o próprio banco garante, mesmo que o código tenha bug.
`NOT NULL` proíbe valor vazio; `UNIQUE` proíbe repetição (e cria um índice
automaticamente); `CHECK` valida uma condição.

**Pegadinha:** o `CHECK` deixa passar `NULL`, porque a comparação com nulo dá
"desconhecido", e não "falso". Para exigir o valor, use `NOT NULL` junto.

**No código:** `score SMALLINT NOT NULL CHECK (score BETWEEN 1 AND 5)`

### Tipos de dados

**O que é:** escolher o tipo certo descreve o dado e evita erros.
- `NUMERIC(10, 2)` para dinheiro: exato, sem erros de arredondamento.
- `SMALLINT` para números inteiros pequenos, como uma nota de 1 a 5.
- `TEXT` para códigos de barras: podem começar com zero e nunca entram em conta.

**Cuidado:** `0.1 + 0.2` em ponto flutuante dá `0.30000000000000004`. Em
Python, dinheiro vai como `Decimal("149.90")`, e `Decimal(str(x))` quando vier
de um float.

---

## LLM

### Prompt de sistema

**O que é:** a mensagem com papel `system`, que define **como** o modelo deve
se comportar. A mensagem `user` traz os **dados** da consulta.

### Temperatura

**O que é:** o grau de aleatoriedade das respostas. Zero faz o modelo sempre
escolher a opção mais provável: bom para extrair dados (consistência). Valores
maiores dão mais variedade: bom para redigir textos.

**No código:** `options={"temperature": 0}`

### Saída estruturada

**O que é:** forçar o LLM a responder num formato exato, descrito por um
schema JSON. O Pydantic descreve o formato como uma classe e valida a resposta.
O `Literal` limita um campo a valores permitidos, e o modelo não consegue
inventar outro.

**No código:**
```python
class Filtros(BaseModel):
    preco_max: Optional[float] = None
    tipo: Optional[Literal["produto", "servico", "politica"]] = None
```

---

## Dados

### DataFrame

**O que é:** a tabela do pandas, com linhas e colunas.

**No código:** `pd.read_csv(...)`, `.shape`, `.columns`, `.head()`, `.info()`

### NaN

**O que é:** "Not a Number", a forma do pandas marcar um valor vazio.

**No código:** `.isna()` e `.notna()` dão True/False para cada linha; somando
com `.sum()`, contam os vazios ou preenchidos (True vale 1, False vale 0).

### Viés de seleção

**O que é:** quando os dados não representam todo o grupo, só quem decidiu
participar. Sempre pergunte **quem** gerou os dados que você tem.

**Exemplo:** só 42% dos clientes da Olist escreveram comentário, e os que
escrevem são os extremos: muitas notas 5 e 1, poucas notas 2.

### Média × mediana

**O que é:** a média é sensível a valores extremos (um texto de 2.000
caracteres puxa a média para cima). A mediana é o valor do meio, com tudo
ordenado, e não se deixa levar por extremos.

**Exemplo:** tamanho médio dos comentários por nota: 102 caracteres na nota 1,
52 na nota 5. Clientes insatisfeitos escrevem o dobro.

### Expressão regular (regex)

**O que é:** uma linguagem para descrever padrões de texto.
- `\s+`: um ou mais espaços em branco de qualquer tipo (inclui `\r` e `\n`).
- `\d{3}`: três dígitos.
- `?`: o item anterior é opcional.
- `|`: "ou".

**Pegadinha:** o `|` fica com a **primeira** alternativa que funcionar, não a
mais longa. `\r|\n|\s+` transforma `\r\n` em **dois** espaços; `\s+` sozinho
resolve.

**No código:** `.str.replace(r"\s+", " ", regex=True)`

### Princípios de limpeza

- **Não destrua o dado original:** crie colunas novas, como `texto_limpo`.
- **Meça antes de investir:** 17 códigos de rastreio contra 4.003 quebras de linha.
- **Confira com os olhos:** contagens dizem quantos, não como ficou.
- **Não apague o que pode ter valor:** os comentários curtos ficaram.
- **Filtre primeiro, processe depois:** remover duplicatas antes de limpar o texto.
- **Saiba a hora de parar:** limpeza nunca fica perfeita.

### Duplicatas

**O que é:** linhas repetidas segundo algum critério. No dataset da Olist,
322 avaliações tinham o `review_id` repetido.

**No código:** `.duplicated().sum()` conta; `.drop_duplicates(subset=["review_id"], keep="first")` remove.

### Estimativa por amostra

**O que é:** medir o tempo numa amostra e projetar para o total, antes de
processar um volume grande.

**Exemplo:** 2.000 avaliações em 12 s → 41.431 em cerca de 4 minutos.

---

## Python, numpy e pandas

### Lista × array

**O que é:** a lista do Python é uma caixa genérica; o array do numpy é feito
para contas, posição por posição.

**Exemplo:** `[1, 2] + [2, 1]` dá `[1, 2, 2, 1]` (junta); com arrays dá `[3 3]` (soma).

### Objeto × dicionário

**O que é:** um dicionário usa chave (`produto["preco"]`); um objeto usa
atributo (`produto.preco`). No `print`, o dicionário aparece com `{}` e
dois-pontos; o objeto aparece como `campo=valor`. A resposta do Ollama é um
objeto que aceita as duas formas.

### Parênteses chamam a função

**O que é:** `.duplicated` é a função; `.duplicated()` é o resultado dela.

**Erro típico:** `'function' object has no attribute 'sum'`

### axis, keepdims e broadcasting

**O que é:** `axis=1` calcula ao longo de cada linha (uma norma por vetor).
`keepdims=True` mantém o resultado em formato de coluna `(n, 1)`, para que a
divisão aconteça linha a linha (broadcasting).

### argsort

**O que é:** devolve os **índices** em ordem crescente de valor, e não os
valores.

**Exemplo:** `scores = [0.2, 0.9, 0.5]` → `argsort` = `[0, 2, 1]` →
`[::-1]` = `[1, 2, 0]` → `[:2]` = `[1, 2]`.

### zip

**O que é:** percorre várias sequências em paralelo, como um zíper: a cada
volta, entrega um item de cada uma, na mesma posição.

**No código:**
```python
for nota, review_id, texto, vetor in zip(lote["review_score"], lote["review_id"], lote["texto_limpo"], vetores):
```

### Processamento em lotes

**O que é:** em vez de processar tudo de uma vez, dividir em pedaços.
Mandar 100 textos por chamada ao modelo é mais rápido do que 1 por vez, e
mais seguro do que 41 mil de uma vez.

**No código:** `for i in range(0, len(df), 100):` com `df.iloc[i:i + 100]`

### `if __name__ == "__main__"`

**O que é:** ao importar um arquivo, o Python executa tudo que está solto
nele. O código dentro desse bloco só roda quando o arquivo é executado
diretamente, e não quando é importado. Assim o mesmo arquivo serve de módulo e
de script de teste.

### Ler um traceback

**O que é:** leia de baixo para cima. A última linha diz **o quê** aconteceu
(`KeyError`, `ValueError`...); as de cima dizem **onde**, e os `^^^` apontam o
trecho exato.

### Testes que podem falhar

**O que é:** um teste só tem valor se daria resultado diferente quando a
operação falhasse.

**Exemplo:** depois de remover duplicatas, `.nunique()` não prova nada (já
dava 41.431 antes); `.duplicated().sum()` dando 0 prova (antes dava 322).