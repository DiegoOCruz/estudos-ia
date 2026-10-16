# Glossário

Conceitos estudados no repositório, cada um com uma explicação curta, um
exemplo pequeno e onde ele aparece no código.

- [Matemática](#matemática)
- [Embeddings](#embeddings)
- [RAG](#rag)
- [Banco de dados](#banco-de-dados)
- [LLM](#llm)
- [Avaliação](#avaliação)
- [Dados](#dados)
- [Python, numpy e pandas](#python-numpy-e-pandas)
- [Infraestrutura](#infraestrutura)

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

**No código:** o campo `prompt_eval_count` da resposta do `ollama.embed` (e do
`ollama.chat`).

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
inteira a partir de 3 documentos); "disponível" (o banco nem tem estoque). Na
classificação: "Como realizo a troca?" virava problema de atendimento, porque
o cliente mencionou contato com a loja.

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
indexação. Rodar duas vezes não duplica nada. Na classificação, a mesma ideia
de outro jeito: o script só processa `WHERE sentimento IS NULL`, então rodar
de novo continua de onde parou.

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
"desconhecido", e não "falso". Para exigir o valor, use `NOT NULL` junto. Às
vezes isso é o que se quer: nas colunas de classificação, o `CHECK` valida os
valores preenchidos e deixa passar as linhas ainda não classificadas.

**No código:** `score SMALLINT NOT NULL CHECK (score BETWEEN 1 AND 5)`

### Tipos de dados

**O que é:** escolher o tipo certo descreve o dado e evita erros.
- `NUMERIC(10, 2)` para dinheiro: exato, sem erros de arredondamento.
- `SMALLINT` para números inteiros pequenos, como uma nota de 1 a 5.
- `TEXT` para códigos de barras: podem começar com zero e nunca entram em conta.

**Cuidado:** `0.1 + 0.2` em ponto flutuante dá `0.30000000000000004`. Em
Python, dinheiro vai como `Decimal("149.90")`, e `Decimal(str(x))` quando vier
de um float.

### Array (`TEXT[]`)

**O que é:** uma coluna que guarda uma lista. É o equivalente, no banco, de um
`list` do Python. O psycopg converte uma lista Python para array sozinho.

**Operadores:**
- `'x' = ANY(coluna)`: o valor está dentro do array?
- `coluna <@ ARRAY[...]`: todos os itens do array estão contidos na lista?
- `unnest(coluna)`: "abre" o array em linhas, um item por linha, para contar
  com `GROUP BY`. Quem tem dois itens conta duas vezes.

**No código:**
```sql
problemas TEXT[] CHECK (problemas <@ ARRAY['atraso_ou_nao_recebido', ...]::TEXT[])

SELECT unnest(problemas) AS problema, count(*) FROM avaliacoes GROUP BY problema;
```

### NULL × vazio

**O que é:** `NULL` quer dizer "não se sabe" ou "ainda não foi feito"; um valor
vazio (`{}`, `''`) quer dizer "foi feito, e não há nada". São coisas
diferentes, e o banco distingue as duas.

**Exemplo:** `problemas` NULL = avaliação ainda não classificada; `{}` =
classificada, e o modelo não achou problema nenhum.

**Pegadinha:** um array NULL não contém nada: `'x' = ANY(NULL)` não encontra a
linha.

### ALTER TABLE e migração

**O que é:** `ALTER TABLE` muda uma tabela que já existe (acrescenta colunas,
troca restrições) sem apagar os dados. Uma **migração** é um arquivo que leva
um banco existente de uma versão para a próxima; o `schema` cria do zero. Os
dois caminhos precisam chegar ao mesmo banco.

**Restrição não se edita:** remove (`DROP CONSTRAINT`) e cria de novo
(`ADD CONSTRAINT`), pelo nome que aparece no `\d tabela`.

**No código:** `schema_classificacao.sql` (colunas novas) e
`migracao_frete_prazo.sql` (restrição com as categorias novas).

### Transação (BEGIN / COMMIT)

**O que é:** um conjunto de alterações que acontece inteiro ou não acontece.
Até o `COMMIT`, as alterações ficam pendentes; se algo falhar, nada é gravado.

**Exemplo:** na migração, `BEGIN; DROP ...; ADD ...; COMMIT;` evita o instante
em que a coluna ficaria sem regra. No script, `conn.commit()` a cada lote:
uma interrupção perde só o lote em andamento, inteiro, nunca pela metade.

### Sorteio reproduzível (`setseed`)

**O que é:** o `ORDER BY random()` sorteia linhas diferentes a cada execução.
O `setseed` fixa o ponto de partida, e o sorteio sai sempre igual. Serve para
comparar duas coisas (dois modelos, duas configurações) nos mesmos dados.

**No código:** `conn.execute("SELECT setseed(0.42)")` antes da consulta, na
mesma conexão.

### ILIKE

**O que é:** busca por texto exato, sem diferenciar maiúsculas e minúsculas.
`%` significa "qualquer coisa".

**Exemplo:** `WHERE comment ILIKE 'Recebi o carregador%'` acha o comentário
inteiro a partir do começo dele.

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

**Cuidado:** zero não garante resultado 100% igual (veja "Não determinismo").

### Saída estruturada

**O que é:** forçar o LLM a responder num formato exato, descrito por um
schema JSON. O Pydantic descreve o formato como uma classe e valida a resposta.
O `Literal` limita um campo a valores permitidos, e o modelo não consegue
inventar outro. A `description` de um `Field` vai junto no schema e funciona
como uma instrução grudada no campo.

**No código:**
```python
class Filtros(BaseModel):
    preco_max: Optional[float] = None
    tipo: Optional[Literal["produto", "servico", "politica"]] = None
```

**Ordem dos campos importa:** o LLM escreve da esquerda para a direita, e o que
já escreveu influencia o resto. Em `Classificacao`, `problemas` vem antes de
`sentimento`.

### Few-shot

**O que é:** em vez de só **descrever** a tarefa, **mostrar** exemplos
resolvidos. O modelo costuma aprender mais com um exemplo do que com uma
definição.

**Como:** uma conversa falsa no começo das mensagens: o `user` manda um texto
e o `assistant` responde com o JSON certo, várias vezes; só então vem o texto
real.

**Cuidados:**
- Os exemplos **não podem ser os casos do teste**: o modelo copiaria a
  resposta, e o teste não provaria nada.
- Mostrar os **dois lados da fronteira**: se todos os exemplos fossem "isto não
  é atraso", o modelo aprenderia a evitar a categoria.
- Exemplo e definição precisam concordar: quando discordam, o exemplo costuma
  vencer.

**No código:** `EXEMPLOS` e `montar_mensagens` no `classificacao.py`.

### O modelo segue a definição ao pé da letra

**O que é:** o modelo aplica exatamente o que a definição diz, não o que você
quis dizer. E é fisgado por palavras que batem com a definição.

**Exemplos:** "ainda não está com ele" fez "quando chegar farei uma avaliação"
virar atraso (o cliente não reclamava). "Não recebi, tive que buscar no
correio" virava atraso por causa do "Não recebi".

### O prompt funciona como um todo

**O que é:** mudar uma linha muda o contexto inteiro que o modelo lê, e o
efeito pode aparecer longe de onde você mexeu.

**Exemplo:** ajustar a definição de atendimento fez o 14B acertar um caso de
prazo que nada tinha a ver com atendimento. Acrescentar duas categorias fez
ele errar casos que antes acertava.

### Modelo de raciocínio

**O que é:** modelos (como a família Qwen3) que escrevem um texto escondido,
"pensando", antes da resposta. Você não vê esse texto, mas ele é gerado token
por token e custa tempo. Para tarefas simples e bem definidas, pode não ajudar.

**Exemplo:** o 27B levava 3,10 s por comentário; com o raciocínio desligado,
0,70 s.

**No código:** `ollama.chat(..., think=False)`

### Não determinismo com temperatura zero

**O que é:** mesmo com temperatura zero, o mesmo texto pode dar um resultado
diferente. A GPU soma milhares de números em ponto flutuante, e a ordem das
somas pode variar, como no `0.1 + 0.2`. Quase sempre não muda nada; nos casos
de fronteira, quando o modelo está quase empatado entre duas respostas, vira.

**Exemplo:** 1 em 60 classificações mudou entre duas execuções idênticas.

### Contexto (`num_ctx`)

**O que é:** o espaço máximo de tokens que o modelo reserva para uma conversa.
A reserva ocupa memória da placa, mesmo que o prompt seja curto. Enquanto o
prompt couber, o tamanho do contexto não muda a resposta.

**Exemplo:** o prompt da classificação usa ~1.000 tokens; reservar 32.768 é
desperdício. Com 4 pedidos simultâneos, a reserva quadruplica.

**No código:** `options={"temperature": 0, "num_ctx": 4096}`

---

## Avaliação

### Teste com resposta esperada

**O que é:** uma lista de casos, cada um com a resposta certa, conferida
automaticamente (✓/✗ e placar). Conferir de olho comparando com anotações
fica sujeito a erro quando os casos passam de uns poucos.

**Como cresce:** cada falha nova encontrada em dados reais vira um caso, que
impede a volta dela. Uma falha conhecida fica no teste com a resposta
**certa**, aparecendo como ✗: colocar o que o modelo responde para "passar"
faria o teste mentir.

**No código:** a lista `casos` no `if __name__ == "__main__"` do
`classificacao.py`, comparando com `set()` para a ordem dos problemas não
importar.

### Linha de base

**O que é:** o resultado **antes** de uma mudança, medido nas mesmas condições.
Sem ela, não dá para dizer se a mudança melhorou.

**Exemplo:** rodar o teste com o prompt atual (7/10) antes de ajustar a
definição de atendimento (9/10).

### Uma mudança de cada vez

**O que é:** mudando duas coisas juntas (a definição e o modelo, por exemplo),
se o resultado melhorar, não dá para saber qual delas resolveu.

**Exemplo:** voltei a definição de atraso para a original antes de testar o
few-shot.

### Caso ambíguo no teste

**O que é:** um caso que admite duas respostas razoáveis não separa um modelo
bom de um ruim: só gera ruído no placar. Melhor retirar.

**Exemplo:** "veio apenas de microfibra" (pedido incompleto, ou produto errado,
ou os dois). O 27B levou ✗ por uma resposta defensável, e o placar empatou dois
modelos que não estavam empatados.

### Placar esconde informação

**O que é:** dois resultados iguais podem errar coisas diferentes. Olhe
**quais** casos falharam, não só quantos.

**Exemplo:** 14B e 27B com 7/10; um errava uma nuance de leitura e o outro não.

### Amostra descobre; não mede

**O que é:** uma amostra pequena serve para **descobrir** que algo existe, mas
não para **medir** quanto. Proporções em poucos casos variam muito.

**Exemplo:** 4 casos de frete caro em 15 sugeriam ~900 no total; a contagem
real deu 264.

### Regra de três (estatística)

**O que é:** se você observou **zero** erros em *n* casos, a taxa real de erro
provavelmente fica abaixo de **3/n** (com 95% de confiança).

**Exemplo:** 0 erros em 15 → abaixo de 20%. Para afirmar "abaixo de 5%", seriam
necessários uns 60 casos sem erro.

### Categoria larga é um "outro" disfarçado

**O que é:** uma categoria com definição ampla absorve o que não se encaixa em
lugar nenhum, e a categoria `outro` fica pequena por isso, não porque as
categorias sejam boas. Um `outro` pequeno é um bom sinal, mas não uma prova.

**Exemplo:** "atendimento **ou financeiro**" escondia as reclamações de frete
caro.

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

**Exemplo:** 2.000 avaliações em 12 s → 41.431 em cerca de 4 minutos. Na
classificação: 50 comentários em 35 s → cerca de 8 horas para todos.

**Cuidado:** a projeção oscila com a amostra (comentários curtos são mais
rápidos). Trate como ordem de grandeza, não precisão de minuto.

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

### Desempacotar tuplas

**O que é:** separar os itens de uma tupla em variáveis numa linha só. O número
de variáveis precisa bater com o de itens.

**Exemplos:** `for texto, resposta in EXEMPLOS:`; uma tupla de um item só leva
vírgula: `(N,)` e `for (texto,) in linhas`.

**Erro típico:** `too many values to unpack (expected 2)`: um item da lista
tinha mais partes do que as variáveis do `for`.

### List comprehension

**O que é:** um `for` compacto que já monta a lista.

**Exemplo:** `[classificar(t) for t in textos]`

### Conjunto (`set`)

**O que é:** uma coleção sem ordem e sem repetição. Serve para comparar listas
quando a ordem não importa: `['a', 'b']` e `['b', 'a']` são listas diferentes,
mas conjuntos iguais.

**No código:** `set(c.problemas) == set(problemas_esperados)`

### lambda

**O que é:** uma função de uma linha, sem nome. Útil para adaptar uma função
que recebe dois argumentos a um lugar que só passa um.

**Exemplo:** `lambda t: classificar(t, LLM)`

### Processamento em lotes

**O que é:** em vez de processar tudo de uma vez, dividir em pedaços.
Mandar 100 textos por chamada ao modelo é mais rápido do que 1 por vez, e
mais seguro do que 41 mil de uma vez.

**No código:** `for i in range(0, len(df), 100):` com `df.iloc[i:i + 100]`. Na
classificação, um `while True` que pergunta ao banco o que ainda falta
(`LIMIT 100`) e para quando a consulta volta vazia.

### Módulo × script

**O que é:** um módulo é feito para ser importado; um script, para ser
executado. Nome de módulo segue as regras de nome de variável, então **não
pode começar com número**: `from 11_classificar import ...` é erro de sintaxe.

**No código:** `classificacao.py` (módulo) e `11_medir_classificacao.py`
(script); o mesmo padrão de `dados_olist.py` e `08_explorar_avaliacoes.py`.

### `if __name__ == "__main__"`

**O que é:** ao importar um arquivo, o Python executa tudo que está solto
nele. O código dentro desse bloco só roda quando o arquivo é executado
diretamente, e não quando é importado. Assim o mesmo arquivo serve de módulo e
de script de teste.

### Threads (`ThreadPoolExecutor`)

**O que é:** um grupo de "trabalhadores" que executam chamadas ao mesmo tempo.
No Python, threads não aceleram contas pesadas, mas funcionam bem quando o
programa passa o tempo **esperando** (uma resposta de rede, do banco, do
Ollama).

**No código:** `executor.map(lambda t: classificar(t, LLM), textos)`, que
devolve os resultados na ordem original.

### Cronômetro (`perf_counter`)

**O que é:** o relógio do Python para medir intervalos: alta precisão e imune a
ajustes no relógio do sistema.

**Cuidado:** a primeira chamada a um modelo inclui carregá-lo na GPU. Faça uma
chamada de **aquecimento** fora do cronômetro.

**No código:** `inicio = time.perf_counter()` ... `time.perf_counter() - inicio`

### Ler um traceback

**O que é:** leia de baixo para cima. A última linha diz **o quê** aconteceu
(`KeyError`, `ValueError`...); as de cima dizem **onde**, e os `^^^` apontam o
trecho exato.

### Testes que podem falhar

**O que é:** um teste só tem valor se daria resultado diferente quando a
operação falhasse.

**Exemplo:** depois de remover duplicatas, `.nunique()` não prova nada (já
dava 41.431 antes); `.duplicated().sum()` dando 0 prova (antes dava 322). Na
classificação: interromper o script de propósito com Ctrl+C e conferir que
ficaram gravadas 400 linhas, um múltiplo de 100.

---

## Infraestrutura

### psql × bash

**O que é:** o prompt diz onde você está. `rag=#` é o psql (SQL e comandos com
`\`); `root@...#` ou `(rag) usuario@...$` é o bash (comandos do Linux).

**Pegadinhas:**
- `\q` sai do psql; `\d tabela` mostra as colunas; `\x` mostra um registro por
  bloco (bom para textos longos).
- Um comando com `\` usa o resto da linha como argumento: o SQL vai na linha
  seguinte.
- No bash, `<` lê de um arquivo: `WHERE nota <= 3` colado no bash procura um
  arquivo chamado `=`.

### Ambiente virtual

**O que é:** uma instalação de Python separada, com os pacotes do projeto. O
`(rag)` no prompt mostra que está ativo. Depois de reiniciar a máquina, todo
terminal novo começa sem ele.

**Erro típico:** `ModuleNotFoundError` ao rodar um script que funcionava ontem.

**No código:** `source ~/.venvs/rag/bin/activate`

### Rodar sem depender do terminal (`nohup`)

**O que é:** um script longo morre se o terminal fechar ou a conexão cair. O
`nohup` mantém o processo vivo.

**No código:**
```bash
nohup python -u script.py > arquivo.log 2>&1 &
tail -f arquivo.log        # acompanhar (Ctrl+C sai só do tail)
pgrep -af script.py        # está rodando?
pkill -f script.py         # parar
```
- `-u`: sem buffer, para o log aparecer na hora.
- `2>&1`: os erros vão para o mesmo log.
- `&`: em segundo plano.
- O log é gravado na pasta onde o comando foi disparado.

### Largura de banda da memória (GPU)

**O que é:** para gerar cada token, a placa lê o modelo inteiro da memória. A
velocidade de geração depende mais de quantos GB por segundo ela lê do que da
força de cálculo.

**Conta aproximada:** tokens/s ≈ largura de banda ÷ tamanho do modelo.

**Exemplo:** 17 GB (27B) contra 9 GB (14B) explicam ~2× de diferença de
velocidade, não 13×. O resto era raciocínio escondido.

### Ollama: comandos e configuração

**O que é:**
- `ollama ps`: modelos carregados. A coluna PROCESSOR mostra se o modelo coube
  na GPU (`100% GPU`) ou foi dividido com a CPU (muito mais lento).
- `ollama list`: modelos instalados. `ollama stop <modelo>`: libera a GPU.
- `OLLAMA_KEEP_ALIVE`: quanto tempo o modelo fica carregado depois do último
  uso.
- `OLLAMA_NUM_PARALLEL`: quantos pedidos simultâneos. Vale para o servidor
  inteiro, multiplica a memória de contexto e nem toda arquitetura suporta (o
  log diz). Hoje está desligado.
- `docker logs ollama`: o log diz o porquê das coisas, como "model architecture
  does not currently support parallel requests".

### Volume do Docker

**O que é:** uma área de armazenamento que vive fora do container. Apagar o
container não apaga o volume; um container novo montando o mesmo volume
enxerga os mesmos arquivos. Variáveis de ambiente só são definidas na criação
do container, então mudá-las exige apagar e recriar.

**No código:** `-v ollama:/root/.ollama` (os modelos) e
`-v pgdata:/var/lib/postgresql/data` (o banco).

### Depois de uma queda de energia

1. `docker ps`: se `pgvector` e `ollama` não aparecem, `docker start pgvector ollama`.
2. Ativar o ambiente virtual (o prompt deve mostrar `(rag)`).
3. Disparar o script de novo: ele continua de onde parou.