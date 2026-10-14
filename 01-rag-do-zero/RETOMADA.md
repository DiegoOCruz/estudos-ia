# Retomada — 01-rag-do-zero

Última sessão: 14/10/2026

## Onde parei

Comecei o primeiro item dos "Próximos passos" do README: **classificar cada
comentário da Olist com o LLM**, para filtrar pelo conteúdo e não pela nota.

Arquivo: `11_classificar.py` (funcionando, testado com 7 exemplos fixos).

## O que já está pronto

- **Classe `Classificacao` (Pydantic)** com dois campos:
  - `problemas`: lista (pode ser vazia) de categorias permitidas por `Literal`:
    `atraso_ou_nao_recebido`, `problema_na_entrega`, `pedido_incompleto`,
    `produto_errado_ou_diferente`, `defeito_ou_qualidade`,
    `atendimento_ou_financeiro`, `outro`
  - `sentimento`: `positivo`, `negativo`, `misto` ou `neutro`
- **`PROMPT_SISTEMA`** com a definição de cada categoria e de cada sentimento.
- **Few-shot:** 4 exemplos (`EXEMPLOS`) entregues como conversa falsa
  (user → assistant) pela função `montar_mensagens`.
- **`classificar(texto)`**: `qwen2.5:14b`, temperatura 0, saída estruturada.
  Só o texto vai para o modelo, a nota não.

## Decisões tomadas (e por quê)

- **Categorias tiradas dos dados:** li 30 comentários sorteados (nota ≤ 3)
  antes de definir as categorias.
- **Lista, e não categoria única:** um comentário pode citar vários problemas
  (as cortinas: pedido incompleto e produto errado).
- **Atraso e não recebido juntos:** na hora em que o cliente escreve, não dá
  para saber se o produto vai chegar atrasado ou nunca.
- **Sentimento separado dos problemas:** "do que fala" × "como se sente".
  Comentário positivo fica com a lista vazia.
- **`problemas` antes de `sentimento`:** o LLM escreve da esquerda para a
  direita; o sentimento sai coerente com os problemas já listados.
- **A nota não vai para o modelo:** para detectar divergências, como
  "PRODUTO ENTREGUE NO PRAZO COM QUALIDADE" com nota 1.
- **`outro` existe de propósito:** evita forçar um problema na categoria
  errada. Contar os `outro` depois mostra se falta categoria.

## O que aprendi

- **Mudar a definição não resolveu e piorou outro caso.** "Ainda não está com
  ele" fez "Ainda não posso avaliar... quando chegar" virar atraso: o modelo
  seguiu a definição ao pé da letra. Por isso o teste roda os 7 casos, não só
  os que eu queria consertar.
- **Uma mudança de cada vez:** voltei a definição original antes de testar o
  few-shot, para saber o que resolveu.
- **Few-shot funcionou onde a definição falhou:** resolveu "tive que buscar no
  correio" (problema na entrega, não atraso).
- **Exemplos do prompt ≠ exemplos do teste:** senão o modelo só copia a
  resposta e o teste não prova nada.
- **Exemplos dos dois lados da fronteira:** incluí um atraso de verdade, para
  o modelo não aprender a evitar a categoria.

## Limitação conhecida

"Muuuuito demorado, apesar de entregue no prazo!" continua classificado como
`atraso_ou_nao_recebido` (deveria ser `outro`). A palavra "demorado" pesa mais
do que "no prazo". Aceito: é raro (1 em 30) e o erro vai para uma categoria
vizinha. A contagem de atrasos fica um pouco inflada. **Registrar no README
quando o passo terminar.**

## Próximo passo

**Medir o tempo antes de classificar tudo:**

1. Pegar uma amostra de comentários do banco (não os 7 fixos)
2. Classificar e cronometrar
3. Regra de três para as 41.431 (o LLM é bem mais lento que o `bge-m3`;
   pode dar horas)
4. Decidir: rodar tudo de noite, só uma parte, ou um modelo menor

Depois disso: coluna(s) nova(s) na tabela `avaliacoes`, classificação em
lotes, e contar quantos caíram em `outro`.

## Lembretes práticos

- Subir o banco: `docker start pgvector`
- Entrar no psql: `docker exec -it pgvector psql -U rag -d rag`
  (ou, já dentro do container, `psql -U rag -d rag`)
- `rag=#` é o psql (SQL e comandos com `\`); `root@...#` é o bash.
  `\q` sai do psql.
- Ver as colunas de uma tabela: `\d avaliacoes`
  (colunas: `nota`, `comment`; não é `score`)
- Leitura mais fácil de textos longos no psql: `\x`