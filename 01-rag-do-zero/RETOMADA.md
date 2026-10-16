# Retomada — 01-rag-do-zero

Última sessão: 16/10/2026

## Onde parei

A **classificação das 41.431 avaliações está completa e analisada**, com nove
categorias de problema e o sentimento de cada comentário. README (itens 13 a
15) e GLOSSARIO (seções novas: Avaliação e Infraestrutura) estão atualizados
com tudo o que foi feito e aprendido.

Antes de seguir:

- [ ] **Commit** de tudo: `classificacao.py`, `schema_classificacao.sql`,
  `migracao_frete_prazo.sql`, `13_medir_paralelo.py`, `README.md`,
  `GLOSSARIO.md`, `RETOMADA.md`
- [ ] *(Opcional)* Atualizar no README a tabela nota × reclamação (item 14) com
  os números finais: ela usa os da primeira classificação, antes de reclassificar
  as ~3.800 linhas de atendimento e `outro`.

```sql
SELECT nota, sentimento, count(*)
FROM avaliacoes GROUP BY nota, sentimento ORDER BY nota, sentimento;
```

## Próximo passo

**Combinar a busca semântica com o filtro por problema**, no
`10_buscar_avaliacoes.py`. É juntar os dois mundos: a busca acha comentários
**parecidos** com uma frase, e o filtro garante **sobre o que** eles reclamam.

Ideia de consulta:

```sql
SELECT 1 - (embedding <=> %(q)s) AS score, nota, comment
FROM avaliacoes
WHERE 'problema_na_entrega' = ANY(problemas)
ORDER BY embedding <=> %(q)s
LIMIT %(k)s
```

Perguntas para investigar:

- A busca com filtro por problema traz resultados melhores do que o filtro
  `nota <= 2`? (O item 14 mostrou que a nota perde 35% das reclamações.)
- O índice HNSW continua sendo usado quando há um `WHERE` junto? Conferir com
  `EXPLAIN ANALYZE` (lembrar do item 12: o planejador pode preferir o
  `Seq Scan`).
- Faz sentido o filtro ser extraído da pergunta pelo LLM, como no self-query do
  catálogo (`extrair_filtros`)? Ex.: "clientes reclamando de frete" →
  `problema = 'frete_caro'`.

## Estado do ambiente

- **Modelo da classificação:** `qwen3.8:27b` com `think=False`. Teste:
  `python classificacao.py` deve dar 15/16 (o ✗ é o "Produto ce qualidade",
  falha conhecida).
- **Ollama:** configuração original, **sem** `OLLAMA_NUM_PARALLEL` (revertido;
  ver README item 15). O `num_ctx: 4096` na `classificar` pode ficar.
- **Banco:** `problemas TEXT[]` com `CHECK` das nove categorias, `sentimento`,
  `modelo_classificacao`. Nenhuma linha com `sentimento IS NULL`.

## Lembretes práticos

- Subir o banco e o Ollama: `docker start pgvector ollama`
- Ativar o ambiente virtual: `source ~/.venvs/rag/bin/activate` (prompt com `(rag)`)
- Entrar no psql: `docker exec -it pgvector psql -U rag -d rag`
- `rag=#` é o psql; `(rag) mfsever@...$` é o bash. `\q` sai do psql.
- Comandos com `\` usam o resto da linha: o SQL vai na linha seguinte.
- Colunas: `nota`, `comment`, `problemas`, `sentimento` (não é `score`).
  Ver com `\d avaliacoes`.
- Procurar dentro do array: `'frete_caro' = ANY(problemas)`; contar por
  categoria: `unnest(problemas)`.
- Conferir se uma edição foi salva: `grep -n "trecho" arquivo.py`
- Script longo: `nohup python -u script.py > arquivo.log 2>&1 &`