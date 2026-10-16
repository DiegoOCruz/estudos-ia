import time

import psycopg

from classificacao import classificar
from comum import CONEXAO, LLM
LOTE = 100


with psycopg.connect(CONEXAO) as conn:
    restantes = conn.execute(
        "SELECT count(*) FROM avaliacoes WHERE sentimento IS NULL"
    ).fetchone()[0]
    print(f"{restantes:,} avaliações por classificar")

    classificar("aquecimento", LLM)

    inicio = time.perf_counter()
    feitas = 0

    while True:
        lote = conn.execute(
            "SELECT id, comment FROM avaliacoes WHERE sentimento IS NULL ORDER BY id LIMIT %s",
            (LOTE,),
        ).fetchall()
        if not lote:
            break

        for id_, texto in lote:
            c = classificar(texto, LLM)
            conn.execute(
                "UPDATE avaliacoes SET problemas = %s, sentimento = %s, modelo_classificacao = %s WHERE id = %s",
                (c.problemas, c.sentimento, LLM, id_),
            )
        conn.commit()

        feitas += len(lote)
        decorrido = time.perf_counter() - inicio
        falta = (restantes - feitas) * decorrido / feitas
        print(f"{feitas:,}/{restantes:,}  ({decorrido / 60:.0f} min, faltam ~{falta / 60:.0f} min)")