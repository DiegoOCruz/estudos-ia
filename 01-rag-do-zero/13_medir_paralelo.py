import time
from concurrent.futures import ThreadPoolExecutor

import psycopg

from classificacao import classificar

from comum import CONEXAO, LLM

N = 60


with psycopg.connect(CONEXAO) as conn:
    conn.execute("SELECT setseed(0.42)")
    linhas = conn.execute(
        "SELECT comment FROM avaliacoes ORDER BY random() LIMIT %s", (N,)
    ).fetchall()
textos = [texto for (texto,) in linhas]


def medir(trabalhadores: int):
    inicio = time.perf_counter()
    with ThreadPoolExecutor(max_workers=trabalhadores) as executor:
        resultados = list(executor.map(lambda t: classificar(t, LLM), textos))
    return time.perf_counter() - inicio, resultados


classificar("aquecimento", LLM)

tempo_1, resultados_1 = medir(1)
print(f"1 por vez:     {tempo_1:5.1f} s ({tempo_1 / N:.2f} s cada)")

tempo_4, resultados_4 = medir(4)
print(f"4 ao mesmo tempo: {tempo_4:5.1f} s ({tempo_4 / N:.2f} s cada)")

print(f"Ganho: {tempo_1 / tempo_4:.1f}x")

diferentes = sum(a != b for a, b in zip(resultados_1, resultados_4))
print(f"Classificações diferentes entre as duas rodadas: {diferentes}/{N}")