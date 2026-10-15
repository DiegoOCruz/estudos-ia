import time
import psycopg
from classificacao import classificar
from comum import CONEXAO, LLM

N = 50
TOTAL = 41_431

with psycopg.connect(CONEXAO) as conn:
    linhas = conn.execute(
        "SELECT nota, comment FROM avaliacoes ORDER BY random() LIMIT %s",
        (N,),
    ).fetchall()

classificar("aquecimento")

inicio = time.perf_counter()
resultados = [(nota, texto, classificar(texto)) for nota, texto in linhas]
duracao = time.perf_counter() - inicio
print(f"LLM: {LLM}")
por_comentario = duracao / N
print(f"{N} comentários em {duracao:.1f} s ({por_comentario:.2f} s cada)")
print(f"Projeção para {TOTAL:,}: {por_comentario * TOTAL / 3600:.1f} horas\n")

for nota, texto, c in resultados[:15]:
    print(f"nota {nota} | {c.sentimento:8} {c.problemas}")
    print(f"         {texto[:80]}\n")
