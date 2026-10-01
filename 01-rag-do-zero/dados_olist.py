import pandas as pd


REGEX = [r"[A-Z]{2}\s?\d{3}\s?\d{3}\s?\d{3}\s?BR"]

def limpar_avaliacoes(df):
    avaliacoes = (
        df[df["review_comment_message"].notna()]
        .copy()
        .drop_duplicates(subset=["review_id"], keep="first")
    )
    avaliacoes["texto_limpo"] = (
        avaliacoes["review_comment_message"]
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(REGEX[0], "[RASTREIO]", regex=True)
        .str.strip()
    )
    return avaliacoes[
        ["review_id", "review_score", "texto_limpo"]
    ]

if __name__ == "__main__":
    df = pd.read_csv("dados/olist_order_reviews_dataset.csv")
    print(limpar_avaliacoes(df).head(10))