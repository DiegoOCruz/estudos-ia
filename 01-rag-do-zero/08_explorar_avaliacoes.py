import pandas as pd

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv('dados/olist_order_reviews_dataset.csv')
#print(df)
com_texto = df[df["review_comment_message"].notna()].copy()
#print(com_texto['review_score'].value_counts())
#print(com_texto['review_comment_message'].sample(10))
# for comments in com_texto['review_comment_message'].sample(10):
#     print(comments)
#print(com_texto['review_comment_message'].str.len().groupby(com_texto['review_score']).mean())
REGEX = [r'[A-Z]{2}\s?\d{3}\s?\d{3}\s?\d{3}\s?BR']

# print(com_texto['review_comment_message'].str.contains(REGEX[0], regex=True).sum())
# print(com_texto['review_comment_message'].str.contains(r'\r|\n', regex=True).sum())

com_texto['texto_limpo'] = com_texto['review_comment_message'].str.replace(r'\r|\n', ' ', regex=True).str.replace(REGEX[0], '[RASTREIO]', regex=True).str.strip()
# print(com_texto.columns)
# tinha_rastreio = com_texto[com_texto["review_comment_message"].str.contains(REGEX[0], regex=True)]

# for original, limpo in zip(tinha_rastreio["review_comment_message"], tinha_rastreio["texto_limpo"]):
#     print("ANTES: ", original)
#     print("DEPOIS:", limpo)
#     print()