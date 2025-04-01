import pandas as pd
import numpy as np
from ast import literal_eval
from embedding import embedding_query
from BAAI import embedding



def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load the dataset and convert embeddings
datafile_path = "BAAI.csv"
df = pd.read_csv(datafile_path, index_col=0)
df.to_csv("just_import")
df["Embedding"] = df.Embedding.apply(literal_eval).apply(np.array)
df.to_csv("import_csv")


# Function to search through the reviews
def search_reviews(df, text, n=3, pprint=True):
    #product_embedding = embedding_query(text)
    product_embedding = embedding(text)[0]

    df["Similarity"] = df.Embedding.apply(lambda x: cosine_similarity(x, product_embedding))
    df.to_csv("after_similarity")
    results = df.sort_values("Similarity", ascending=False).head(n)
    
    '''if pprint:
        for r in results:
            print(r[:200])
            print()
    return results'''
    return results

# Example searches
results = search_reviews(df, "What is the relationship between my job and meidcare?", n=3)
print(results)
