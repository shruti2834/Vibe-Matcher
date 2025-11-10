import os, time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

client = OpenAI()

def get_embedding(text, model="text-embedding-ada-002"):
    """Get embedding vector for a piece of text."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def vibe_match(query, df, emb_matrix, top_k=3, threshold=0.7):
    """Return top-k matching products for a given query."""
    q_emb = get_embedding(query)
    sims = cosine_similarity([q_emb], emb_matrix).flatten()
    top_idx = np.argsort(-sims)[:top_k]

    results = []
    for i in top_idx:
        results.append({
            "name": df.loc[i, "name"],
            "desc": df.loc[i, "desc"],
            "vibes": df.loc[i, "vibes"],
            "score": float(sims[i])
        })

    best_score = results[0]["score"]
    fallback = None if best_score >= threshold else f"No strong match for '{query}' — try a broader vibe."
    return results, best_score, fallback
