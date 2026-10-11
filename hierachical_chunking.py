import pandas as pd
import numpy as np
import re
import pickle
from sentence_transformers import SentenceTransformer

print("🔄 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("📖 Loading Excel dataset...")
df = pd.read_excel("tableofcontent_final_merged.xlsx")

chunked_docs = []
metadata = []

print("📚 Performing TOC-aware hierarchical chunking...")

toc_pattern = re.compile(r'(\d+(?:\.\d+)*)\s*[.)-]?\s*([^0-9]+)')

for _, row in df.iterrows():
    author = str(row["Author"]).strip()
    book_title = str(row["Title"]).strip()
    toc_text = str(row["Table of Contents"]).strip()

    # =================================================
    # 📘 BOOK-LEVEL CHUNK (CRITICAL FIX)
    # =================================================
    book_chunk = (
        f"Book Title: {book_title}\n"
        f"Author: {author}\n"
        f"Description: This volume is part of the Mahabharata."
    )

    chunked_docs.append(book_chunk)
    metadata.append({
        "author": author,
        "book_title": book_title,
        "type": "book"
    })

    # -------------------------------------------------
    # Skip TOC chunking if TOC is empty
    # -------------------------------------------------
    if not toc_text or toc_text.lower() == "nan":
        continue

    # Normalize TOC (Excel-safe)
    toc_text = toc_text.replace("\n", " ").replace("\r", " ")

    matches = toc_pattern.findall(toc_text)

    # =================================================
    # 📑 TOC-LEVEL CHUNKS
    # =================================================
    for number, content in matches:
        content = content.strip()
        if len(content) < 5:
            continue

        toc_chunk = (
            f"Book Title: {book_title}\n"
            f"Author: {author}\n"
            f"Table of Contents Entry: {content}"
        )

        chunked_docs.append(toc_chunk)
        metadata.append({
            "author": author,
            "book_title": book_title,
            "section_number": number,
            "type": "toc"
        })

print(f"🔢 Total chunks created (book + TOC): {len(chunked_docs)}")

print("🔍 Encoding chunks...")
embeddings = embedder.encode(
    chunked_docs,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("💾 Saving embeddings and metadata...")
np.savez_compressed("doc_embeddings.npz", embeddings=embeddings)

with open("chunked_data.pkl", "wb") as f:
    pickle.dump((chunked_docs, metadata), f)

print("✅ Chunking completed successfully.")
