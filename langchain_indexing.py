import pandas as pd
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def main():
    print("🔄 Loading embedding model (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("📖 Loading Excel dataset...")
    try:
        df = pd.read_excel("tableofcontent_final_merged.xlsx")
    except FileNotFoundError:
        print("❌ Error: 'tableofcontent_final_merged.xlsx' not found.")
        return

    documents = []
    
    print("📚 Performing TOC-aware hierarchical chunking...")
    toc_pattern = re.compile(r'(\d+(?:\.\d+)*)\s*[.)-]?\s*([^0-9]+)')

    for _, row in df.iterrows():
        author = str(row.get("Author", "")).strip()
        book_title = str(row.get("Title", "")).strip()
        toc_text = str(row.get("Table of Contents", "")).strip()

        # =================================================
        # 📘 BOOK-LEVEL CHUNK
        # =================================================
        book_content = (
            f"Book Title: {book_title}\n"
            f"Author: {author}\n"
            f"Description: This volume is part of the library collection."
        )
        
        book_metadata = {
            "author": author,
            "book_title": book_title,
            "type": "book"
        }
        
        documents.append(Document(page_content=book_content, metadata=book_metadata))

        # -------------------------------------------------
        # Skip TOC chunking if TOC is empty
        # -------------------------------------------------
        if not toc_text or toc_text.lower() == "nan":
            continue

        # Normalize TOC
        toc_text = toc_text.replace("\n", " ").replace("\r", " ")
        matches = toc_pattern.findall(toc_text)

        # =================================================
        # 📑 TOC-LEVEL CHUNKS
        # =================================================
        for number, content in matches:
            content = content.strip()
            if len(content) < 5:
                continue

            toc_chunk_content = (
                f"Book Title: {book_title}\n"
                f"Author: {author}\n"
                f"Table of Contents Entry: {content}"
            )
            
            toc_metadata = {
                "author": author,
                "book_title": book_title,
                "section_number": number,
                "type": "toc"
            }
            
            documents.append(Document(page_content=toc_chunk_content, metadata=toc_metadata))

    print(f"🔢 Total chunks created: {len(documents)}")
    
    print("🔍 Creating FAISS Vector Store (this may take a while)...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print("💾 Saving FAISS index locally...")
    vectorstore.save_local("faiss_index_langchain")
    
    print("✅ Indexing completed successfully. Saved to 'faiss_index_langchain'.")

if __name__ == "__main__":
    main()
