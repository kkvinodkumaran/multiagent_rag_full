import os
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

class RAGStore:
    def __init__(self, persist_dir: str, embed_model_name: str):
        os.makedirs(persist_dir, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
        self.db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=120
        )

    def add_texts(self, docs: List[str], metadata: dict | None = None):
        chunks = []
        metas = []
        for d in docs:
            for c in self.splitter.split_text(d):
                chunks.append(c)
                metas.append(metadata or {})
        if chunks:
            self.db.add_texts(chunks, metadatas=metas)

    def retrieve(self, query: str, k: int = 6) -> List[str]:
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        return [d.page_content for d in docs]
