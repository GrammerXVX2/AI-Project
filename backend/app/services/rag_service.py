import asyncio

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from backend.app.config import RAG_DB_DIR, RAG_THRESHOLD


class RAGService:
    def __init__(self):
        print("[RAG] Initializing RAG Service...")
        # Initialize on CPU for now as per original code
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.client = PersistentClient(path=str(RAG_DB_DIR))
        self.collection = self.client.get_or_create_collection("codebase")
        self.threshold = RAG_THRESHOLD

    def search(self, query: str, n_results: int = 1):
        try:
            query_emb = self.embedder.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_emb, n_results=n_results, include=["documents", "distances"]
            )

            if not results["documents"] or not results["documents"][0]:
                return None

            # Безопасное получение дистанции (через переменную, чтобы успокоить линтер)
            distances = results.get("distances")
            if distances and distances[0]:
                distance = distances[0][0]
                print(f"[RAG DEBUG] Dist: {distance}")

                if distance > self.threshold:
                    return None
            else:
                print("[RAG WARNING] No distance data returned, skipping threshold check.")

            return results["documents"][0][0]
        except Exception as e:
            print(f"[RAG ERROR] Search failed: {e}")
            return None

    async def asearch(self, query: str, n_results: int = 1):
        """Async wrapper for search"""
        return await asyncio.to_thread(self.search, query, n_results)


# Global instance
rag_service = RAGService()
