from agno.knowledge.website import WebsiteKnowledgeBase
from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.json import JSONKnowledgeBase
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.pgvector import PgVector
from agno.embedder.ollama import OllamaEmbedder
import asyncio
from threading import Thread


class KnowledgeWorkerThread(Thread):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.knowledge_base = None
        self.error = None

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.knowledge_base = loop.run_until_complete(
                self.getJSONKnowledgeBase(self.path)
            )
        except Exception as e:
            self.error = e
        finally:
            loop.close()

    @staticmethod
    async def getWebsiteKnowledgeBase(url: str) -> WebsiteKnowledgeBase:
        knowledge_base = WebsiteKnowledgeBase(
            urls=[url],
            max_links=10,
            vector_db=PgVector(
                table_name="website_documents",
                db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
                embedder=OllamaEmbedder(id="llama3.2", dimensions=3072),
            ),
        )
        await knowledge_base.aload()
        return knowledge_base

    @staticmethod
    async def getTextKnowledgeBase(path: str) -> TextKnowledgeBase:
        knowledge_base = TextKnowledgeBase(
            path=path,
            vector_db=PgVector(
                table_name="text_embeddings",
                db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
                embedder=OllamaEmbedder(id="llama3.2", dimensions=3072),
            ),
        )
        await knowledge_base.aload()
        return knowledge_base

    async def getJSONKnowledgeBase(self, path: str) -> JSONKnowledgeBase:
        knowledge_base = JSONKnowledgeBase(
            path=path,
            vector_db=PgVector(
                table_name="json_embeddings",
                db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
                embedder=OllamaEmbedder(id="llama3.2", dimensions=3072),
            ),
        )
        await knowledge_base.aload()
        return knowledge_base

    @staticmethod
    async def getCSVKnowledgeBase(path: str) -> CSVKnowledgeBase:
        knowledge_base = CSVKnowledgeBase(
            path=path,
            vector_db=PgVector(
                table_name="csv_embeddings",
                db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
                embedder=OllamaEmbedder(id="llama3.2", dimensions=3072),
            ),
        )
        await knowledge_base.aload()
        return knowledge_base
