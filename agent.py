# agent.py
import os
import json
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from supabase import create_client, Client

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from settings import (
    OPENAI_API_KEY,
    SUPABASE_URL,
    SUPABASE_KEY,
    PERSIST_DIR,
    MODEL,
)

logger = logging.getLogger("uvicorn.error")

# --- Initialize Supabase Client ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Successfully connected to Supabase.")
except Exception as e:
    logger.error(f"Failed to connect to Supabase: {e}")
    supabase = None

# --- Agent Definition ---
class SheetRAGAgent:
    def __init__(self) -> None:
        self._index = None
        self._parser = SimpleNodeParser()
        os.makedirs(PERSIST_DIR, exist_ok=True)
        self._graph = self._build_graph()

    def fetch_supabase_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetches data from all specified tables in Supabase."""
        if not supabase:
            raise ConnectionError("Supabase client is not initialized.")
        
        data = {}
        tables_to_fetch = ["checklist", "delegation"]

        for table in tables_to_fetch:
            try:
                response = supabase.table(table).select("*").execute()
                if response.data:
                    data[table] = response.data
                else:
                    logger.warning(f"No data returned from table: {table}")
                    data[table] = []
            except Exception as e:
                logger.error(f"Error fetching from table {table}: {e}")
                data[table] = []
        return data

    def _docs_from_data(self, data: Dict[str, List[Dict[str, Any]]]) -> List[Document]:
        docs: List[Document] = []
        for sheet_name, rows in data.items():
            for i, row in enumerate(rows):
                # Ensure all values are strings for consistent processing
                clean_row = {k: str(v) for k, v in row.items()}
                docs.append(
                    Document(
                        text=json.dumps(clean_row, ensure_ascii=False),
                        metadata={**clean_row, "sheetName": sheet_name},
                        doc_id=f"{sheet_name}-row-{i+1}",
                    )
                )
        return docs

    def rebuild_index(self) -> None:
        logger.info("[Index] Building fresh index from Supabase data…")
        data = self.fetch_supabase_data()
        docs = self._docs_from_data(data)
        nodes = self._parser.get_nodes_from_documents(docs)

        embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

        for node in tqdm(nodes, desc="Embedding nodes", unit="row"):
            node.embedding = embed_model.get_text_embedding(node.get_content())

        storage_ctx = StorageContext.from_defaults()
        self._index = VectorStoreIndex(nodes, storage_context=storage_ctx)
        self._index.storage_context.persist(persist_dir=PERSIST_DIR)
        logger.info(
            "[Index] Built with %d rows from %d tables.",
            sum(len(v) for v in data.values()),
            len(data),
        )

    def refresh(self, force: bool = True) -> None:
        if force:
            self.rebuild_index()
        else:
            self.ensure_index()

    def ensure_index(self) -> None:
        if self._index is not None:
            return
        
        docstore = Path(PERSIST_DIR) / "docstore.json"
        if docstore.exists():
            try:
                storage_ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
                self._index = load_index_from_storage(storage_ctx)
                logger.info("[Index] Loaded from disk.")
                return
            except Exception as e:
                logger.warning(f"[Index] Failed to load from disk, rebuilding. Reason: {e}")
        
        self.rebuild_index()

    def _semantic_answer(self, question: str) -> str:
        self.ensure_index()
        llm = OpenAI(model=MODEL, api_key=OPENAI_API_KEY)
        query_engine = self._index.as_query_engine(llm=llm)
        response = query_engine.query(question)
        return str(response)

    # --- LangGraph Orchestration ---
    class State(BaseModel):
        question: str
        answer: Optional[str] = None

    def _node_rag(self, state: "SheetRAGAgent.State") -> "SheetRAGAgent.State":
        try:
            state.answer = self._semantic_answer(state.question)
        except Exception as e:
            logger.error(f"RAG Error: {e}")
            state.answer = "I encountered an error while searching for an answer."
        return state

    def _build_graph(self):
        g = StateGraph(self.State)
        g.add_node("rag", self._node_rag)
        g.set_entry_point("rag")
        g.add_edge("rag", END)
        return g.compile()

    def chat(self, message: str) -> str:
        initial_state = self.State(question=message)
        final_state = self._graph.invoke(initial_state)
        return final_state.get("answer", "I could not find an answer.")