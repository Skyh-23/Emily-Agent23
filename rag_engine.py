"""
RAG (Retrieval-Augmented Generation) Module — SQLite Vault
==========================================================
- Manages vault storage using SQLite database (replaces vault.txt)
- Auto-migrates existing vault.txt data on first run
- Creates embeddings with sentence-transformers
- Retrieves top-K relevant chunks via cosine similarity
"""

import os
import sqlite3
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import VAULT_FILE, VAULT_DB, TOP_K_RESULTS, CHUNK_SIZE_CHARS, EMBEDDING_MODEL
from embedding_model import get_embedding_model

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self):
        self.model = get_embedding_model()  # Shared singleton — no duplicate load
        self.vault_db = VAULT_DB
        self.vault_file = VAULT_FILE  # For migration
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

        # Initialize SQLite database
        self._init_db()

        # Migrate vault.txt → SQLite if needed
        self._migrate_from_txt()

        # Load existing vault
        self._reload_vault()
        print(f"✅  RAG engine ready  ({len(self.chunks)} chunks in vault)")
        logger.info("RAG engine ready: %d chunks, model=%s", len(self.chunks), EMBEDDING_MODEL)

    # ------------------------------------------------------------------
    # SQLite database management
    # ------------------------------------------------------------------
    def _init_db(self):
        """Create the SQLite vault database and table if not exists."""
        conn = sqlite3.connect(self.vault_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vault (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        logger.info("SQLite vault initialized: %s", self.vault_db)

    def _migrate_from_txt(self):
        """Auto-migrate vault.txt content to SQLite on first run."""
        if not os.path.exists(self.vault_file):
            return

        # Check if migration already done
        conn = sqlite3.connect(self.vault_db)
        count = conn.execute("SELECT COUNT(*) FROM vault").fetchone()[0]
        conn.close()

        if count > 0:
            # Already has data, skip migration
            return

        try:
            with open(self.vault_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:
                # Insert each line as a separate entry
                lines = [line.strip() for line in content.split("\n") if line.strip()]
                conn = sqlite3.connect(self.vault_db)
                for line in lines:
                    conn.execute("INSERT INTO vault (content) VALUES (?)", (line,))
                conn.commit()
                conn.close()

                print(f"📦  Migrated {len(lines)} entries from vault.txt → vault.db")
                logger.info("Migrated %d entries from vault.txt", len(lines))

                # Rename old file as backup
                backup = self.vault_file + ".bak"
                os.rename(self.vault_file, backup)
                print(f"📦  Old vault.txt backed up to {backup}")
        except Exception as e:
            logger.error("Migration failed: %s", e)
            print(f"⚠️  Migration failed: {e}")

    # ------------------------------------------------------------------
    # Vault management  (voice commands)
    # ------------------------------------------------------------------
    def insert_info(self, text: str):
        """Add text to the SQLite vault and re-embed."""
        conn = sqlite3.connect(self.vault_db)
        conn.execute("INSERT INTO vault (content) VALUES (?)", (text.strip(),))
        conn.commit()
        conn.close()

        # Phase-2 NSB: best-effort graph extraction for ingested document text.
        try:
            from graph_extractor import extract_document_graph
            from graph_store import upsert_node, add_edge

            extracted = extract_document_graph(
                text=text,
                source="rag_vault",
                metadata={"vault_db": self.vault_db},
            )

            for node in extracted.get("nodes", []):
                upsert_node(
                    node_id=node["id"],
                    content=node["content"],
                    node_type=node.get("node_type", "document"),
                    tags=node.get("tags", []),
                    metadata=node.get("metadata", {}),
                    importance=node.get("importance", 1.0),
                )

            for edge in extracted.get("edges", []):
                add_edge(
                    source_id=edge["source_id"],
                    target_id=edge["target_id"],
                    relationship=edge.get("relationship", "related"),
                    weight=edge.get("weight", 0.5),
                    metadata=edge.get("metadata", {}),
                )
        except Exception as e:
            logger.warning("Graph extract skipped for RAG insert: %s", e)

        print(f'📝  Inserted into vault: "{text[:80]}..."')
        logger.info("Inserted into vault: %d chars", len(text))
        self._reload_vault()

    def print_info(self) -> str:
        """Return current vault contents."""
        conn = sqlite3.connect(self.vault_db)
        rows = conn.execute("SELECT content FROM vault ORDER BY id").fetchall()
        conn.close()

        if not rows:
            return "The vault is empty, boss."

        content = "\n".join(row[0] for row in rows)
        return content

    def delete_info(self):
        """Delete all vault entries."""
        conn = sqlite3.connect(self.vault_db)
        conn.execute("DELETE FROM vault")
        conn.commit()
        conn.close()

        self.chunks = []
        self.embeddings = None
        print("🗑️  Vault cleared.")
        logger.info("Vault cleared")

    # ------------------------------------------------------------------
    # PDF ingestion
    # ------------------------------------------------------------------
    def ingest_pdf(self, pdf_path: str):
        """Extract text from a PDF file and add to vault."""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(pdf_path)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text.strip())

            full_text = "\n".join(text_parts)
            if full_text:
                self.insert_info(full_text)
                print(f"📄  Ingested PDF '{pdf_path}' → {len(full_text)} chars added to vault.")
                logger.info("Ingested PDF: %s (%d chars)", pdf_path, len(full_text))
            else:
                print(f"⚠️  No text extracted from '{pdf_path}'.")
        except Exception as e:
            print(f"❌  PDF ingestion error: {e}")
            logger.error("PDF ingestion error: %s", e)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def get_relevant_context(self, query: str) -> str:
        """Return the top-K most relevant vault chunks for a query."""
        if self.embeddings is None or len(self.chunks) == 0:
            return ""

        query_emb = self.model.encode([query])
        similarities = cosine_similarity(query_emb, self.embeddings)[0]

        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:TOP_K_RESULTS]

        relevant = []
        for idx in top_indices:
            if similarities[idx] > 0.15:       # similarity threshold
                relevant.append(self.chunks[idx])

        if relevant:
            context = "\n---\n".join(relevant)
            print(f"📚  RAG retrieved {len(relevant)} relevant chunk(s)")
            logger.info("RAG retrieved %d chunk(s)", len(relevant))
            return context
        return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reload_vault(self):
        """Re-read vault from SQLite, chunk it, and create embeddings."""
        conn = sqlite3.connect(self.vault_db)
        rows = conn.execute("SELECT content FROM vault ORDER BY id").fetchall()
        conn.close()

        if not rows:
            self.chunks = []
            self.embeddings = None
            return

        # Combine all entries
        content = "\n".join(row[0] for row in rows)

        if not content.strip():
            self.chunks = []
            self.embeddings = None
            return

        # Chunk the text
        self.chunks = self._chunk_text(content, CHUNK_SIZE_CHARS)

        # Create embeddings
        if self.chunks:
            self.embeddings = self.model.encode(self.chunks)
            print(f"🔢  Embedded {len(self.chunks)} chunks")
            logger.info("Embedded %d chunks", len(self.chunks))

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> list[str]:
        """Split text into overlapping chunks."""
        sentences = text.replace("\n", " ").split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks


# ── quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = RAGEngine()

    # Test insert
    rag.insert_info("Meeting with marketing team at 3pm tomorrow about Q2 campaign.")
    rag.insert_info("Budget review scheduled for Friday with the finance department.")

    # Test retrieval
    context = rag.get_relevant_context("What meetings do I have?")
    print(f"\nRetrieved context:\n{context}")

    # Test print
    print(f"\nVault contents:\n{rag.print_info()}")
