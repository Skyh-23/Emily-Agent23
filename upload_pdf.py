"""
Document & PDF Memory Ingestion
================================
Extracts text from PDF / TXT / DOCX files and embeds each chunk
directly into LanceDB (same DB Emily uses for memory search).

Emily can then recall document content naturally via semantic search —
no extra plumbing needed.

Usage:
    python upload_pdf.py <file_path>          # ingest one file
    python upload_pdf.py <file_path> --list   # show all doc memories
    python upload_pdf.py --list               # list all document memories
    python upload_pdf.py --delete <filename>  # delete memories for a file
"""

import sys
import os
import re
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

CHUNK_SIZE   = 600   # characters per chunk (sweet spot for bge-m3)
CHUNK_OVERLAP = 80   # overlap between chunks for context continuity


# ── Text extraction ────────────────────────────────────────────────────

def extract_text_pdf(path: str) -> str:
    """Extract text from PDF using PyPDF2 (already in requirements)."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t.strip())
        return "\n\n".join(pages)
    except ImportError:
        # Try pdfplumber as fallback
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n\n".join(
                    p.extract_text() or "" for p in pdf.pages
                )
        except ImportError:
            print("❌  Install PyPDF2:  pip install PyPDF2")
            sys.exit(1)


def extract_text_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_text_docx(path: str) -> str:
    try:
        import docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        print("❌  Install python-docx:  pip install python-docx")
        sys.exit(1)


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_pdf(path)
    elif ext in (".txt", ".md", ".csv", ".json"):
        return extract_text_txt(path)
    elif ext == ".docx":
        return extract_text_docx(path)
    else:
        print(f"⚠️  Unsupported format '{ext}' — trying plain text read.")
        return extract_text_txt(path)


# ── Chunking ───────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks on sentence boundaries."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= chunk_size:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from previous
            if current and overlap > 0:
                # Take last `overlap` chars of previous chunk as context
                overlap_text = current[-overlap:].strip()
                current = (overlap_text + " " + sent).strip()
            else:
                current = sent.strip()

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) > 20]  # drop tiny fragments


# ── Ingest into LanceDB ────────────────────────────────────────────────

def ingest_file(path: str):
    if not os.path.exists(path):
        print(f"❌  File not found: {path}")
        sys.exit(1)

    filename = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()

    print(f"\n📄  Reading: {filename}")
    text = extract_text(path)

    if not text.strip():
        print("⚠️  No text could be extracted from this file.")
        sys.exit(1)

    print(f"📝  Extracted {len(text):,} characters")

    chunks = chunk_text(text)
    print(f"🔪  Split into {len(chunks)} chunks (≤{CHUNK_SIZE} chars each)")

    # Import LanceDB memory store
    try:
        from memory_store import add_memory
    except ImportError:
        print("❌  Could not import memory_store. Run from E:\\Local_Voice\\")
        sys.exit(1)

    doc_type = "pdf" if ext == ".pdf" else "document"
    tags = ["document", doc_type, filename.replace(" ", "_")]

    print(f"\n💾  Embedding & storing in LanceDB...")
    success = 0
    failed  = 0

    for i, chunk in enumerate(chunks):
        content = f"[From document: {filename}]\n{chunk}"
        try:
            add_memory(
                content=content,
                tags=tags,
                metadata={
                    "source": "document",
                    "filename": filename,
                    "file_path": path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_type": ext,
                },
                importance=0.75,  # Slightly lower than conversations (1.0)
            )
            success += 1
            # Progress bar
            bar = int((i + 1) / len(chunks) * 30)
            print(f"  [{'█' * bar}{'░' * (30 - bar)}] {i+1}/{len(chunks)}", end="\r")
        except Exception as e:
            failed += 1
            logger.warning("Chunk %d failed: %s", i, e)

    print(f"\n\n✅  Done! {success} chunks saved to LanceDB ({failed} failed)")
    print(f"    Emily will now remember content from: {filename}")
    print(f"    Try asking her: 'What does {filename} say about...'")


# ── List document memories ─────────────────────────────────────────────

def list_document_memories():
    try:
        from memory_store import get_all_memories
        memories = get_all_memories()
    except Exception as e:
        print(f"❌  Could not load memories: {e}")
        return

    doc_memories = [
        m for m in memories
        if "document" in (m.get("tags") or [])
    ]

    if not doc_memories:
        print("📭  No document memories found.")
        return

    # Group by filename
    files: dict[str, list] = {}
    for m in doc_memories:
        fname = (m.get("metadata") or {}).get("filename", "unknown")
        files.setdefault(fname, []).append(m)

    print(f"\n📚  Document Memories ({len(doc_memories)} chunks from {len(files)} files):")
    print("─" * 50)
    for fname, chunks in sorted(files.items()):
        print(f"  📄  {fname}  ({len(chunks)} chunks)")
    print("─" * 50)


# ── Delete document memories ───────────────────────────────────────────

def delete_document_memories(filename: str):
    try:
        from memory_store import get_all_memories, delete_memory
    except ImportError:
        print("❌  delete_memory not available in memory_store.")
        return

    try:
        memories = get_all_memories()
    except Exception as e:
        print(f"❌  Could not load memories: {e}")
        return

    targets = [
        m for m in memories
        if (m.get("metadata") or {}).get("filename", "") == filename
    ]

    if not targets:
        print(f"📭  No memories found for file: {filename}")
        return

    print(f"🗑️   Deleting {len(targets)} chunks for '{filename}'...")
    deleted = 0
    for m in targets:
        try:
            delete_memory(m["id"])
            deleted += 1
        except Exception as e:
            logger.warning("Delete failed for %s: %s", m.get("id"), e)

    print(f"✅  Deleted {deleted}/{len(targets)} chunks.")


# ── Entry point ────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        return

    if args == ["--list"] or (len(args) == 1 and args[0] == "--list"):
        list_document_memories()
        return

    if len(args) == 2 and args[0] == "--delete":
        delete_document_memories(args[1])
        return

    # Ingest file
    path = args[0]
    ingest_file(path)

    if "--list" in args:
        print()
        list_document_memories()


if __name__ == "__main__":
    main()
