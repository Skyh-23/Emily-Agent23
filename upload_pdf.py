"""
PDF-to-Vault Ingestion Script
Converts a PDF file to text and adds it to vault.txt for RAG retrieval.

Usage:
    python upload_pdf.py <path_to_pdf>
"""

import sys
import os
from rag_engine import RAGEngine


def main():
    if len(sys.argv) < 2:
        print("Usage:  python upload_pdf.py <path_to_pdf>")
        print("Example: python upload_pdf.py documents/report.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"❌  File not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.lower().endswith(".pdf"):
        print(f"❌  Not a PDF file: {pdf_path}")
        sys.exit(1)

    print(f"📄  Uploading PDF: {pdf_path}")
    rag = RAGEngine()
    rag.ingest_pdf(pdf_path)
    print(f"\n✅  Done! The PDF content is now in vault.txt and embedded for RAG.")
    print(f"    Total chunks in vault: {len(rag.chunks)}")


if __name__ == "__main__":
    main()
