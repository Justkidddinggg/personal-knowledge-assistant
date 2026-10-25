import sys
from pathlib import Path

from app.ingestion.loader import load_document

doc = load_document(Path(sys.argv[1]))
print(f"{doc.filename} | type={doc.file_type} | pages={len(doc.pages)} | hash={doc.doc_id[:12]}")
for p in doc.pages[:3]:
    print(f"--- page={p.page} ---")
    print(p.text[:300])