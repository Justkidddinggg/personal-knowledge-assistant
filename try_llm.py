from app.rag.llm import get_llm

print(get_llm().chat([{"role": "user", "content": "Reply with exactly: OK"}]))