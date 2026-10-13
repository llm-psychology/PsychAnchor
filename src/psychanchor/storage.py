import os
from pathlib import Path

def rag_directory():
    return Path(os.getenv("PSYCHANCHOR_DATA_DIR", "data/psychanchor")).resolve() / "rag"

def history_directory():
    return Path(os.getenv("CONVERSATION_DATA_DIR", "data/conversations")).resolve()
