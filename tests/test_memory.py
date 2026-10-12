import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import numpy as np
import pytest
from psychanchor.interviewer import Interviewer
from psychanchor.prompts import get_prompt


@pytest.mark.parametrize("top_k", [1, 2, 10])
def test_retrieval_filters_far_vectors_and_handles_small_index(top_k):
    obj = Interviewer()
    obj.get_embedding_for_text = AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.0, 0.0])]))
    index = obj.build_vector_index(np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32))
    docs, distances = asyncio.run(obj.retrieve_similar_docs("q", index, ["near", "far"], top_k=top_k, similarity_threshold=0.7))
    assert docs == ["near"]
    assert distances.tolist() == [0.0]


def test_empty_index_returns_no_memory():
    obj = Interviewer()
    obj.get_embedding_for_text = AsyncMock(side_effect=AssertionError("Empty retrieval must not call a provider"))
    index = obj.build_vector_index(np.empty((0,2),dtype=np.float32))
    docs, distances = asyncio.run(obj.retrieve_similar_docs("q", index, []))
    assert docs == [] and len(distances) == 0


def test_dynamic_prompt_selects_only_needed_templates():
    prompt = get_prompt("dynamic_interview_generator", template_keys=["system", "context_template"], static_interview_data="HISTORY_SENTINEL")
    assert "HISTORY_SENTINEL" in prompt["context_template"]


def test_save_and_reload_faiss(tmp_path, monkeypatch):
    monkeypatch.setenv("PSYCHANCHOR_DATA_DIR", str(tmp_path))
    obj = Interviewer()
    vectors = np.array([[0.0, 1.0]], dtype=np.float32)
    qa = [{"q":"question", "a":"answer"}]
    async def run():
        index = obj.build_vector_index(vectors)
        await obj.save_rag_database("5c771e5f3c", vectors, index, obj.format_qa_pairs(qa), qa)
        restored, docs, pairs = await obj.load_rag_database("5c771e5f3c")
        assert restored.ntotal == 1
        assert pairs == qa and docs == ["Q：question A：answer"]
    asyncio.run(run())
