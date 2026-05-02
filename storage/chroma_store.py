"""ChromaDB persistence layer for responses and evaluation scores."""

import json

import chromadb

from config import CHROMA_PATH


class ChromaStore:
    def __init__(self, path: str = None):
        self._client = chromadb.PersistentClient(path=path or CHROMA_PATH)
        self._responses = self._client.get_or_create_collection("responses")
        self._results = self._client.get_or_create_collection("results")

    # ── Responses ─────────────────────────────────────────────────────────────

    def store_response(
        self,
        question_id: str,
        model_name: str,
        run_index: int,
        reasoning: str,
        answer: str,
        embedding: list[float],
    ):
        doc_id = f"{question_id}__{model_name}__{run_index}"
        self._responses.upsert(
            ids=[doc_id],
            documents=[reasoning],
            embeddings=[embedding],
            metadatas=[{
                "question_id": question_id,
                "model_name": model_name,
                "run_index": run_index,
                "answer": answer,
            }],
        )

    def get_response(self, question_id: str, model_name: str, run_index: int = 0) -> dict | None:
        doc_id = f"{question_id}__{model_name}__{run_index}"
        res = self._responses.get(ids=[doc_id], include=["documents", "metadatas", "embeddings"])
        if not res["ids"]:
            return None
        return {
            "reasoning": res["documents"][0],
            "embedding": res["embeddings"][0],
            **res["metadatas"][0],
        }

    # ── Results ───────────────────────────────────────────────────────────────

    def store_result(self, question_id: str, model_name: str, scores: dict):
        doc_id = f"{question_id}__{model_name}"
        flat_scores = {k: float(v) for k, v in scores.items() if isinstance(v, (int, float))}
        self._results.upsert(
            ids=[doc_id],
            documents=[json.dumps(scores)],
            metadatas=[{"question_id": question_id, "model_name": model_name, **flat_scores}],
        )

    def get_all_results(self) -> list[dict]:
        res = self._results.get(include=["documents", "metadatas"])
        if not res["ids"]:
            return []
        return [
            {**json.loads(doc), **meta}
            for doc, meta in zip(res["documents"], res["metadatas"])
        ]

    def get_results_for_model(self, model_name: str) -> list[dict]:
        res = self._results.get(
            where={"model_name": model_name},
            include=["documents", "metadatas"],
        )
        return [json.loads(doc) for doc in res["documents"]]
