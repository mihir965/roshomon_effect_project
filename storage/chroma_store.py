"""ChromaDB persistence layer for LLM responses.

Caches the expensive part — the LLM query and the resulting reasoning embedding —
keyed by (question_id, model_name, run_index). A hash of the question text is
stored alongside so we can invalidate automatically when a question_id is reused
for different text (e.g. after a dataset rebuild).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

import chromadb

from config import CHROMA_PATH


def question_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class ChromaStore:
    def __init__(self, path: str = None):
        self._path = path or CHROMA_PATH
        self._client = chromadb.PersistentClient(path=self._path)
        self._responses = self._client.get_or_create_collection("responses")

    # ── Responses ─────────────────────────────────────────────────────────────

    def store_response(
        self,
        question_id: str,
        question_text: str,
        model_name: str,
        run_index: int,
        reasoning_steps: list[str],
        final_answer: str,
        raw_response: str,
        reasoning_embedding: list[float],
    ) -> None:
        doc_id = f"{question_id}__{model_name}__{run_index}"
        joined = "\n".join(reasoning_steps)
        self._responses.upsert(
            ids=[doc_id],
            documents=[joined],
            embeddings=[reasoning_embedding],
            metadatas=[{
                "question_id": question_id,
                "question_hash": question_hash(question_text),
                "model_name": model_name,
                "run_index": run_index,
                "final_answer": final_answer,
                "raw_response": raw_response,
                "steps_json": json.dumps(reasoning_steps),
            }],
        )

    def get_response(
        self,
        question_id: str,
        question_text: str,
        model_name: str,
        run_index: int,
    ) -> Optional[dict]:
        """Return cached response dict or None (cache miss / question text changed)."""
        doc_id = f"{question_id}__{model_name}__{run_index}"
        try:
            res = self._responses.get(
                ids=[doc_id],
                include=["metadatas", "embeddings"],
            )
        except Exception:
            return None
        if not res["ids"]:
            return None
        meta = res["metadatas"][0]
        if meta.get("question_hash") != question_hash(question_text):
            # The question text has changed under us; invalidate.
            return None
        try:
            steps = json.loads(meta["steps_json"])
        except Exception:
            return None
        return {
            "reasoning_steps": steps,
            "final_answer": meta.get("final_answer", ""),
            "raw_response": meta.get("raw_response", ""),
            "reasoning_embedding": res["embeddings"][0],
        }

    # ── Maintenance ───────────────────────────────────────────────────────────

    def count(self) -> int:
        try:
            return self._responses.count()
        except Exception:
            return 0

    def wipe(self) -> None:
        """Delete the entire on-disk cache directory."""
        # Drop the open client first so the file handles release.
        del self._client
        del self._responses
        p = Path(self._path)
        if p.exists():
            shutil.rmtree(p)
        # Re-init fresh
        self._client = chromadb.PersistentClient(path=self._path)
        self._responses = self._client.get_or_create_collection("responses")
