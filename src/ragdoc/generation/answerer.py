"""Answer generation over retrieved context.

Two backends behind one interface:

- ClaudeAnswerer: sends numbered context chunks to the Anthropic API with a
  grounding prompt that instructs the model to cite chunk numbers and refuse
  when the context is insufficient (the standard anti-hallucination pattern).
- ExtractiveAnswerer: zero-dependency fallback that returns the most
  query-relevant sentences verbatim. Useful for offline demos and as a
  baseline when evaluating whether the LLM actually adds value.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from ragdoc.retrieval.bm25 import tokenize
from ragdoc.retrieval.hybrid import RetrievalResult

SYSTEM_PROMPT = """You are a careful document Q&A assistant.
Answer the user's question using ONLY the numbered context chunks provided.
Rules:
- Cite the chunks you used inline, e.g. [1] or [2][3].
- If the context does not contain the answer, say so plainly instead of guessing.
- Be concise and factual."""


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[RetrievalResult]
    backend: str


def format_context(results: list[RetrievalResult]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] (source: {r.chunk.source})\n{r.chunk.text}")
    return "\n\n".join(blocks)


class ClaudeAnswerer:
    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                'anthropic is not installed. Install with: pip install "ragdoc[llm]"'
            ) from exc
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic()
        self._model = model

    def answer(self, question: str, results: list[RetrievalResult]) -> Answer:
        if not results:
            return Answer("No relevant context was found in the index.", [], "claude")
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n\n{format_context(results)}\n\nQuestion: {question}",
                }
            ],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return Answer(text=text, sources=results, backend="claude")


class ExtractiveAnswerer:
    """Returns the sentences from the retrieved chunks that best match the query."""

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def answer(self, question: str, results: list[RetrievalResult]) -> Answer:
        if not results:
            return Answer("No relevant context was found in the index.", [], "extractive")
        query_terms = set(tokenize(question))
        scored: list[tuple[float, int, str]] = []
        for chunk_rank, result in enumerate(results):
            sentences = re.split(r"(?<=[.!?])\s+", result.chunk.text)
            for sent in sentences:
                terms = set(tokenize(sent))
                if not terms:
                    continue
                overlap = len(query_terms & terms) / len(query_terms | terms)
                if overlap > 0:
                    scored.append((overlap, chunk_rank, sent.strip()))
        scored.sort(key=lambda t: (-t[0], t[1]))
        picked = [sent for _, _, sent in scored[: self.max_sentences]]
        text = (
            " ".join(picked)
            if picked
            else "The retrieved context did not contain a direct answer."
        )
        return Answer(text=text, sources=results, backend="extractive")


def auto_answerer(model: str = "claude-sonnet-4-5"):
    """Prefer the LLM backend when configured; fall back to extractive."""
    try:
        return ClaudeAnswerer(model=model)
    except (ImportError, RuntimeError):
        return ExtractiveAnswerer()
