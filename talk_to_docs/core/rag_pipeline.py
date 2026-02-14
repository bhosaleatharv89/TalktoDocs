"""End-to-end RAG answer generation pipeline."""
from __future__ import annotations

from core.retriever import Retriever
from services.llm_service import BaseLLMService


class RAGPipeline:
    """Combine retrieval and LLM generation into grounded Q&A."""

    def __init__(self, retriever: Retriever, llm_service: BaseLLMService) -> None:
        self.retriever = retriever
        self.llm_service = llm_service

    def answer(self, question: str) -> tuple[str, list[dict]]:
        """Produce an answer and citation metadata for retrieved chunks."""
        retrieved = self.retriever.retrieve(question)
        if not retrieved:
            return (
                "I could not find sufficiently relevant information in the indexed documents.",
                [],
            )

        context_parts = []
        citations: list[dict] = []
        for idx, item in enumerate(retrieved, start=1):
            context_parts.append(
                f"[{idx}] Source: {item.source_file} | Relevance: {item.score:.3f}\n{item.text}"
            )
            citations.append(
                {
                    "rank": idx,
                    "source": item.source_file,
                    "score": round(item.score, 4),
                }
            )

        context = "\n\n".join(context_parts)
        prompt = (
            "Use the context below to answer the question. "
            "Only state information that appears in the context. "
            "If the context is insufficient, say so explicitly.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer with a concise explanation and include references in [n] format."
        )
        answer = self.llm_service.generate(prompt)
        return answer, citations
