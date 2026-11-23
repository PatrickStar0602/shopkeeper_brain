from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients


class RerankerNode(BaseNode):

    name = "reranker_node"

    def process(self, state: QueryGraphState)->QueryGraphState:
        #对rrf_ chunks和web_search_docs格式统一处理
        reranker_inputs = self._collect_rerank_inputs(state)

        #调用BGE-Reranker-large模型进行打分，并按分数排序
        rewritten_query = state.get("rewritten_query")
        reranked_docs = self._rerank(rewritten_query, reranker_inputs)

        #断崖检测，获取TopK
        cutoff_results = self._cliff_cutoff(reranked_docs)
        state["reranked_docs"] = cutoff_results

        return state

    def _collect_rerank_inputs(self, state)->List[Dict[str, Any]]:
        results = []
        rrf_chunks = state.get("rrf_chunks", [])
        for chunk in rrf_chunks:
            results.append({
                "chunk_id":chunk.get("chunk_id"),
                "content":chunk.get("content") or "",
                "title":chunk.get("title"),
                "url":None,
                "score":"local"
            })

        web_search_docs = state.get("web_search_docs", [])
        for doc in web_search_docs:
            results.append({
                "chunk_id":None,
                "content":doc.get("content") or "",
                "title":doc.get("title"),
                "url":doc.get("url"),
                "score":"web"
            })

        return results

    def _rerank(self, rewritten_query, reranker_inputs):
        if not reranker_inputs:
            return []
        reranked_docs = []
        #打分
        reranker_client = AIClients.get_bge_rerank_client()
        sentence_pairs = [(rewritten_query, input_dict.get("content")) for input_dict in reranker_inputs]
        scores = reranker_client.compute_score(sentence_pairs)
        for i, score in enumerate(scores):
            doc = reranker_inputs[i]
            reranked_docs.append({**doc, "score": score})
        #排序
        reranked_docs = sorted(reranked_docs, key=lambda x: x.get("score"), reverse=True)
        #返回
        return reranked_docs

    def _cliff_cutoff(self, reranked_docs):
        scores = [doc.get("score") for doc in reranked_docs]
        if not scores:
            return []
        normalized_scores = self._normalize_scores(scores)
        #断崖检测
        min_top_k = self.config.rerank_min_top_k
        max_top_k = self.config.rerank_max_top_k
        max_bond = max_top_k if max_top_k < len(normalized_scores) else len(normalized_scores)
        end_index = max_bond
        #找到第一个断崖
        for i in range(min_top_k-1, max_bond-1):
            current_score = normalized_scores[i]
            next_score = normalized_scores[i+1]
            if current_score - next_score > self.config.rerank_gap_abs:
                end_index = i + 1
                break
        return reranked_docs[:end_index]

    def _normalize_scores(self, scores):
        #归一化
        max_score = max(scores)
        min_score = min(scores)
        normalized_scores = []
        for i, score in enumerate(scores):
            new_score = (score - min_score) / (max_score - min_score) if max_score != min_score else 0
            normalized_scores.append(new_score)
        return normalized_scores


