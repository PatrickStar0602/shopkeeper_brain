from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RrfMergeNode(BaseNode):

    name = "rrf_merge_node"

    def process(self, state:QueryGraphState)->QueryGraphState:
        #从state中获取两路结果
        embedding_chunks = state.get("embedding_chunks", [])
        hyde_embedding_chunks = state.get("hyde_embedding_chunks", [])

        #标准化两路结果的数据结构
        validated_embedding_chunks = self._validate_search_chunks(embedding_chunks)
        validated_hyde_embedding_chunks = self._validate_search_chunks(hyde_embedding_chunks)

        #设置两路权重
        rrf_inputs = [
            (validated_embedding_chunks, 1.0),
            (validated_hyde_embedding_chunks, 1.0)
        ]

        #进行RRF融合
        rrf_results = self._rrf_merge(rrf_inputs)
        
        #取出Top_K
        rrf_chunks = rrf_results[:self.config.rrf_max_results]

        #设置到state中"rrf_chunks"
        state["rrf_chunks"] = rrf_chunks

        #返回state
        return state

    def _validate_search_chunks(self, embedding_chunks):
        formatted_chunk_list = []
        if not embedding_chunks or not isinstance(embedding_chunks, list):
            return []
        for chunk in embedding_chunks:
            if chunk and isinstance(chunk, dict):
                id = chunk.get("id")
                entity = chunk.get("entity")
                formatted_chunk = {
                    "chunk_id": id,
                    "file_title": entity.get("file_title",""),
                    "title": entity.get("title",""),
                    "content": entity.get("content",""),
                    "item_name": entity.get("item_name","")
                }
                formatted_chunk_list.append(formatted_chunk)
            else:
                return []
        return formatted_chunk_list

    def _rrf_merge(self, rrf_inputs)->List[Dict[str, Any]]:
        chunk_scores = {}
        chunk_data = {}
        for chunk_list, weight in rrf_inputs:
            for index, chunk in enumerate(chunk_list):
                chunk_id = chunk.get("chunk_id")
                rank = index + 1
                rrf_score = weight / (self.config.rrf_k + rank)
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + rrf_score
                chunk_data.setdefault(chunk_id, chunk)
        #结果处理
        rrf_results = []
        for chunk_id, score in chunk_scores.items():
            chunk = chunk_data.get(chunk_id)
            rrf_results.append((chunk, score))
        #按照分数排序
        rrf_results = sorted(rrf_results,key=lambda x:x[1],reverse=True)
        #从rrf_results取出chunk的列表
        rrf_chunks = [chunk for chunk, score in rrf_results]
        return rrf_chunks

