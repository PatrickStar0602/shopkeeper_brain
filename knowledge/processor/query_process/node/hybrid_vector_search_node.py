from typing import Dict, Any

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query


class HybridVectorSearchNode(BaseNode):

    name = "hybrid_vector_search_node"

    def process(self, state: QueryGraphState)->Dict[str, Any]:
        #校验参数
        rewritten_query = state.get("rewritten_query", "")
        item_names = state.get("item_names", [])

        #获取嵌入模型客户端
        bge_m3 = AIClients.get_bge_m3_client()

        #查询问题向量化
        vectors = generate_bge_m3_hybrid_vectors(bge_m3, [rewritten_query])
        dense_vector = vectors["dense"][0]
        sparse_vector = vectors["sparse"][0]

        #获取Milvus客户端
        milvus_client = StorageClients.get_milvus_client()

        #创建混合搜索请求
        hybrid_requests = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr="item_name in {list}",
            expr_params={"list": item_names}
        )

        #执行混合搜索
        hybrid_result = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_requests,
            output_fields=["file_title", "title", "content", "item_name"]
        )

        #处理搜索结果
        if not hybrid_result or not hybrid_result[0]:
            #return state
            return {"embedding_chunks":[]}

        #state["embedding_chunks"] = hybrid_result[0]
        #return state
        return {"embedding_chunks": hybrid_result[0]}
