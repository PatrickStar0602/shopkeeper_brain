from typing import Dict, Any

from langchain.messages import SystemMessage, HumanMessage
from pymilvus import SearchResult

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query


class HyDEVectorSearchNode(BaseNode):

    name = "hyde_vector_search_node"

    def process(self, state: QueryGraphState)->Dict[str, Any]:
        #参数校验
        rewritten_query = state.get("rewritten_query", "")
        item_names = state.get("item_names", [])

        #利用LLM生成原始查询的假设性文档
        #获取LLM客户端
        llm_client = AIClients.get_llm_client(response_format=False)
        #提示词
        system_prompt = f"您是一位{item_names}的技术文档邻域专家，主要擅长编写技术文档、操作手册、文档规格说明"
        user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(
            item_names=item_names,
            rewritten_query=rewritten_query
        )
        #调用大模型
        llm_res = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        hyde_doc = llm_res.content.strip() if llm_res else None

        #对修改后的原始查询+生成的假设性文档进行向量化
        document = f"{rewritten_query}\n{hyde_doc}"
        bge_m3 = AIClients.get_bge_m3_client()
        vectors = generate_bge_m3_hybrid_vectors(bge_m3, [document])
        dense_vector = vectors["dense"][0]
        sparse_vector = vectors["sparse"][0]

        #获取Milvus客户端
        milvus_client = StorageClients.get_milvus_client()

        #生成混合向量检索请求
        hybrid_requests = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr="item_name in {list}",
            expr_params={"list": item_names}
        )

        #进行混合向量检索
        res:SearchResult = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_requests,
            output_fields=["file_title", "title", "content", "item_name"]
        )

        #处理返回结果
        if not res or not res[0]:
            return {"hybrid_embedding_chunks":[]}
        return {"hybrid_embedding_chunks":res[0]}
