import logging

from knowledge.processor.query_process.base import setup_logging
from knowledge.processor.query_process.node.answer_output_node import AnswerOutputNode
from knowledge.processor.query_process.node.hybrid_vector_search_node import HybridVectorSearchNode
from knowledge.processor.query_process.node.hyde_vector_search_node import HyDEVectorSearchNode
from knowledge.processor.query_process.node.item_name_confirmed_node import ItemNameConfirmedNode
from knowledge.processor.query_process.node.reranker_node import RerankerNode
from knowledge.processor.query_process.node.rrf_merge_node import RrfMergeNode
from knowledge.processor.query_process.node.web_mcp_search_node import WebMcpSearchNode
from knowledge.processor.query_process.state import QueryGraphState, get_default_state
from langgraph.graph import StateGraph


def my_router(state: QueryGraphState):
    if state.get("answer"):
        return True
    else:
        return False

def create_query_graph()->StateGraph:
    graph = StateGraph(QueryGraphState)
    #添加节点
    graph.add_node("item_name_confirmed_node", ItemNameConfirmedNode())
    #添加多路召回虚拟节点
    graph.add_node("multi_search", lambda x: x)
    graph.add_node("hybrid_vector_search_node", HybridVectorSearchNode())
    graph.add_node("hyde_vector_search_node", HyDEVectorSearchNode())
    graph.add_node("web_mcp_search_node", WebMcpSearchNode())
    #添加汇聚虚拟节点
    graph.add_node("join_node", lambda x: {})
    graph.add_node("rrf_merge_node", RrfMergeNode())
    graph.add_node("reranker_node", RerankerNode())
    graph.add_node("answer_output_node", AnswerOutputNode())

    #添加边
    graph.add_edge("__start__", "item_name_confirmed_node")
    graph.add_conditional_edges("item_name_confirmed_node", my_router,{
        True:"answer_output_node",
        False:"multi_search"
    })
    #从multi_search分发到三路召回节点
    graph.add_edge("multi_search", "hybrid_vector_search_node")
    graph.add_edge("multi_search", "hyde_vector_search_node")
    graph.add_edge("multi_search", "web_mcp_search_node")
    #三路召回结果汇聚到join_node
    graph.add_edge("hybrid_vector_search_node", "join_node")
    graph.add_edge("hyde_vector_search_node", "join_node")
    graph.add_edge("web_mcp_search_node", "join_node")

    graph.add_edge("join_node", "rrf_merge_node")
    graph.add_edge("rrf_merge_node", "reranker_node")
    graph.add_edge("reranker_node", "answer_output_node")
    graph.add_edge("answer_output_node", "__end__")

    return graph.compile()

#测试
if __name__ == "__main__":
    import json

    setup_logging(logging.INFO)
    graph:StateGraph = create_query_graph()
    state = get_default_state()
    state["session_id"] = "s101"
    state["task_id"] = "1001"
    state["original_query"] = "华为擎云940配置？"
    state = graph.invoke(state)
    json_str = json.dumps(state, indent=4, ensure_ascii=False)
    print(json_str)

