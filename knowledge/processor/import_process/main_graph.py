import logging
import json

from langgraph.graph import StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.embeding_chunks_node import EmbeddingChunksNode
from knowledge.processor.import_process.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.milvus_import_node import MilvusImportNode
from knowledge.processor.import_process.state import ImportGraphState, get_default_state
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.nodes.md_img_node import MdImgNode
from knowledge.processor.import_process.nodes.document_spliter_node import DocumentSpliterNode


def my_router(state: ImportGraphState):
    is_pdf_read_enabled = state.get("is_pdf_read_enabled")
    is_md_read_enabled = state.get("is_md_read_enabled")
    if is_pdf_read_enabled:
        return "pdf"
    elif is_md_read_enabled:
        return "md"
    else:
        return "unknown"

def create_import_graph()-> StateGraph:
    #创建状态图
    graph = StateGraph(ImportGraphState)

    #添加节点
    graph.add_node("entry_node", EntryNode())
    graph.add_node("pdf_to_md_node", PdfToMdNode())
    graph.add_node("md_img_node", MdImgNode())
    graph.add_node("document_spliter_node", DocumentSpliterNode())
    graph.add_node("item_name_recognition_node", ItemNameRecognitionNode())
    graph.add_node("embedding_chunks_node", EmbeddingChunksNode())
    graph.add_node("milvus_import_node", MilvusImportNode())

    #添加边
    graph.add_edge("__start__", "entry_node")
    graph.add_conditional_edges("entry_node", my_router, {
        "pdf": "pdf_to_md_node",
        "md": "md_img_node",
        "unknown": "__end__"
    })
    graph.add_edge("pdf_to_md_node", "md_img_node")
    graph.add_edge("md_img_node", "document_spliter_node")
    graph.add_edge("document_spliter_node", "item_name_recognition_node")
    graph.add_edge("item_name_recognition_node", "embedding_chunks_node")
    graph.add_edge("embedding_chunks_node", "milvus_import_node")
    graph.add_edge("milvus_import_node", "__end__")

    #编译状态图
    return graph.compile()

#测试
if __name__ == "__main__":
    #开启日志
    setup_logging(logging.DEBUG)

    graph = create_import_graph()

    state = get_default_state()
    state["import_file_path"] = r"D:\Python\shopkeeper_brain\knowledge\processor\import_process\import_files\万用表RS-12的使用.pdf"
    state["file_dir"] = r"D:\Python\shopkeeper_brain\knowledge\processor\import_process\import_files"

    state = graph.invoke(state)
    json_str = json.dumps(state, indent=4, ensure_ascii=False)
    print(json_str)