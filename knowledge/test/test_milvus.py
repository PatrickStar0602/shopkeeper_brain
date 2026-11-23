from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors

if __name__ == "__main__":

    file_title = "万用表使用说明"
    item_name = "RS-12数字万用表"

    bge_m3 = AIClients.get_bge_m3_client()
    result = generate_bge_m3_hybrid_vectors(bge_m3, [item_name])
    item_name_dense = result["dense"][0]
    item_name_sparse = result["sparse"][0]

    milvus_client = StorageClients.get_milvus_client()
