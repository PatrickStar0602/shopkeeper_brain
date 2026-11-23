import json
import logging
from typing import Tuple, List, Dict, Any

from langchain.messages import SystemMessage, HumanMessage
from knowledge.processor.query_process.base import BaseNode

from knowledge.processor.query_process.config import get_config
from knowledge.processor.query_process.state import QueryGraphState, get_default_state
from knowledge.prompts.query_prompt import ITEM_NAME_USER_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages


class ItemNameConfirmedNode(BaseNode):

    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self.extractor = _ItemNameExtractor()
        self.aligner = _ItemNameAligner()

    def process(self, state:QueryGraphState)->QueryGraphState:
        #从state中获取原始查询问题"original_query"
        original_query = state.get("original_query")
        #根据session_id查询历史纪录
        session_id = state.get("session_id")
        history_list = get_recent_messages(
            session_id=session_id,
            limit=8
        )
        state["history"] = history_list
        #格式化历史记录
        nick_names = {
            "user": "用户",
            "assistant": "助手"
        }
        formatted_lines = []
        for message in history_list:
            role = message.get("role")
            text = message.get("text")
            formatted_line = nick_names.get(role) + ": " + text
            formatted_lines.append(formatted_line)
        history_content = "\n".join(formatted_lines)
        #将原始查询问题与历史记录作为上下文，封装提示词，调用LLM生成问题的商品名，并对问题改写
        item_names, rewritten_query = self.extractor.extract_item_name(original_query, history_content)
        #将LLM生成到item_name与向量数据库中的item_name进行对齐，并分类
        if item_names:
            confirmed, options = self.aligner.search_and_align(item_names)
        else:
            confirmed, options = [], []
        #决策处理
        if confirmed:
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query
        elif options:
            state["answer"] = f"我不确定您指的是哪个商品，请问您是在询问以下商品吗？\n[{', '.join(options)}]"
        else:
            state["answer"] = "我无法理解您的问题，请您重新描述。"
        return state

class _ItemNameExtractor:
    def extract_item_name(self, original_query:str, history_content:str)->Tuple[List[str],str]:
        #封住提示词
        system_prompt = "你是一位商品名提取专家，请从用户的问题以及历史对话中提取相关的商品名以及改写原始查询"
        history_text = history_content if history_content.strip() else "暂无历史上下文"
        user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=history_text,
            query=original_query
        )
        #调用LLM
        llm_client = AIClients.get_llm_client()
        llm_response = llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        llm_result = llm_response.content
        #解析结果
        item_names, rewritten_query = self._clean_and_parse_llm_result(llm_result)
        #如果改写后的查询为空，则使用原始查询
        rewritten_query = rewritten_query if rewritten_query else original_query
        return item_names, rewritten_query

    def _clean_and_parse_llm_result(self, llm_result:str)->Tuple[List[str], str]:
        #json字符串反序列化，str--->对象
        json_obj = json.loads(llm_result)
        #获取商品名列表
        raw_item_names = json_obj.get("item_names", [])
        if isinstance(raw_item_names, list):
            item_names = [name.strip() for name in raw_item_names]
        else:
            item_names = []
        #获取改写后的查询
        raw_rewritten_query = json_obj.get("rewritten_query", "")
        if isinstance(raw_rewritten_query, str):
            rewritten_query = raw_rewritten_query.strip()
        else:
            rewritten_query = ""
        return item_names, rewritten_query

class _ItemNameAligner:
    def search_and_align(self, item_names:List[str]) -> Tuple[List[str], List[str]]:
        #根据LLM生成的item_names，先进行向量化，然后进行混合向量检索
        search_results = self._search_vector(item_names)
        json_str = json.dumps(search_results, indent=4, ensure_ascii=False)
        print(json_str)
        #商品名对齐
        confirmed, options = self._item_name_score_aligns(search_results)
        #如果confirmed有两个及以上的商品名称，则进行分数差异化过滤
        if len(confirmed) >= 2:
            confirmed = self._item_name_score_filter(confirmed, search_results)
        return confirmed, options

    def _search_vector(self, item_names:List[str])->List[Dict[str, Any]]:
        search_results = []
        #获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            logging.error(f"Milvus连接获取失败：{e}")
            return []
        #获取bge-m3模型客户端
        try:
            bge_m3 = AIClients.get_bge_m3_client()
        except Exception as e:
            logging.error(f"BGE-M3模型获取失败：{e}")
            return []
        #对所有item_name进行向量化
        item_names_vectors = generate_bge_m3_hybrid_vectors(bge_m3, item_names)
        for index, item_name in enumerate(item_names):
            dense_vector = item_names_vectors["dense"][index]
            sparse_vector = item_names_vectors["sparse"][index]
            #混合检索
            #创建多个AnnSearchRequest实例
            requests = create_hybrid_search_requests(dense_vector, sparse_vector)
            #执行混合搜索
            res = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=get_config().item_name_collection,
                search_requests=requests,
            )
            #解析混合检索的结果
            if res:
                hybrid_hits = res[0]
                current_item_name_results = []
                for hit in hybrid_hits:
                    current_item_name_results.append({
                        "item_name": hit.entity.get("item_name"),
                        "score": hit.distance
                    })
            #将结果保存到search_results中
            search_results.append({
                "extracted_names": item_name,
                "matches": current_item_name_results
            })
        return search_results


    def _item_name_score_aligns(self, search_results):
        config = get_config()
        confirmed = []
        options = []

        for search_result in search_results:
            extract_name = search_result.get("extracted_names")
            matches = search_result.get("matches")
            #将matches中的结果按得分从高到低进行排序
            matches_sorted = sorted(matches, key=lambda x: x["score"], reverse=True)
            #从matches_sorted中获取评分高于高置信阈值的结果
            high = [match for match in matches_sorted if match["score"] >= config.item_name_high_confidence]
            if high:
                exact_hit = next((h for h in high if str(h["item_name"]) == extract_name), None)
                if exact_hit:
                    if exact_hit["item_name"] not in confirmed:
                        confirmed.append(exact_hit["item_name"])
                elif len(high) == 1:
                    if high[0]["item_name"] not in confirmed:
                        confirmed.append(high[0]["item_name"])
                else:
                    #如果high有多个高置信，且high[0]和high[1]之间的得分差距大于阈值，则选择high[0]
                    if high[0]["score"] - high[1]["score"] > config.item_name_score_gap:
                        if high[0]["item_name"] not in confirmed:
                            confirmed.append(high[0]["item_name"])
                    else:
                        #如果有多个高置信，且high[0]和high[1]之间的得分差距小于阈值，则将多个高置信放入options
                        for h in high[:config.item_name_max_options]:
                            picked = h.get("item_name")
                            if picked not in options and picked not in confirmed:
                                options.append(picked)
            else:
                mid = [match for match in matches_sorted if match["score"] >= config.item_name_mid_confidence]
                if mid:
                    for m in mid[:config.item_name_max_options]:
                        if m.get("item_name") not in options:
                            options.append(m.get("item_name"))
        return confirmed, options

    def _item_name_score_filter(self, confirmed, search_results):
        #构建商品名->最高分数的映射
        item_name_score = {}
        for search_result in search_results:
            match = search_result.get("matches", [])
            for m in match:
                score = m.get("score", 0)
                item_name = m.get("item_name")
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)

        #防御性检查
        if not item_name_score:
            return confirmed

        #取出分数最大的作为基准
        max_score = max(item_name_score.values())
        final_confirmed = []
        for name in confirmed:
            name_score = item_name_score.get(name, 0)
            if max_score - name_score > get_config().item_name_score_gap:
                continue
            else:
                final_confirmed.append(name)

        return final_confirmed

# 测试
if __name__ == "__main__":
    node = ItemNameConfirmedNode()
    state = get_default_state()
    state["original_query"] = "帮我对比一下华为擎云940和740的区别"

    state = node.process(state)
    json_str = json.dumps(state, indent=4, ensure_ascii=False)
    print(json_str)

