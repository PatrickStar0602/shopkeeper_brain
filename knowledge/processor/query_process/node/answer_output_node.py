from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query_prompt import ANSWER_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import get_sse_queue, create_sse_queue, SSEEvent, push_sse_event
from knowledge.utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):

    name = "answer_output_node"

    def process(self, state: QueryGraphState)->QueryGraphState:
        #从state获取数据
        answer = state.get("answer")
        is_stream = state.get("is_stream", True)
        task_id = state.get("task_id")

        #根据state中有无answer决定是直接返回答案还是调用LLM
        if answer:
            #直接回答
            self._push_exist_answer(task_id, is_stream, answer)
        else:
            #构造prompt
            prompt:str = self._build_prompt(state)

            #调用LLM
            self._generate_answer(task_id, is_stream, prompt, state)

        #保存历史记录
        self._save_history(state)

        return state

    def _push_exist_answer(self, task_id, is_stream, answer):
        if is_stream:
            queue = get_sse_queue(task_id)
            if not queue:
                create_sse_queue(task_id)
            push_sse_event(
                task_id=task_id,
                event=SSEEvent.FINAL,
                data={"answer": answer}
            )
        else:
            set_task_result(
                task_id=task_id,
                key="answer",
                value=answer
            )

    def _build_prompt(self, state):
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        rerank_docs = state.get("reranked_docs")
        history_list = state.get("history")
        #格式化上下文
        format_content = self._format_content(rerank_docs, self.config.max_context_chars)
        used_chars = len(format_content)
        #格式化历史记录
        formatted_history = self._format_history(history_list, used_chars)

        prompt = ANSWER_PROMPT.format(
            context=format_content,
            history=formatted_history,
            item_names=item_names,
            question=rewritten_query
        )
        return prompt.strip()

    def _format_content(self, rerank_docs, max_context_chars)->str:
        formatted_lines = []
        used_chars = 0
        for index, doc in enumerate(rerank_docs):
            #获取文档内容
            content = doc.get("content")
            chunk_id = doc.get("chunk_id")
            title = doc.get("title")
            source = doc.get("source")
            url = doc.get("url")
            score = doc.get("score")
            #拼接文档内容
            line_1 = f"[文档：{index+1}]"
            line_2 = f"chunk_id={chunk_id} title={title} source={source} url={url} score={score}"
            line_3 = f"{content}"
            doc_line = f"{line_1}\n{line_2}\n{line_3}\n"
            #统计已使用的字符数
            if used_chars + len(doc_line) + 2 < max_context_chars:
                formatted_lines.append(doc_line)
                used_chars += len(doc_line) + 2
            else:
                break
        return "\n\n".join(formatted_lines)

    def _format_history(self, history_list, used_chars):
        #历史记录可以使用的最大字符数
        history_max_chars = self.config.max_context_chars - used_chars
        if history_max_chars <= 0:
            return ""
        #格式化历史记录
        nick_name = {
            "user": "用户",
            "assistant": "助手"
        }
        formatted_lines = []
        used_chars = 0
        for history in history_list:
            role = history.get("role")
            text = history.get("text")
            line = f"{nick_name.get(role)}: {text}"
            if used_chars + len(line) + 1 <= history_max_chars:
                formatted_lines.append(line)
                used_chars += len(line) + 1
            else:
                break
        return "\n".join(formatted_lines)

    def _generate_answer(self, task_id, is_stream, prompt, state):
        #创建LLM客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except Exception as e:
            self.logger.error(f"LLM模型客户端创建失败: {e}")
            state["answer"] = "很抱歉，获取答案失败！"
            self._push_exist_answer(task_id, is_stream, state["answer"])
            return
        #根据is_stream是否流式输出，采用不同的方式调用LLM
        if is_stream:
            llm_result = self._llm_stream(llm_client, prompt, task_id)
            state["answer"] = llm_result
        else:
            llm_result = self._llm_invoke(llm_client, prompt, task_id)
            state["answer"] = llm_result

    def _llm_stream(self, llm_client, prompt, task_id):
        #采用流式调用大模型
        llm_result = ""
        try:
            for section in llm_client.stream(prompt):
                delta_date = getattr(section, "content", "")
                if delta_date:
                    push_sse_event(
                        task_id=task_id,
                        event=SSEEvent.DELTA,
                        data={"delta": delta_date}
                    )
                    llm_result += delta_date
            push_sse_event(
                task_id=task_id,
                event=SSEEvent.FINAL,
                data={}
            )
        except Exception as e:
            return "很抱歉，获取答案失败！"
        return llm_result

    def _llm_invoke(self, llm_client, prompt, task_id):
        llm_resp = llm_client.invoke(prompt)
        llm_result = getattr(llm_resp, "content", "")
        if not llm_result:
            return "很抱歉，获取答案失败！"
        set_task_result(
            task_id=task_id,
            key="answer",
            value=llm_result
        )
        return llm_result

    def _save_history(self, state):
        try:
            query = state.get("original_query")
            save_chat_message(
                session_id=state.get("session_id"),
                role="user",
                text=query,
                rewritten_query=state.get("rewritten_query"),
                item_names=state.get("item_names")
            )

            answer = state.get("answer")
            save_chat_message(
                session_id=state.get("session_id"),
                role="assistant",
                text=answer,
                rewritten_query=state.get("rewritten_query"),
                item_names=state.get("item_names")
            )
        except Exception as e:
            self.logger.error(f"保存历史记录失败: {e}")
