import asyncio
import json
from typing import List, Dict, Any

from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult, TextContent

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class WebMcpSearchNode(BaseNode):

    name = "web_mcp_search_node"

    def process(self, state: QueryGraphState) -> Dict[str, Any]:
        #获取参数
        rewritten_query = state.get("rewritten_query", "")
        item_names = state.get("item_names", [])

        #调用MCP工具进行网络搜索
        mcp_search_results = asyncio.run(self._call_mcp(rewritten_query))

        #处理结果
        if mcp_search_results:
            return {"web_search_docs": mcp_search_results}
        return {"web_search_docs":[]}

    async def _call_mcp(self, rewritten_query)->List[Dict[str, Any]]:
        mcp_search_results = []
        #创建MCP服务访问客户端
        async with MCPServerStreamableHttp(
            name="联网搜索",
            params={
                "url":self.config.mcp_dashscope_base_url,
                "headers":{"Authorization":f"Bearer {self.config.mcp_dashscope_api_key}"}
            },
            cache_tools_list=True,
            max_retry_attempts=3
        ) as mcp_client:
            #调用MCP服务下的工具
            mcp_response:CallToolResult = await mcp_client.call_tool(
                tool_name="bailian_web_search",
                arguments={
                    "query": rewritten_query,
                    "count": 5,
                    "timeout": 60
                }
            )
            #处理MCP工具返回结果
            result:List[TextContent] = mcp_response.content
            #从结果中获取JSON字符串
            json_str : str = result[0].text
            #反序列化
            obj = json.loads(json_str)
            pages = obj.get("pages", [])
            for page in pages:
                title = page.get("title", "")
                url = page.get("url", "")
                snippet = page.get("snippet", "")
                mcp_search_results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "content": snippet
                })
        return mcp_search_results

