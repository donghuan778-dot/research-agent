"""
智能调研助手 Agent：用 LangChain + LangGraph + DeepSeek + LlamaIndex 实现。
4 个工具：搜索、保存报告、读取文件、知识库检索（RAG）。
"""

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.prebuilt import create_react_agent
from tools import save_report, read_local_file, search_knowledge_base

load_dotenv()

# 1. 创建 LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 2. 创建工具列表（4 个工具）
search_tool = DuckDuckGoSearchResults(
    name="search",
    description="搜索互联网上的信息。当你需要查找最新资料、验证事实时使用。",
)

tools = [search_tool, save_report, read_local_file, search_knowledge_base]

# 3. 创建 Agent
system_prompt = """你是一个智能调研助手。用户给你一个话题，你的工作流程是：
1. 先用 search_knowledge_base 工具查找本地知识库中已有的相关资料
2. 用 search 工具搜索互联网上的最新信息（可以搜索多次，用不同关键词）
3. 综合本地知识库和互联网搜索结果，撰写结构化的中文调研报告
4. 用 save_report 工具将报告保存到本地文件
严格按照 save_report 的参数格式（title、introduction、content、conclusion）来组织内容，不要在各部分中重复总结。
请用中文回答。"""

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt,
)


def research(topic: str) -> str:
    """运行调研 Agent，打印每一步的 think/act 过程"""

    print(f"\n{'='*60}")
    print(f"开始调研：{topic}")
    print(f"{'='*60}")

    step = 0
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": topic}]},
    ):
        for node_name, node_output in chunk.items():
            step += 1
            messages = node_output.get("messages", [])

            for msg in messages:
                if node_name == "agent":
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"\n--- Step {step}: Think ---")
                        for tc in msg.tool_calls:
                            print(f"  决定调用工具: {tc['name']}")
                            print(f"  参数: {tc['args']}")
                    elif hasattr(msg, "content") and msg.content:
                        print(f"\n--- Step {step}: 最终输出 ---")
                        return msg.content

                elif node_name == "tools":
                    print(f"\n--- Step {step}: Act ---")
                    print(f"  工具 '{msg.name}' 返回结果:")
                    content = str(msg.content)
                    print(f"  {content[:200]}{'...' if len(content) > 200 else ''}")

    return "调研完成"


if __name__ == "__main__":
    topic = input("请输入调研话题：")
    result = research(topic)
    print(f"\n{'='*60}")
    print(result)
