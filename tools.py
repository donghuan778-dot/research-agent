"""
自定义工具：用 LangChain 的 @tool 装饰器定义。
"""

import os
import json
from datetime import datetime
from langchain_core.tools import tool
from rag import query_knowledge


@tool
def save_report(title: str, introduction: str, content: str, conclusion: str) -> str:
    """将调研报告分为引言，内容，结论三部分然后保存为本地文件。当调研完成需要保存结果时使用。"""

    # 创建 reports 文件夹
    os.makedirs("reports", exist_ok=True)

    # 用时间戳生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/{timestamp}_{title}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 引言\n\n{introduction}\n\n")
        f.write(f"## 内容\n\n{content}\n\n")
        f.write(f"## 结论\n\n{conclusion}\n\n")

    return f"报告已保存到 {filename}"


@tool
def read_local_file(file_path: str) -> str:
    """读取本地文件内容。当需要读取用户指定的本地文件时使用。"""

    if not os.path.exists(file_path):
        return f"文件不存在：{file_path}"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 限制返回长度，防止 token 爆炸
    if len(content) > 5000:
        content = content[:5000] + "\n\n...（内容过长，已截断）"

    return content


@tool
def search_knowledge_base(question: str) -> str:
    """搜索本地知识库。当需要查找已有的调研资料、技术文档、学习笔记时使用。
    这个工具搜索的是本地保存的文档，不是互联网。"""

    return query_knowledge(question, top_k=3)
