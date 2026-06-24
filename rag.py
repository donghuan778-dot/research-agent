"""
RAG 模块：用 LlamaIndex 构建本地知识库。

核心流程：
1. 读取 knowledge_base/ 文件夹下的文档
2. 切成小块 + 向量化（Embedding）
3. 存入 ChromaDB 向量数据库
4. 提供查询接口：输入问题 → 返回最相关的文档片段
"""

import os
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def build_index():
    """
    构建知识库索引。

    这个函数做 3 件事（对应 RAG 流程的前 4 步）：
    1. 读取 knowledge_base/ 下所有文档  → Document
    2. 切块 + 向量化                    → Embedding
    3. 存入 ChromaDB                    → Storage
    """

    # 1. 读取文档
    reader = SimpleDirectoryReader(input_dir=KNOWLEDGE_DIR)
    documents = reader.load_data()
    print(f"读取了 {len(documents)} 个文档")

    # 2. 设置 Embedding 模型（免费的中文模型，本地运行）
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 3. 设置文档切分器（每块 512 字符，重叠 50 字符防止切断上下文）
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    # 4. 设置 ChromaDB 向量数据库
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("research_kb")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5. 创建索引（自动完成：切块 → 向量化 → 存入数据库）
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
    )

    print(f"索引构建完成，存储在 {CHROMA_DIR}")
    return index


def load_index():
    """
    加载已有的索引（不用每次都重建）。
    """
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("research_kb")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return index


def query_knowledge(question: str, top_k: int = 3) -> str:
    """
    查询知识库。

    这是 RAG 的最后两步：
    5. 检索（Retrieval）：找到最相关的文档块
    6. 返回结果（不在这里生成，让 Agent 的 LLM 来生成）

    Args:
        question: 用户的问题
        top_k: 返回几个最相关的文档块

    Returns:
        拼接好的相关文档内容
    """
    if not os.path.exists(CHROMA_DIR):
        return "知识库还没有建立，请先运行 build_index()"

    index = load_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    if not nodes:
        return "在知识库中没有找到相关内容。"

    results = []
    for i, node in enumerate(nodes, 1):
        score = node.score if node.score else 0
        results.append(f"【片段{i}】（相关度: {score:.2f}）\n{node.text}")

    return "\n\n---\n\n".join(results)


if __name__ == "__main__":
    print("正在构建知识库索引...")
    build_index()

    print("\n测试查询：")
    questions = [
        "什么是 ReAct 框架？",
        "LangChain 和 LlamaIndex 有什么区别？",
        "RAG 的工作流程是什么？",
    ]
    for q in questions:
        print(f"\n问题：{q}")
        result = query_knowledge(q)
        print(result[:300])
        print("...")
