# 智能调研助手

基于 LangChain + LlamaIndex + DeepSeek 的 AI Agent，输入一个话题，自动搜索互联网和本地知识库，生成结构化调研报告。

## 技术栈

- **LangChain / LangGraph** — Agent 编排和工具调用
- **LlamaIndex** — 本地知识库 RAG 检索
- **DeepSeek API** — LLM 大语言模型
- **ChromaDB** — 向量数据库
- **DuckDuckGo** — 互联网搜索

## 功能

- 互联网搜索（DuckDuckGo）
- 本地知识库检索（LlamaIndex RAG）
- 结构化报告生成（引言 / 内容 / 结论）
- 本地文件读取

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key

# 3. 构建知识库索引
python rag.py

# 4. 运行 Agent
python agent.py
```

## 项目结构

```
research_agent/
├── agent.py              # Agent 主程序
├── tools.py              # 自定义工具（保存报告、读文件、知识库检索）
├── rag.py                # RAG 模块（LlamaIndex 知识库构建和查询）
├── knowledge_base/       # 知识库文档（可自行添加）
├── requirements.txt      # Python 依赖
├── .env.example          # 环境变量模板
└── README.md
```

## 添加知识库文档

将 `.md` 或 `.txt` 文件放入 `knowledge_base/` 文件夹，然后重新运行 `python rag.py` 重建索引即可。
