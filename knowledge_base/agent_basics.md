# Agent 开发基础知识

## 什么是 AI Agent
AI Agent（智能体）是一种能够自主感知环境、做出决策并采取行动的人工智能系统。与传统的 LLM 对话不同，Agent 具备以下核心能力：
- **工具调用**：Agent 可以使用搜索引擎、数据库、API 等外部工具
- **自主规划**：Agent 能够将复杂任务分解为多个子步骤
- **记忆管理**：Agent 可以记住之前的对话和操作结果
- **循环推理**：Agent 通过 Think → Act → Observe 循环不断推进任务

## ReAct 框架
ReAct（Reasoning + Acting）是目前最流行的 Agent 架构：
1. **Think**：LLM 分析当前状态，决定下一步行动
2. **Act**：调用工具执行具体操作
3. **Observe**：获取工具返回结果
4. 重复以上步骤直到任务完成

## 主流 Agent 开发框架（2025年）
- **LangChain / LangGraph**：最成熟的 Agent 框架，生态丰富，支持复杂工作流
- **LlamaIndex**：专注于 RAG 和数据索引，适合知识密集型应用
- **AutoGen**：微软开源，多 Agent 协作框架
- **CrewAI**：专注于多 Agent 角色扮演和协作
- **Dify**：低代码 Agent 平台，适合快速搭建

## Tool Calling（工具调用）
LLM 通过 Function Calling 机制来使用工具：
1. 开发者定义工具的名称、描述、参数格式
2. LLM 在回复中输出 tool_calls（JSON 格式）
3. 程序解析 tool_calls 并执行对应函数
4. 将执行结果返回给 LLM 继续推理
