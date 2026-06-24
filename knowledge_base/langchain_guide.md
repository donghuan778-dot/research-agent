# LangChain 使用指南

## LangChain 是什么
LangChain 是一个用于开发 LLM 驱动应用的 Python 框架。它提供了标准化的接口来连接各种 LLM、工具和数据源。

## 核心组件
### ChatModel
ChatModel 是 LangChain 对各种 LLM API 的统一封装：
- `ChatOpenAI`：连接 OpenAI 或兼容 API（如 DeepSeek）
- `ChatAnthropic`：连接 Claude
- `ChatGoogleGenerativeAI`：连接 Gemini

### Tools（工具）
LangChain 定义工具有两种方式：
1. **@tool 装饰器**：最简单的方式，给函数加装饰器
2. **BaseTool 子类**：更灵活，适合复杂工具

使用 @tool 装饰器的例子：
```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式的结果"""
    return str(eval(expression))
```

### LangGraph
LangGraph 是 LangChain 的图编排引擎，用于构建复杂的 Agent 工作流：
- `create_react_agent`：快速创建 ReAct Agent
- 支持自定义状态图、条件分支、并行执行

## 与 LlamaIndex 的对比
| 特性 | LangChain | LlamaIndex |
|------|-----------|------------|
| 核心定位 | 通用 Agent 框架 | 数据索引和检索 |
| 工具生态 | 非常丰富 | 专注 RAG |
| 学习曲线 | 中等 | 较低 |
| 最佳场景 | 复杂工作流 | 知识库问答 |

## 最佳实践
1. 用 LangChain 做 Agent 编排和工具调用
2. 用 LlamaIndex 做文档索引和 RAG 检索
3. 两者结合使用效果最佳
