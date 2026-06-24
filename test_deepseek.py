"""测试 DeepSeek API 是否能正常调用"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

response = llm.invoke("你好，请用一句话介绍你自己")
print("DeepSeek 返回：", response.content)
print("\n✅ API 调用成功！")
