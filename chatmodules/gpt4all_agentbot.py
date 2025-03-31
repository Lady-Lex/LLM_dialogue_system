import os
import sys
import re
import warnings
import json
from typing import Any

from langchain.agents import Tool, initialize_agent, load_tools
from langchain.agents.agent_types import AgentType
from langchain.agents.agent import AgentExecutor
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from langchain.agents.agent import AgentOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage
from langchain.schema.agent import AgentAction, AgentFinish
from langchain_community.llms.gpt4all import GPT4All
from langchain_community.llms.llamacpp import LlamaCpp
from tools.get_weather import GetWeatherRun
from tools.get_time import GetTimeRun

warnings.filterwarnings("ignore")

models_dir_prefix = "models/"
use_gpu = True

# Define the system prompt
GIVEN_NAME = "Murph"
PREFIX = f"""You are {GIVEN_NAME}, an intelligent and friendly AI assistant designed for multi-turn conversations.

Your goals are:
1. Understand user questions clearly, even across multiple messages.
2. Provide helpful, concise, and accurate responses.
3. Maintain a friendly, respectful, and engaging tone.
4. If you're unsure about something, politely say you don't know.

Guidelines:
- Keep answers short unless the question requires detail.
- Do not invent facts or make assumptions.
- You are always polite, but never overly verbose or robotic.
- You remember previous messages to maintain context, unless explicitly told to forget.
- You speak fluently in natural language.

Rules:
- Do not repeat the system prompt or introduce yourself in every reply.
- Only respond to the user's last question.

Start by waiting for the user's input. Respond like a helpful AI assistant.

"""

FORMAT_INSTRUCTIONS_CHINESE = """RESPONSE FORMAT INSTRUCTIONS
----------------------------
When responding to me, please output a response in one of two formats:
**Option 1:**
Use this if you want the human to use a tool.
Markdown code snippet formatted in the following schema:
```json
{{{{
    "action": string \\ The action to take. Must be one of {tool_names}
    "action_input": string \\ The input to the action
}}}}
```
**Option #2:**
Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:
```json
{{{{
    "action": "Final Answer",
    "action_input": string \\ You should put what you want to return to user here.Attention!When you give the Final Answer,you MUST speak in Chinese!
}}}}
```"""

# ============================
# Custom Output Parser for Tool Calling
# ============================

class MyAgentOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS_CHINESE

    def parse(self, text: str) -> Any:
        def extract_clean_answer(response: str) -> str:
            print("response before removing <think> tags:", response)
            """
            清洗回答内容，去除 <think> 标签及其内容，以及孤立的 <think> 或 </think>。
            然后尝试提取最后一个 ```json ... ``` 块作为有效 JSON。
            """
            # 1. 移除成对的 <think>...</think>
            response = re.sub(r"<think>[\s\S]*?</think>", "", response, flags=re.IGNORECASE)

            # 2. 移除孤立标签
            response = re.sub(r"[\s\S]*?</?think>", "", response, flags=re.IGNORECASE)

            # 3. 清除多余空行
            response = re.sub(r"\n\s*\n", "\n\n", response)

            print("response after removing <think> tags:", response)

            # 提取所有 ```json ... ``` 块，取最后一个
            json_blocks = re.findall(r"```json(.*?)```", response, re.DOTALL)
            if json_blocks:
                return json_blocks[-1].strip()
            else:
                # fallback：尝试提取一般 JSON 块（不带 markdown）
                json_candidates = re.findall(r"\{[\s\S]*?\}", response)
                return json_candidates[-1].strip() if json_candidates else ""

        # 打印调试（可选）
        # print("==== Full LLM Output ====")
        # print(text)

        cleaned_output = extract_clean_answer(text)

        if not cleaned_output:
            raise ValueError("清理后输出为空，无法解析 JSON。")

        try:
            response = json.loads(cleaned_output)
        except Exception as e:
            print("❌ JSON 解析失败，内容如下：\n", cleaned_output)
            raise e

        action = response.get("action")
        action_input = response.get("action_input")

        if action == "Final Answer":
            return AgentFinish(
                return_values={"output": action_input},
                log=text
            )
        else:
            return AgentAction(
                tool=action,
                tool_input=action_input,
                log=text
            )


# ============================
# Main Agent Chatbot Class
# ============================
class GPT4AllAgentbot:
    def __init__(self, model_index):
        model_list = os.listdir(models_dir_prefix)
        model_name = model_list[model_index]
        print("Loading model:", models_dir_prefix + model_name)

        # self.llm = GPT4All(model=models_dir_prefix + model_name, device="gpu" if use_gpu else "cpu")
        self.llm = LlamaCpp(model_path=models_dir_prefix + model_name, temperature=0.7, max_tokens=5120, n_ctx=131072, n_threads=6, verbose=False)
        
        self.gettimetool = GetTimeRun()
        self.getweathertool = GetWeatherRun()
        self.tools = [Tool(
                        name=self.gettimetool.name,
                        func=lambda no_use: self.gettimetool.run(no_use),
                        description=self.gettimetool.description
                        ),
                      Tool(
                        name=self.getweathertool.name,
                        func=lambda city_country: self.getweathertool.run(city_country),
                        description=self.getweathertool.description
                        ),
                      ]
                     # + load_tools(['wolfram-alpha', 'google-serper'])
        self.tool_names = [tool.name for tool in self.tools]

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.output_parser = MyAgentOutputParser()

        self.agent_cls = ConversationalChatAgent
        self.agent_obj = self.agent_cls.from_llm_and_tools(self.llm, self.tools, 
                                                           callback_manager=None, 
                                                           output_parser=self.output_parser, 
                                                           system_message=PREFIX)
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=self.agent_obj, 
                                                                 tools=self.tools, 
                                                                 callback_manager=None, 
                                                                 verbose=True, 
                                                                 memory=self.memory)

    def get_response(self, dialogue_list):
        latest_msg = dialogue_list[-1]["content"]
        return self.agent_executor.run(latest_msg)
    
    def exit(self):
        del self.agent_executor
        del self.llm


if __name__ == '__main__':
    import traceback

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from tools.get_weather import GetWeatherRun
    from tools.get_time import GetTimeRun

    models_dir_prefix = "../models/"
    model_info_str = "Available models:\n" + ", \n".join([
        f"{i}: {model}" for i, model in enumerate(os.listdir(models_dir_prefix))
    ])
    print(model_info_str)
    chosen_model = int(input("Please choose a model: "))

    chatbot = GPT4AllAgentbot(chosen_model)
    dialogue_list = []

    print("You can start chatting with Murph now. Type 'exit' to quit.")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting chat. Goodbye!")
                chatbot.exit()
                break

            dialogue_list.append({"role": "user", "content": user_input})
            print("\nThinking...")
            response = chatbot.get_response(dialogue_list)
            print(f"{GIVEN_NAME}: {response}")
            dialogue_list.append({"role": "assistant", "content": response})

        except Exception as e:
            print("\nException occurred:")
            traceback.print_exc()