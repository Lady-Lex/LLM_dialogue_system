import os
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
from langchain_community.llms import GPT4All, LlamaCpp
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
        cleaned_output = text.strip()
        if "```json" in cleaned_output:
            _, cleaned_output = cleaned_output.split("```json")
        if "```" in cleaned_output:
            cleaned_output, _ = cleaned_output.split("```")
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json") :]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[len("```") :]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[: -len("```")]
        cleaned_output = cleaned_output.strip()
        response = json.loads(cleaned_output)
        return {"action": response["action"], "action_input": response["action_input"]}

# ============================
# Main Agent Chatbot Class
# ============================
class GPT4AllAgentbot:
    def __init__(self, model_index):
        model_list = os.listdir(models_dir_prefix)
        model_name = model_list[model_index]
        print("Loading model:", models_dir_prefix + model_name)

        # self.llm = GPT4All(model=models_dir_prefix + model_name, device="gpu" if use_gpu else "cpu")
        self.llm = LlamaCpp(model_path=models_dir_prefix + model_name, temperature=0.7, max_tokens=512, n_ctx=2048, n_threads=6, verbose=False)
        
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

        # self.agent_executor: AgentExecutor = initialize_agent(
        #     tools=[self.weather_tool, self.time_tool],
        #     llm=self.llm,
        #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #     verbose=True,
        #     handle_parsing_errors=True,
        #     output_parser=MyAgentOutputParser()
        # )

    def get_response(self, dialogue_list):
        latest_msg = dialogue_list[-1]["content"]
        return self.agent_executor.run(latest_msg)

    def exit(self):
        del self.agent_executor
        del self.llm


if __name__ == '__main__':
    import traceback

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
            print("\n🤖 Thinking...")
            response = chatbot.get_response(dialogue_list)
            print(f"{GIVEN_NAME}: {response}")
            dialogue_list.append({"role": "assistant", "content": response})

        except Exception as e:
            print("\n🚨 Exception occurred:")
            traceback.print_exc()