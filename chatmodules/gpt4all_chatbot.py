import os
import re
import warnings
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import AIMessage
from langchain_community.llms.gpt4all import GPT4All
from langchain_community.llms.llamacpp import LlamaCpp

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

system_prompt = SystemMessagePromptTemplate.from_template(PREFIX)
# answer_prompt = GIVEN_NAME + ", please answer directly and helpfully:\n"
answer_prompt = ", Answer:\n"

prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("User: {question}\n\n" + answer_prompt)
])

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

class GPT4AllChatbot:
    def __init__(self, model_index):
        model_list = os.listdir(models_dir_prefix)
        model_name = model_list[model_index]
        print("Loading model:", models_dir_prefix + model_name)

        # self.llm = GPT4All(model=models_dir_prefix + model_name, device="gpu" if use_gpu else "cpu")
        self.llm = LlamaCpp(model_path=models_dir_prefix + model_name, temperature=0.7, max_tokens=5120, n_ctx=131072, n_threads=6, verbose=False)
        self.chain = LLMChain(llm=self.llm, prompt=prompt, memory=memory)

    def get_response(self, dialogue_list):
        result = self.chain.invoke({"question": dialogue_list[-1]["content"]})

        # 从 memory 返回中提取 AI 回复
        for msg in reversed(self.chain.memory.chat_memory.messages):
            if isinstance(msg, AIMessage):
                # return msg.content
                return self.extract_clean_answer(msg.content)

        return ""

    def extract_clean_answer(self, response: str) -> str:
        """
        清洗回答内容，去除 <think> 标签及其内容，以及孤立的 <think> 或 </think>。
        """
        # 1. 移除成对的 <think>...</think>
        response = re.sub(r"<think>[\s\S]*?</think>", "", response, flags=re.IGNORECASE)

        # 2. 移除孤立标签
        response = re.sub(r"[\s\S]*?</?think>", "", response, flags=re.IGNORECASE)

        # 3. 清除多余空行
        response = re.sub(r"\n\s*\n", "\n\n", response)

        return response.strip()

    def reset_memory(self):
        self.chain.memory.clear()

    def exit(self):
        del self.chain
        del self.llm


if __name__ == '__main__':
    models_dir_prefix = "../models/"
    model_info_str = "Available models:\n" + ", \n".join([
        f"{i}: {model}" for i, model in enumerate(os.listdir(models_dir_prefix))
    ])
    print(model_info_str)
    chosen_model = int(input("Please choose a model: "))

    chatbot = GPT4AllChatbot(chosen_model)
    dialogue_list = []

    print("You can start chatting with Murph now. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting chat. Goodbye!")
            chatbot.exit()
            break

        dialogue_list.append({"role": "user", "content": user_input})
        response = chatbot.get_response(dialogue_list)
        print(f"{GIVEN_NAME}: {response}")
        dialogue_list.append({"role": "assistant", "content": response})