import os
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_community.llms import GPT4All

import warnings
warnings.filterwarnings("ignore")


# Define the system prompt
GIVEN_NAME = "Murph"

PREFIX = f"""{GIVEN_NAME} is a large language model trained by nomic-ai.

{GIVEN_NAME} is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, {GIVEN_NAME} is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

{GIVEN_NAME} is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, {GIVEN_NAME} is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, {GIVEN_NAME} is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, {GIVEN_NAME} is here to assist."""

system_prompt = PREFIX

models_dir_prefix = "models/"

use_gpu = True

prompt = ChatPromptTemplate.from_messages(
    [
        system_prompt,
        MessagesPlaceholder(variable_name="chat_history"),
        # HumanMessagePromptTemplate.from_template("{question}")
        HumanMessagePromptTemplate.from_template("Question: {question}\n\nPlease answer directly and don't say anything as Human. \nAnswer:")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


class GPT4AllChatbot:
    def __init__(self, model_index):
        model_list = os.listdir(models_dir_prefix)
        model_name = model_list[model_index]
        print("Loading model:", models_dir_prefix + model_name)

        self.llm = GPT4All(model=models_dir_prefix + model_name, device="gpu" if use_gpu else "cpu")

        self.chain = LLMChain(llm=self.llm, prompt=prompt, memory=memory)

    def get_response(self, dialogue_list):
        result = self.chain.invoke({"question": dialogue_list[-1]})
        # print(result)
        # result_content = self.extract_response(result["chat_history"][-1].content)
        result_content = result["chat_history"][-1].content
        return result_content

    def extract_response(self, text):
        start_marker = f"{GIVEN_NAME}: "
        end_marker = "Human: "

        start_index = text.find(start_marker)
        if start_index == -1:
            return text.strip()

        end_index = text.find(end_marker, start_index)
        if end_index == -1:
            return text[start_index + len(start_marker):].strip()
        else:
            return text[start_index + len(start_marker):end_index].strip()

    def reset_memory(self):
        memory.clear()

    def exit(self):
        # if we don't delete the chain and llm objects manually, the memory and GPU memory may not be released
        del self.chain
        del self.llm


if __name__ == '__main__':
    # excecute the chatbot and chat in the terminal
    models_dir_prefix = "../models/"

    model_info_str = "Available models: \n" + ", \n".join([f"{i}: {model}" for i, model in enumerate(os.listdir(models_dir_prefix))])
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

        dialogue_list.append(user_input)
        response = chatbot.get_response(dialogue_list)
        print(f"{GIVEN_NAME}: {response}")
        dialogue_list.append(response)
