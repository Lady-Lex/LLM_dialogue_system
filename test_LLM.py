from gpt4all import GPT4All


if __name__ == '__main__':
    model = GPT4All(model_name='Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf', device="gpu")

    print("You can start chatting now. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        response = model.generate(user_input)
        print(f"model: {response}")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting chat. Goodbye!")
            chatbot.exit()
            break
