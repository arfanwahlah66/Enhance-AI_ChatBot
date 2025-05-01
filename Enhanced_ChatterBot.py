from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import logging
import os

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)

# Create or reuse chatbot instance
chatbot = ChatBot(
    'FreeBirdsBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///freebirds_chatbot.db',
    logic_adapters=[
        'chatterbot.logic.BestMatch',
        'chatterbot.logic.MathematicalEvaluation'
    ],
    read_only=True  # Prevents the bot from learning in production use
)

# Training function
def train_chatbot():
    trainer = ListTrainer(chatbot)
    training_data = [
        "Hi", "Hello!",
        "How are you?", "I'm doing well, thank you!",
        "What are you doing?", "I'm here to help you with your questions.",
        "Tell me a joke", "Why did the computer go to therapy? It had too many bytes of emotional baggage.",
        "Bye", "Goodbye! Have a nice day."
    ]
    trainer.train(training_data)
    print("[INFO] Chatbot training completed.")

# Check if database exists; train only if not
if not os.path.exists("freebirds_chatbot.db"):
    train_chatbot()

# Main chat loop
print("Start chatting with FreeBirdsBot! (type 'exit' or 'quit' to end)")
while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("FreeBirdsBot: Goodbye! 👋")
            break
        elif user_input == "":
            print("FreeBirdsBot: Please say something so I can respond.")
            continue
        response = chatbot.get_response(user_input)
        print("FreeBirdsBot:", response)
    except (KeyboardInterrupt, EOFError):
        print("\nFreeBirdsBot: Chat ended. Bye!")
        break
