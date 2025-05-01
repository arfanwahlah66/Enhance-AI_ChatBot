# NLP and Deep Learning Libraries
import numpy as np
import tensorflow as tf
import tflearn
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Utility Libraries
import json
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess intents
def load_intents(file_path='intents.json'):
    with open(file_path) as file:
        return json.load(file)

def preprocess_intents(intents, ignore_words=['?']):
    words, classes, documents = [], [], []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = sorted(set([stemmer.stem(w.lower()) for w in words if w not in ignore_words]))
    classes = sorted(list(set(classes)))
    return words, classes, documents

def create_training_data(words, classes, documents):
    training, output_empty = [], [0] * len(classes)
    for doc in documents:
        bag = [1 if w in [stemmer.stem(word.lower()) for word in doc[0]] else 0 for w in words]
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training)
    return list(training[:, 0]), list(training[:, 1])

# Build neural network model
def build_model(input_size, output_size):
    tf.compat.v1.reset_default_graph()
    net = tflearn.input_data(shape=[None, input_size])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, output_size, activation='softmax')
    net = tflearn.regression(net)
    return tflearn.DNN(net)

# Clean user sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [stemmer.stem(word.lower()) for word in sentence_words]

# Bag of Words conversion
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    return np.array([1 if w in sentence_words else 0 for w in words])

# Classification
def classify(sentence, model, words, classes):
    ERROR_THRESHOLD = 0.25
    probabilities = model.predict([bow(sentence, words)])[0]
    results = [[i, p] for i, p in enumerate(probabilities) if p > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [(classes[r[0]], r[1]) for r in results]

# Generate response
def get_response(sentence, intents, model, words, classes):
    intents_list = classify(sentence, model, words, classes)
    if intents_list:
        for intent_tag, _ in intents_list:
            for intent in intents['intents']:
                if intent['tag'] == intent_tag:
                    return random.choice(intent['responses'])
    return "I'm not quite sure how to respond to that. 🤖"

# Main function to train/load and chat
def main():
    if os.path.exists("training_data"):
        print("[INFO] Loading existing training data and model...")
        data = pickle.load(open("training_data", "rb"))
        words = data['words']
        classes = data['classes']
        train_x = data['train_x']
        train_y = data['train_y']
        model = build_model(len(train_x[0]), len(train_y[0]))
        model.load("model.tflearn")
    else:
        print("[INFO] Processing intents and training new model...")
        intents = load_intents()
        words, classes, documents = preprocess_intents(intents)
        train_x, train_y = create_training_data(words, classes, documents)
        model = build_model(len(train_x[0]), len(train_y[0]))
        model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")
        pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))

    intents = load_intents()
    print("Start chatting with FreeBirdsBot! (type 'exit' to quit)\n")
    while True:
        inp = input("You: ").strip()
        if inp.lower() in ['exit', 'quit']:
            print("FreeBirdsBot: Goodbye! 👋")
            break
        response = get_response(inp, intents, model, words, classes)
        print("FreeBirdsBot:", response)

if __name__ == "__main__":
    main()
