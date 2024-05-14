from flask import Flask, request, render_template, jsonify
from scipy import spatial 
import ast  
from openai import OpenAI 
import os
import pandas as pd
import re
import tiktoken
import json

REVIEWS_FILE = 'reviews.json'

client = OpenAI(
    api_key= "sk-proj-z18dQbDp7j0g3WsDtCrBT3BlbkFJ8iFn2gviIhJ5YaeGeWrV",
)

embeddings_path = "emdeddings_dataset.csv"
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"


df = pd.read_csv(embeddings_path)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below information from the Sizzlin Sahara Steakhouse Restaurant. Answer as a virtual assistance for the restaurant. Try your best to answer all the questions using the provided information. If the answer cannot be found in the info, write "I could not find a satisifcatory answer for your question. Please, contact our Customer Service Assistant, Emelda, on +263 77 334 4079 or visit our website (https://sizzlinsahara.com) for more information."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nINFORMATION FOR Sizzlin Sahara Steakhouse Restaurant:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Sizzlin Sahara Steakhouse Restaurant."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for reservation page
@app.route('/templates/reservation.html')
def reservation():
    return render_template('reservation.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({'error': 'Question parameter is required'}), 400

    # You can add your logic here to handle the question
    response = {'answer': ask(query=question)
    }

    return jsonify(response), 200

def load_reviews():
    if os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE, 'r') as file:
            return json.load(file)
    return []

def save_reviews(reviews):
    with open(REVIEWS_FILE, 'w') as file:
        json.dump(reviews, file, indent=4)

@app.route('/submit_review', methods=['POST'])
def submit_review():
    data = request.get_json()
    review = data.get('review')

    if not review:
        return jsonify({'error': 'Review text is required'}), 400

    reviews = load_reviews()
    reviews.append({'review': review})

    save_reviews(reviews)
    
    return jsonify({'message': 'Review submitted successfully'}), 200
if __name__ == '__main__':
    app.run(debug=True)
