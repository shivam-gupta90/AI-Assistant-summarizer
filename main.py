from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from openai import OpenAI

app = Flask(__name__)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Act like a helpful personal assistant."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.7,
        max_tokens=512
    )

    answer = response.choices[0].message.content

    return jsonify({"response": answer})


@app.route("/summarize", methods=["POST"])
def summarize():
    email_text = request.form.get("email")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an expert email assistant."
            },
            {
                "role": "user",
                "content": f"Summarize the following email in 2-3 sentences:\n\n{email_text}"
            }
        ],
        temperature=0.3,
        max_tokens=512
    )

    summary = response.choices[0].message.content

    return jsonify({"summary": summary})


if __name__ == "__main__":
    app.run(debug=True)