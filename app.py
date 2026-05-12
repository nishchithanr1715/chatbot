import os
from flask import Flask, render_template, request, jsonify, session
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

ENDPOINT = "https://models.github.ai/inference"
MODEL = "deepseek/DeepSeek-V3-0324"
TOKEN = os.environ.get("GITHUB_TOKEN", "")

SYSTEM_PROMPT = (
    "You are a helpful, friendly, and intelligent AI assistant. "
    "Answer questions clearly and concisely. "
    "Be conversational and engaging."
)

client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)


def build_messages(history: list, user_message: str) -> list:
    """Build the message list for the API call."""
    messages = [SystemMessage(SYSTEM_PROMPT)]
    for turn in history:
        messages.append(UserMessage(turn["user"]))
        messages.append(AssistantMessage(turn["assistant"]))
    messages.append(UserMessage(user_message))
    return messages


@app.route("/")
def index():
    session.setdefault("history", [])
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    history = session.get("history", [])

    try:
        messages = build_messages(history, user_message)
        response = client.complete(
            messages=messages,
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
            model=MODEL,
        )
        assistant_reply = response.choices[0].message.content

        history.append({"user": user_message, "assistant": assistant_reply})
        # Keep last 20 turns to avoid token overflow
        session["history"] = history[-20:]

        return jsonify({"reply": assistant_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear():
    session["history"] = []
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
