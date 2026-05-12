import os
from flask import Flask, render_template, request, jsonify
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

ENDPOINT = "https://models.github.ai/inference"
MODEL    = "deepseek/DeepSeek-V3-0324"
TOKEN    = os.environ.get("GITHUB_TOKEN", "")

SYSTEM_PROMPT = (
    "You are a helpful, friendly, and intelligent AI assistant. "
    "Answer questions clearly and concisely. "
    "Be conversational and engaging."
)

client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data        = request.get_json()
    user_msg    = (data.get("message") or "").strip()
    history     = data.get("history", [])   # list of {role, content} from client

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # Build message list: system + history + new user turn
    messages = [SystemMessage(SYSTEM_PROMPT)]
    for turn in history[-20:]:              # cap to last 20 turns
        role    = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            messages.append(UserMessage(content))
        elif role == "assistant":
            messages.append(AssistantMessage(content))

    messages.append(UserMessage(user_msg))

    try:
        response = client.complete(
            messages=messages,
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
            model=MODEL,
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
