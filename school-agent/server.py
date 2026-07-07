"""Moon Grammar School AI Agent — a small web app powered by Claude.

Run it with:  python server.py
Then open:    http://localhost:5000
"""

import os
from pathlib import Path

import anthropic
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


def _load_api_key():
    """Fall back to the saved Windows user environment variable if the
    key isn't in this process's environment (e.g. terminal opened before
    the key was saved with setx)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            val, _ = winreg.QueryValueEx(k, "ANTHROPIC_API_KEY")
            if val:
                os.environ["ANTHROPIC_API_KEY"] = val
    except (OSError, ImportError):
        pass  # not on Windows (e.g. running on Vercel) — rely on normal env vars


_load_api_key()
client = anthropic.Anthropic()  # reads the ANTHROPIC_API_KEY environment variable

HERE = Path(__file__).parent


def school_info() -> str:
    return (HERE / "school_info.md").read_text(encoding="utf-8")


def system_prompt(mode: str) -> str:
    prompts = {
        "school": (
            "You are the official AI admission counselor and support assistant "
            "for Moon Grammar School. You answer questions from parents, "
            "students, and visitors about the school, guide them through the "
            "admission process, and help parents choose the right class for "
            "their child. Only use the school information below. If something "
            "is not covered there, say you don't have that information and "
            "suggest contacting the school office or visiting the website "
            "https://munawar-one.vercel.app/ for more details. Keep answers short, warm, "
            "and clear. Be professional, patient, positive, and encouraging. "
            "When it fits naturally, warmly invite people to visit the campus "
            "during school hours. Never argue, never criticize other schools, "
            "and never discuss politics or religion.\n\n"
            "=== SCHOOL INFORMATION ===\n" + school_info()
        ),
        "homework": (
            "You are a patient, encouraging tutor for students of Moon Grammar "
            "School (ages roughly 5-16). Explain concepts step by step in simple "
            "words. Never just give away the final answer to homework — guide the "
            "student to work it out, ask small check-in questions, and celebrate "
            "progress. Match your language to a school student's level."
        ),
        "staff": (
            "You are a writing assistant for the teachers and office staff of "
            "Moon Grammar School. Help them write notices, letters to parents, "
            "report card comments, announcements, and emails in clear, polite, "
            "professional English. When details are missing, use sensible "
            "placeholders like [date] so staff can fill them in."
        ),
    }
    return prompts.get(mode, prompts["school"])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    mode = data.get("mode", "school")
    messages = data.get("messages", [])

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return jsonify(error="No API key set up yet! Get one at "
                             "https://console.anthropic.com, then run in PowerShell: "
                             'setx ANTHROPIC_API_KEY "your-key-here" '
                             "— and restart the app."), 401

    try:
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system_prompt(mode),
            messages=messages,
        )
    except anthropic.AuthenticationError:
        return jsonify(error="Your Anthropic API key is missing or wrong. "
                             "See the instructions in the terminal."), 401
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e.message).lower():
            return jsonify(error="Your Anthropic account is out of credits. "
                                 "Go to https://console.anthropic.com → Plans & Billing "
                                 "and add credits, then try again."), 402
        return jsonify(error=f"The request was rejected: {e.message}"), 400
    except anthropic.RateLimitError:
        return jsonify(error="Too many requests right now — wait a minute and try again."), 429
    except anthropic.APIStatusError as e:
        return jsonify(error=f"The AI service returned an error ({e.status_code}). Try again."), 502
    except anthropic.APIConnectionError:
        return jsonify(error="Could not reach the AI service — check your internet connection."), 502

    reply = next((b.text for b in response.content if b.type == "text"), "")
    return jsonify(reply=reply)


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("=" * 60)
        print("WARNING: No ANTHROPIC_API_KEY found!")
        print("The chat will not work until you set your API key.")
        print("1. Get a key at https://console.anthropic.com")
        print("2. In PowerShell run:")
        print('   setx ANTHROPIC_API_KEY "your-key-here"')
        print("3. Close this window, open a new one, and run me again.")
        print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
