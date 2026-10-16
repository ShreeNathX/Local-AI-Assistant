# SLM

SLM is a local chat application for conversation and coding, with optional live web search. It uses Ollama's documented chat client and the Qwen2.5 1.5B instruction model. The model runs on your machine; web search uses the real `ddgs` metasearch library and requires an internet connection.

This project is the assistant application, not a newly trained foundation model. A useful model trained from random weights needs a large licensed dataset and substantial compute. SLM uses an existing open model so it can answer questions and write code immediately.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Ollama installed and running
- About 1 GB of disk space for the default quantized model

## Setup

1. Install [Ollama](https://ollama.com/download) and start it.
2. In a terminal, download the model:

   ```sh
   ollama pull qwen2.5:1.5b-instruct
   ```

3. Create and activate a Python virtual environment in this project, then install the app dependencies:

   ```sh
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS or Linux:

   ```sh
   source .venv/bin/activate
   ```

   Then, on any platform:

   ```sh
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

4. Open the local address printed by Streamlit in your browser.

## Use

- **Conversation** answers general questions using the local model.
- **Coding** biases responses toward code and software development.
- **Web search** searches the internet for each message, gives the results to the model, and displays linked sources with the answer.

The sidebar lets you change the Ollama model and server address. The defaults can also be set with `SLM_MODEL` and `OLLAMA_HOST` environment variables. Search queries are sent to DDGS search providers; conversation messages are sent only to the local Ollama server.

## Troubleshooting

- If the model is unavailable, run `ollama pull qwen2.5:1.5b-instruct` and confirm Ollama is running.
- If the Ollama address differs from `http://localhost:11434`, change it in the sidebar or set `OLLAMA_HOST`.
- Web search needs internet access. Conversation and coding do not.
- The 1.5B model is intentionally small. It can make mistakes, especially on complex code or reasoning tasks; review generated code before using it.

## Main components

- `app.py`: chat interface, conversation context, model calls, and search results.
- `requirements.txt`: the three runtime dependencies.
