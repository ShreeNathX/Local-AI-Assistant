# Local AI Assistant

[Try the live app](https://local-ai-assistant-by-snm.streamlit.app/)

A small Streamlit assistant for conversation, coding help, and current web search. The model runs on the machine hosting the app. It does not use Ollama or a paid inference API.

## Model and how this app was built

This project does not train a language model from scratch. It uses Qwen2.5-Coder-1.5B-Instruct, an existing instruction-tuned model with 1.54 billion parameters and general conversation and coding capabilities. The app downloads the official Q4_K_M GGUF version from Hugging Face on first use. Quantization reduces the model file to about 1.12 GB so it can run with the CPU-based `llama.cpp` runtime. The model weights are not stored in this GitHub repository.

The model was not trained or fine-tuned in Jupyter. Jupyter is optional for a future fine-tuning project; it is not needed to run this app. The current project connects the pretrained model to a Streamlit chat interface through `llama-cpp-python`. For coding and web-search requests, it retrieves a small set of web results, gives their text to the model, and displays the source links. Search uses the real `ddgs` package. No training dataset or notebook is required to run the current app.

The base model is published under Apache-2.0. Review the [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) and [official GGUF repository](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF) for model details and terms.

## Deploy from GitHub to Streamlit Community Cloud

1. Push this repository to GitHub. Keep virtual environments and model weights out of the repository; `.gitignore` excludes local environments and secrets.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app from the GitHub repository.
3. Select the branch and `app.py` as the app file.
4. In advanced settings, select Python 3.12, then deploy.
5. The first request downloads the model. Allow time and disk space for the roughly 1.1 GB download.

Community Cloud runs the model on its Linux host, not on each visitor's computer. Its resource limits vary; the model plus runtime may exceed the memory available to an app, so a successful deployment cannot be guaranteed on every Community Cloud allocation. If it runs out of memory or is too slow, host the same app on a machine with more RAM or use the local setup below. See [Streamlit's resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-resources-and-limits).

## Run locally

Requirements: 64-bit Windows, macOS, or Linux; Python 3.10 or newer; 8 GB RAM recommended; and about 2 GB of free disk space. Internet is needed to install packages, download the model the first time, and search the web.

Create and activate a virtual environment:

```sh
python -m venv .slm-venv
```

Windows PowerShell:

```powershell
.\.slm-venv\Scripts\Activate.ps1
```

macOS or Linux:

```sh
source .slm-venv/bin/activate
```

Install the dependencies and start the app:

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

The model is cached by Hugging Face after its first download. Conversation and coding generation can then run offline; web search still needs internet access.

## Modes and limitations

- **Conversation:** responds using the local model without searching the web.
- **Coding:** searches for relevant documentation and provides source links with the generated code.
- **Web search:** searches the current question and answers using the retrieved results.

Search results help with current information but do not guarantee that generated code is correct or identical to an example. Check cited documentation and review code before using it. The model is small, so it may struggle with complex or lengthy tasks.
