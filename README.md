# Local AI Assistant

[Try the live app](https://local-ai-assistant-by-snm.streamlit.app/)

A small Streamlit assistant for conversation, coding help, current web search, and question answering over uploaded documents. The model runs on the machine hosting the app. It does not use Ollama or a paid inference API.

## Model and how this app was built

This project does not train a language model from scratch. It uses Qwen2.5-Coder-1.5B-Instruct, an existing instruction-tuned model with 1.54 billion parameters and general conversation and coding capabilities. The app downloads the official Q4_K_M GGUF version from Hugging Face on first use. Quantization reduces the model file to about 1.12 GB so it can run with the CPU-based `llama.cpp` runtime. The model weights are not stored in this GitHub repository.

The model was not trained or fine-tuned in Jupyter. Jupyter is optional for a future fine-tuning project; it is not needed to run this app. The current project connects the pretrained model to a Streamlit chat interface through `llama-cpp-python`. Before every Assistant-mode reply, it tries a web search, gives available results to the model, and displays source links. If search fails or finds nothing, the local model answers without current web information. Search uses the real `ddgs` package. The RAG mode extracts text from an uploaded document, splits it into overlapping passages, ranks passages by BM25 keyword relevance, and gives the best matches to the same model with source labels. Retrieval uses a small built-in implementation and does not download a separate embedding model. No training dataset or notebook is required to run the current app.

The base model is published under Apache-2.0. Review the [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) and [official GGUF repository](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF) for model details and terms.

## Deploy from GitHub to Streamlit Community Cloud

1. Push this repository to GitHub. Keep virtual environments and model weights out of the repository; `.gitignore` excludes local environments and secrets.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app from the GitHub repository.
3. Select the branch and `app.py` as the app file.
4. In advanced settings, select Python 3.12, then deploy.
5. The first request downloads the model. Allow time and disk space for the roughly 1.1 GB download.

In the app, select **RAG** and upload a PDF, DOCX, TXT, Markdown, or CSV file to ask questions about it. Uploads are limited to 10 MB and 250,000 extracted characters. Retrieved passages are shown with each answer so you can verify the citations. Scanned/image-only PDFs are not supported because this version does not include OCR. Since retrieval uses keyword relevance, rephrase a question with terms from the document if no relevant passage is found. The model can still make mistakes; verify the quoted passages for exact wording.

## Context window

The sidebar context setting ranges from 2,048 to 8,192 tokens, with 4,096 as the default. The model's input budget grows with that setting while reserving room for a response. In RAG mode, larger contexts can also include more retrieved passages. A larger context does not make the model more knowledgeable or guarantee more accurate answers. It uses more memory and can slow generation; on Streamlit Community Cloud it may exceed the app's memory allocation. Changing the setting reloads the model configuration, while the downloaded model file remains cached. The app keeps recent chat history until the selected input budget is reached, warns as the conversation approaches that limit, and offers a **Continue answer** button when a reply reaches its output limit.

Uploaded documents are processed by the machine hosting the app and their text is included in model prompts. Do not upload sensitive files to a public deployment.

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

The model is cached by Hugging Face after its first download. In Assistant mode, every request tries web search before the local model responds. When search is unavailable or returns no results, the local model answers without current web information. RAG document processing and answering use the local model, but the first model download needs internet access.

## Modes and limitations

- **Assistant:** handles conversation, coding, and web search in one mode. It tries the web first for every request, including coding and ordinary conversation, then uses the local model. If search is unavailable or has no results, it explains that and answers from the local model; those answers may not include current information.
- **RAG:** answers using passages retrieved from the uploaded file and displays those passages for checking.

Search results help with current information but do not guarantee that generated code is correct or identical to an example. Check cited documentation and review code before using it. The model is small, so it may struggle with complex or lengthy tasks.

