# RAGadd AI - Document Chat Assistant

![RAGadd AI](static/RAGadd.png)

## 🚀 Overview

**RAGadd AI** is an advanced document chatbot that allows you to upload documents and chat with them using state-of-the-art Retrieval Augmented Generation (RAG) technology. Built with Flask, LangChain, and integrated with multiple AI providers, RAGadd AI transforms your documents into an interactive knowledge base.

### ✨ Key Features

* 📄 **Multi-format Document Support** - PDF, DOCX, TXT, CSV, Excel, PowerPoint, HTML, JSON, Python files and more
* 🤖 **Advanced RAG Pipeline** - Powered by LangChain and FAISS vector database
* 💬 **Intelligent Chat Interface** - Natural conversation with your documents
* 🎨 **Modern UI/UX** - Clean, responsive design with dark/light theme support
* 🔒 **Secure API Integration** - API keys stored in `.env` file, not in frontend
* ⚡ **Multiple Model Providers**

  * **Fast (API-based)**: Groq, OpenRouter
  * **Offline (Local models)**: Ollama, LMStudio
* 🔧 **Model Customization** - Change models, temperature, and providers easily in `config.ini`
* 📱 **Mobile Responsive** - Works seamlessly on all devices
* 💾 **Chat Export** - Download your conversations as text files

---

## 🛠️ Technology Stack

* **Backend**: Flask, Python
* **AI/ML**: LangChain, FAISS, HuggingFace Embeddings, LangGraph
* **Document Processing**: PyPDF, python-docx, unstructured
* **Frontend**: HTML5, CSS3, JavaScript
* **AI Providers**: Groq, OpenRouter, Ollama, LMStudio

---

## 📋 Prerequisites

Before installation, ensure you have:

* Python 3.8 or higher
* pip (Python package installer)
* API keys for Groq and/or OpenRouter (if using **fast mode**)

---

## ⚡ Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/HiMahendraBeniwal/RAGadd-app.git
cd RAGadd-app
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add API Keys in `.env`

Create a `.env` file in the project root:

```ini
# .env
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

⚠️ Do **not** add API keys inside the website. They are securely read from `.env`.

### 6. Run the Application

```bash
python main.py
```

### 7. Access the Application

```
http://localhost:5000
```

---

## 🔧 Configuration

### Model Customization (`config.ini`)

You can fully control which models are used by editing `config.ini`.

Example configuration:

```ini
[llms_openrouter]
source = openrouter
model = deepseek/deepseek-r1:free
temperature = 0.9
site_url = http://localhost
site_name = MyApp

[llms_groq]
source = groq
model = moonshotai/kimi-k2-instruct
temperature = 0.0

[llms_ollama]
source = ollama
model = qwen3:4b
temperature = 0.0

[llms_lmstudio]
source = lmstudio
model = qwen/qwen3-4b-2507
temperature = 0.0

[embedding_model]
embedding_model = Qwen/Qwen3-Embedding-0.6B


[offline-order]
order = lmstudio, ollama

[fast-order]
order = groq, openrouter

[default-order]
order = groq, lmstudio, ollama, openrouter

[mode]
order = default

```

* Switch models by editing `model` under the respective provider check model avaialble from providers
* Adjust temperature for creativity
* Choose which providers are used in different modes

---

### Switching Between Modes

You can toggle **modes** directly from the frontend:

* **Fast Mode (API-based)** → Uses Groq & OpenRouter for instant responses
* **Offline Mode (Local models)** → Uses Ollama & LMStudio (no internet required)

Default order is controlled via `[mode]` in `config.ini`.

---

## 📁 Supported File Formats

| Format       | Extension               | Description              |
| ------------ | ----------------------- | ------------------------ |
| PDF          | `.pdf`                  | Portable Document Format |
| Word         | `.docx`, `.doc`         | Microsoft Word documents |
| Text         | `.txt`                  | Plain text files         |
| Spreadsheet  | `.xlsx`, `.xls`, `.csv` | Excel and CSV files      |
| Presentation | `.pptx`, `.ppt`         | PowerPoint presentations |
| Web          | `.html`, `.htm`         | HTML documents           |
| Code         | `.py`, `.js`, `.css`    | Programming files        |
| Data         | `.json`                 | JSON data files          |

---

## 🚀 Deployment

### Local Development

```bash
python main.py
```

### Production Deployment

**Using Waitress (Recommended for Windows):**

```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

**Using Gunicorn (Linux/Mac):**

```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 app:app
```

---

## 🔒 Security Notes

* API keys are stored securely in `.env` (never exposed in frontend)
* Uploaded documents are processed temporarily and cleaned up automatically
* No documents are permanently stored on the server
* All communication uses secure HTTPS protocols

---

## 🛠️ Troubleshooting

### Common Issues

**API Key Not Working (Fast mode):**

* Verify Groq/OpenRouter API key is in `.env`
* Ensure account has sufficient credits (for OpenRouter)
* Stable internet connection is required

**Document Upload Fails:**

* Check file format is supported
* Ensure file size < 50MB
* Try uploading fewer files

**Chat Not Responding:**

* Ensure mode is set correctly (fast/offline)
* Verify models are configured in `config.ini`
* Refresh page and retry

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* [LangChain](https://langchain.com/) for the RAG framework
* [OpenRouter](https://openrouter.ai/) for API-based models
* [Groq](https://groq.com/) for ultra-fast inference
* [Ollama](https://ollama.com/) & [LMStudio](https://lmstudio.ai/) for local model support
* [FAISS](https://faiss.ai/) for vector similarity search
* [HuggingFace](https://huggingface.co/) for embeddings

---

<div align="center">
  <img src="static/Logo.png" alt="RAGadd AI" width="100"/>
  <br>
  <strong>RAGadd AI - Transform Your Documents into Interactive Knowledge</strong>
  <br>
  
</div>

---

