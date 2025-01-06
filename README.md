# AI Chatbot with LangChain and Streamlit

This project demonstrates the creation of an AI-powered chatbot assistant using the LangChain library, Streamlit for the user interface, and Hugging Face embeddings and models. The chatbot processes PDF documents, creates vector embeddings, stores them in FAISS, and retrieves contextually relevant answers to user queries.

---

## Features

1. **PDF Document Processing**:
   - Load PDF files from a directory.
   - Split documents into manageable chunks for processing.

2. **Vector Embeddings**:
   - Generate embeddings using Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` model.
   - Store embeddings locally using FAISS for efficient retrieval.

3. **AI Chatbot**:
   - Powered by a Hugging Face endpoint.
   - Custom prompt templates for tailored responses.
   - Retrieval-based question answering.

4. **Streamlit Integration**:
   - Interactive chat interface.
   - Persistent chat history.

---

## Prerequisites

Ensure the following are installed:
- Python 3.8+
- Required Python libraries (see [Requirements](#requirements))
- Hugging Face account and token
- FAISS library

---

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/ai-chatbot.git
   cd ai-chatbot
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   - Create a `.env` file in the project root.
   - Add your Hugging Face token:
     ```env
     HF_TOKEN=your_huggingface_token
     ```

4. **Prepare PDF Data**:
   - Place your PDF files in the `data/` directory.

---

## Usage

### 1. Preprocess PDF Files
Run the script to load, split, and create vector embeddings:
```bash
python preprocess.py
```
This will generate and save a FAISS vectorstore in the `vectorstore/` directory.

### 2. Run the Chatbot
Start the chatbot application with Streamlit:
```bash
streamlit run app.py
```

Access the chatbot in your web browser at `http://localhost:8501`.

---

## Directory Structure
```
.
├── medical_bot.py               # Streamlit application for the chatbot
├── documents_vectorizer.py      # Preprocesses PDF files and generates vectorstore
├── data/                        # Directory for storing PDF files
├── vectorstore/                 # Directory for storing FAISS vectorstore
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (not included in repo)
└── README.md                    # Documentation
```

---

## Requirements

The project requires the following libraries:

- `streamlit`
- `langchain`
- `langchain-community`
- `langchain-huggingface`
- `langchain-core`
- `huggingface-hub`
- `sentence-transformers`
- `faiss-cpu`
- `PyPDF2`
- `python-dotenv`

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Notes
- The `HUGGINGFACE_REPO_ID` is set to `mistralai/Mistral-7B-Instruct-v0.3`. You can replace this with your desired model.
- The FAISS vectorstore uses `allow_dangerous_deserialization=True` for loading. Use this cautiously in production.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your improvements or suggestions.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- [LangChain](https://langchain.com/)
- [Hugging Face](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)

