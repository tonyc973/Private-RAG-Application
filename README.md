# Local PDF Q&A — Chat with Your PDF, Privately and Offline

A privacy-focused AI app that lets you ask questions about any PDF document **locally**, saying goodbye to API calls to online LLM inference servers.

Built using:
- 🧠 [Mistral 7B Instruct (GGUF)](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
- ⚙️ `llama.cpp` via LangChain
- 🔍 FAISS for vector search
- 🤗 HuggingFace sentence embeddings
- 🎯 Streamlit for an intuitive UI

---

##  Demo

<p align="center">
  <img src="screenshots/demo-1.png" width="700"/>
  <br/>
  <i>Ask questions about any PDF, with local AI.</i>
</p>

---

##  Features

✅ 100% Local — no internet or API keys required  
✅ Ask anything about your PDF  
✅ Powered by open-source models  
✅ Fast and efficient search with FAISS  
✅ Easy and interactive Streamlit interface  
✅ Keeps your data private and local as nothing leaves your machine

---

## 🛠️ How to Run

```bash
# 1.Clone this repository
git clone https://github.com/yourusername/local-pdf-qa.git
cd local-pdf-qa

# 2.Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3.Install dependencies
pip install -r requirements.txt

# 4.Download the Mistral GGUF model
# Visit: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
# Download a quantized version like: mistral-7b-instruct-v0.1.Q4_K_M.gguf
# Place it somewhere on your machine and retain the path

# Make sure to update correctly the model_path in app.py:
# model_path="/your/path/to/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# 5.Run the Streamlit app
streamlit run app.py



