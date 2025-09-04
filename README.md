# 📄 Ask Your PDF with Google Gemini + Streamlit  

A simple Streamlit app that lets you **upload PDFs, process them with Google Gemini embeddings, and ask natural language questions**.  
Built using **LangChain, FAISS, and Google Generative AI**.  

---

## 🚀 Features  
- Upload any PDF  
- Extract text & split into chunks  
- Generate embeddings with **Gemini (`models/embedding-001`)**  
- Store & search vectors with **FAISS**  
- Ask questions and get **context-aware answers**  
- Simple **Streamlit interface**  

---

## 📹 Demo  

https://github.com/vijaypatange21/Ask_your_pdf/blob/main/pdf%20summariser%20video.mp4

## ⚙️ Installation  

```bash
# Clone the repo
git clone https://github.com/your-username/ask-your-pdf-gemini.git
cd ask-your-pdf-gemini

# Create a virtual environment
python -m venv myenv
source myenv/bin/activate   # (Linux/Mac)
myenv\Scripts\activate      # (Windows)
```
## 🔑 Environment Variables

Create a `.env` file and add your Google API key:

`GOOGLE_API_KEY=your_google_api_key_here`

## ▶️ Run the App
```bash
streamlit run app.py
```
Then open: `http://localhost:8501`

## 🛠️ Tech Stack

-> Streamlit
-> LangChain
-> FAISS
-> Google Gemini API

## 📌 Next Steps

Multi-PDF support

Chat history

Better UI & UX

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.
# Install dependencies
pip install -r requirements.txt
