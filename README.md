# 🛠️ Support Agent Chatbot for CDP  

## 📌 Overview  
The **Support Agent Chatbot** is an AI-powered chatbot designed to answer "how-to" questions related to four major **Customer Data Platforms (CDPs)**:  
- **Segment**  
- **mParticle**  
- **Lytics**  
- **Zeotap**  

It extracts information from official documentation, processes user queries, and provides accurate responses. The chatbot supports **advanced "how-to" queries** and **cross-CDP comparisons** using AI-driven retrieval techniques.

🔗 **Live Demo**: [CDP Chatbot](https://chatbot-cdp.onrender.com/)  
🔗 **GitHub Repository**: [Chatbot CDP](https://github.com/rajeevrj256/Chatbot-cdp)  

---

## 🚀 Features  

### ✅ **Core Functionalities**  
- **Answer "How-to" Questions**: Uses **retrieval-based AI** to fetch relevant documentation content.  
- **Extract Information from Documentation**:  
  - Scrapes CDP documentation using **BeautifulSoup (BS4)**.  
  - Chunks, embeds, and stores data in **Pinecone** for efficient retrieval.  
- **Handle Variations in Questions**:  
  - Uses **LangChain NLP** to process and understand different question formats.  
  - Implements **prompt engineering** to refine responses.  

### 🎯 **Bonus Features**  
- **Advanced "How-to" Questions**: Handles complex queries related to platform configurations and integrations.  
- **Cross-CDP Comparisons**:  
  - Implements an **AI agent ("CDP_Comparison")** that retrieves, compares, and presents differences dynamically.  
  - Uses **prompt engineering** to guide responses.  

---

## 🏗️ Tech Stack  

| Component         | Technology Used |
|------------------|----------------|
| **NLP Framework** | LangChain |
| **Vector Storage** | Pinecone |
| **LLM Model** | OpenAI (GPT) |
| **Scraping & Processing** | BeautifulSoup (BS4) |
| **Frontend** | Streamlit |
| **Deployment** | Render.com |

---

## ⚙️ Installation & Setup  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/rajeevrj256/Chatbot-cdp.git
cd Chatbot-cdp
pip install requirements.txt
pip insall -u langchain-community
streamlit run agent.py
```
### **2️⃣ Env File
```
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_ENV=us-east-1
```

```
