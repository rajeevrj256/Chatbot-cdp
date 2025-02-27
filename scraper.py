import os
import requests
import chromadb
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB
DB_DIR = "chroma_db"
chroma_client = chromadb.PersistentClient(DB_DIR)

def scrape_and_store(url: str, base_url: str):
    """Scrapes documentation, chunks text, and stores embeddings in ChromaDB."""
    headers = {"User-Agent": "Mozilla/5.0"}
    visited_urls = set()
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model, client=chroma_client)

    def normalize_url(url):
        parsed = urlparse(url)
        return parsed.scheme + "://" + parsed.netloc + parsed.path.lower()

    def fetch_page(url):
        norm_url = normalize_url(url)
        if norm_url in visited_urls:
            return
        visited_urls.add(norm_url)

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract text
            content = [tag.get_text().strip() for tag in soup.find_all(["p", "li", "h2", "code"]) if tag.get_text().strip()]
            page_text = "\n".join(content)

            if page_text:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(page_text)
                documents = [Document(page_content=chunk) for chunk in chunks]

                vector_store.add_documents(documents)
                print(f"Stored {len(chunks)} chunks from {url}")

            # Find internal links
            for link in soup.find_all("a", href=True):
                full_url = urljoin(base_url, link["href"])
                if full_url.startswith(base_url):
                    fetch_page(full_url)

        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")

    fetch_page(url)
    print("Scraping and storage complete.")

