import os
import requests
from pinecone import Pinecone, ServerlessSpec
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
os.environ.get(PINECONE_API_KEY)

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API keys missing! Check your .env file.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "web-scraper-index"

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  
        metric="cosine",  # Change metric if needed
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region if necessary
    )

index = pc.Index(INDEX_NAME)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def scrape_and_store(url: str, base_url: str):
    """Scrapes documentation, chunks text, and stores embeddings in Pinecone."""
    headers = {"User-Agent": "Mozilla/5.0"}
    visited_urls = set()

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

                # Generate embeddings and store in Pinecone
                for i, chunk in enumerate(chunks):
                    vector = embedding_model.embed_documents([chunk])[0]
                    index.upsert(vectors=[(f"{norm_url}-{i}", vector, {"text": chunk})])

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

