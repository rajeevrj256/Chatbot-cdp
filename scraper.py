import os
import requests
from pinecone import Pinecone, ServerlessSpec
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API keys missing! Check your .env file.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "web-scraper-index"

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

def scrape_and_store(start_url: str, base_url: str):
    """
    Scrapes the content from the developer quickstart page and its sidebar-linked pages.
    Only URLs under the /docs/ path (as found in the sidebar <nav id="hub-sidebar">)
    will be crawled.
    """
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": base_url,
}

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
            
            # Extract main content: try to use the <main> tag, otherwise fallback to <body>
            content_container = soup.find("main")
            if not content_container:
                content_container = soup.body
            content = [
                tag.get_text().strip() 
                for tag in content_container.find_all(["p", "li", "h2", "code"]) 
                if tag.get_text().strip()
            ]
            page_text = "\n".join(content)
            
            if page_text:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(page_text)
                for i, chunk in enumerate(chunks):
                    vector = embedding_model.embed_documents([chunk])[0]
                    index.upsert(vectors=[(f"{norm_url}-{i}", vector, {"text": chunk})])
                print(f"Stored {len(chunks)} chunks from {url}")
            
            # Locate the left sidebar navigation using the provided HTML structure.
            sidebar = soup.find("nav", class_="menu")
            if sidebar:
                # Find all <a> tags with an href attribute within the sidebar.
                links = sidebar.find_all("a", href=True)
                for link in links:
                    href = link.get("href")
                    full_url = urljoin(base_url, href)
                    # Process only documentation pages and avoid external or homepage URLs.
                    if full_url.startswith(urljoin(base_url, "/docs/")):
                        fetch_page(full_url)
            else:
                print(f"No sidebar found in {url}")
                
        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")

    fetch_page(start_url)
    print("Scraping and storage complete.")

if __name__ == "__main__":
    base_url = "https://segment.com"
    start_url = "https://segment.com/docs/"
    scrape_and_store(start_url, base_url)
