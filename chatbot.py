import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as PineconeVectorStore  # Correct import
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables securely
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

if not api_key or not PINECONE_API_KEY or not PINECONE_ENV:
    raise ValueError("Missing API keys. Ensure OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENV are set in .env.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "web-scraper-index"

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
index = pc.Index(index_name)
# Connect to the Pinecone index
vector_store = PineconeVectorStore(index, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant docs

# Define structured prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "You are an AI assistant providing structured responses.\n\n"
        "Context: {context}\n"
        "User Query: {query}\n\n"
        "Based on the given context, provide a detailed yet concise answer."
    ),
)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Setup Retrieval-based QA Chain
qa_chain = load_qa_chain(llm=llm, chain_type="stuff")

def user_query(query: str) -> str:
    """Processes user query using retrieval-augmented generation from Pinecone."""
    docs = retriever.get_relevant_documents(query)  # Retrieve relevant documents
    response = qa_chain.run(input_documents=docs, question=query)  # Run LLM on retrieved docs
    return response

