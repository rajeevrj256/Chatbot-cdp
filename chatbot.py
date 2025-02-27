import os
import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables securely
load_dotenv()
api_key =os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY. Ensure it's set in .env.")

# Initialize ChromaDB
os.environ["OPENAI_API_KEY"] = api_key
DB_DIR = os.path.abspath("chroma_db")
chroma_client = chromadb.PersistentClient(DB_DIR)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store and retriever
vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant docs

# Define structured prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "You are an intelligent AI chatbot that provides clear, structured responses.\n\n"
        "Context: {context}\n"
        "User Query: {query}\n\n"
        "Based on the given context, provide a well-structured and informative answer."
    ),
)

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Setup Retrieval-based QA Chain
qa_chain = load_qa_chain(llm=llm, chain_type="stuff")

def user_query(query: str) -> str:
    """Processes user query using retrieval-augmented generation."""
    docs = retriever.get_relevant_documents(query)  # Retrieve relevant documents
    response = qa_chain.run(input_documents=docs, question=query)  # Run LLM on retrieved docs
    return response

if __name__ == "__main__":
    query = "What is ChromaDB?"
    print(user_query(query))