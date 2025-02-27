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
        "You are an AI assistant Support Agent Chatbot for CDP documentation. You are providing structured responses.\n\n"
        "You can answer how-to questions related to four Customer Data Platforms (CDPs): Segment, mParticle, Lytics, and Zeotap.\n\n"
        "You The chatbot should be able to extract relevant information from the context and use it to provide answers of these CDPs to guide users on how to perform tasks or achieve specific outcomes within each platform\n\n."
        "You can answer questions about the following CDPs: Segment, mParticle, Lytics, and Zeotap.\n\n"
        "You can provide answers in markdown format.\n\n"
        "If information are not available in the context, you can provide a answe I don't know,you can't give an answer if context are not available.\n\n"
        "If the provided information is empty, say that you don't know the answer.\n\n"
        "If user asking same type of question and ans are not avialable in context, you can provide a answer that you don't know,you can't give an answer if context are not available.\n\n"
        "If user aking question which are based on comparison of two or more CDPs,you analysi each CDPs according to comparison content, and provide answer with more detailed in terms of comparsion based .\n\n"    
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

