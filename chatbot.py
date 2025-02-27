import os
import json
import re
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# Load environment variables
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

# Connect to Pinecone vector store
vector_store = PineconeVectorStore.from_existing_index(index_name, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant docs

# Define LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# QA Chain for retrieving answers
qa_chain = load_qa_chain(llm=llm, chain_type="stuff")

# List of CDPs
cdps = ["segment", "mparticle", "lytics", "zeotap"]

# Comparison detection keywords
comparison_keywords = ["compare", "difference", "vs", "versus", "better than"]


def is_comparison_query(query: str) -> bool:
    """Detects whether a query is a comparison request."""
    return any(keyword in query.lower() for keyword in comparison_keywords)


def extract_cdps(query: str):
    """Identifies which CDPs are mentioned in the user query."""
    mentioned_cdps = [cdp for cdp in cdps if cdp.lower() in query.lower()]
    return mentioned_cdps if len(mentioned_cdps) >= 2 else []


def extract_dynamic_topic(query: str, mentioned_cdps):
    """
    Dynamically extracts the topic of comparison by removing CDP names and common comparison words.
    Example:
        Input: "Compare data retention policies of Segment and Lytics"
        Output: "data retention policies"
    """
    # Remove CDP names from the query
    for cdp in mentioned_cdps:
        query = query.replace(cdp, "")

    # Remove comparison keywords
    for keyword in comparison_keywords:
        query = query.replace(keyword, "")

    # Extract remaining words (likely to be the topic)
    topic_words = query.strip()
    return topic_words if topic_words else "general comparison"


def generate_sub_queries(cdps, topic):
    """Generates sub-queries for each CDP based on the extracted topic."""
    return {cdp: f"What is the {topic} of {cdp}?" for cdp in cdps}


def retrieve_information(sub_queries):
    """Retrieves answers for each sub-query from Pinecone."""
    responses = {}
    for cdp, sub_query in sub_queries.items():
        docs = retriever.get_relevant_documents(sub_query)
        response = qa_chain.run(input_documents=docs, question=sub_query) if docs else f"No relevant data for {cdp}."
        responses[cdp] = response
    return responses


def generate_comparison_table(responses, topic):
    """Asks LLM to generate a structured comparison table based on retrieved responses."""
    
    formatted_data = "\n".join([f"- **{cdp}**: {response}" for cdp, response in responses.items()])

    prompt = f"""
    You are an AI assistant creating a structured comparison between different CDPs. 
    Given the following extracted information, create a tabular comparison focused on the topic **{topic}**.

    {formatted_data}

    Provide the response in **Markdown table format** for clarity.
    """

    return llm.predict(prompt)


def process_comparison_query(query: str):
    """Main function to handle comparison-based queries."""
    
    # Step 1: Identify CDPs in query
    mentioned_cdps = extract_cdps(query)
    if not mentioned_cdps:
        return "Comparison requires at least two CDPs. Please specify at least two platforms."

    # Step 2: Dynamically extract topic
    topic = extract_dynamic_topic(query, mentioned_cdps)

    # Step 3: Generate sub-queries
    sub_queries = generate_sub_queries(mentioned_cdps, topic)

    # Step 4: Retrieve responses
    responses = retrieve_information(sub_queries)

    # Step 5: Generate final comparison table
    comparison_result = generate_comparison_table(responses, topic)
    
    return comparison_result


def user_query(query: str) -> str:
    """Determines if the query requires a comparison or a standard response."""
    
    if is_comparison_query(query):
        return process_comparison_query(query)

    # Regular how-to question processing
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant documents found."
    return qa_chain.run(input_documents=docs, question=query)


if __name__ == "__main__":  
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = user_query(query)
        print("\nChatbot: ", response)
