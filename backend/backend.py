import sys
from milvus_utils import connect_milvus, search_milvus
from setupLLM import setup_llm

if __name__ == "__main__":
    # Connect to Milvus
    connect_milvus()

    # Set up LLM
    llm = setup_llm()

    # Example: A query from user
    query = "Hva er NEET?"

    # Assume we have some vectorized version of the query (example vector)
    vector = [0.1, 0.2, 0.3]  # Just a placeholder vector for this example

    # Search Milvus for similar vectors
    results = search_milvus(vector)

    # Generate response using LLM (Here, we're assuming results contain some context)
    context = results[0].entity.get('context')
    response = llm(f"{context} Spørsmål: {query}")

    # Output the generated response
    print(f"Generated Response: {response}")

    sys.exit()
