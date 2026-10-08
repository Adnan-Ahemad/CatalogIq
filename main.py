"""
CatalogueIQ - Main Application

Run:

    python main.py

This file orchestrates the complete RAG pipeline.
"""

from rag import (
    get_vector_store,
    answer_query,
    ConversationMemory,
)

# ---------------------------------------------------------
# Test queries
# ---------------------------------------------------------

TEST_QUERIES = [

    # 1. Product factual lookup
    "What is the battery life of the boAt Airdopes 141 earbuds?",

    # 2. Policy and eligibility
    "Can I return a fashion item bought on sale?",

    # 3. Comparative recommendation
    "Which wireless earbuds under 2000 rupees have ANC and at least 20 hours battery life?",

    # 4. Seller policy lookup
    "What are the image requirements for electronics products?",

    # 5. Multi-hop reasoning
    "If I buy a smartwatch from a third-party seller and it stops working after 45 days, what are my options?",
]

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("CATALOGUEIQ")
    print("Product Intelligence Assistant")
    print("=" * 70)

    print("\nInitializing knowledge base...")

    vector_store = get_vector_store()

    memory = ConversationMemory()

    print("\nKnowledge base ready.")
    print("\nRunning RAG pipeline tests...")

    for query in TEST_QUERIES:

        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        answer = answer_query(
            query=query,
            vector_store=vector_store,
            memory=memory,
        )

        print("\nANSWER:")
        print(answer)

# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
