from pathlib import Path
import json
import re

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

from config import (
    EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    LLM_MODEL,
    TEMPERATURE,
    MAX_CONTEXT_CHUNKS,
    MAX_MEMORY_TURNS,
)

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "vector_store"
COLLECTION_NAME = "catalogueiq"

# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings():
    """Create the embedding model."""

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

# ============================================================
# VECTOR STORE
# ============================================================

def get_vector_store():
    """
    Load the existing ChromaDB vector store.

    If it does not exist, build it from the knowledge base.
    """

    embeddings = create_embeddings()

    if CHROMA_DIR.exists():
        print("\nExisting ChromaDB index found.")
        print("Loading existing vector store...")

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        try:
            count = vector_store._collection.count()
            print(f"      Chunks available: {count}")
        except Exception:
            pass

        return vector_store

    print("\nNo existing ChromaDB index found.")
    print("Building index for the first time...")

    from ingestion import (
        load_all_documents,
        structure_aware_chunking,
    )

    print("\n[1/4] Loading knowledge base...")

    documents = load_all_documents()

    print(f"      Documents loaded: {len(documents)}")

    print("\n[2/4] Applying structure-aware chunking...")

    chunks = structure_aware_chunking(documents)

    print(f"      Chunks created: {len(chunks)}")

    print("\n[3/4] Creating embeddings...")
    print(f"      Model: {EMBEDDING_MODEL}")

    print("\n[4/4] Building ChromaDB index...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print("      ChromaDB index created successfully.")

    return vector_store

# ============================================================
# LLM
# ============================================================

def create_llm():
    """Create the GPT-4o-mini generation model."""

    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
    )

# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """
    Stores the most recent conversation turns.

    One turn consists of:
        user query
        assistant answer

    The memory is limited to MAX_MEMORY_TURNS.
    """

    def __init__(self, max_turns=MAX_MEMORY_TURNS):
        self.max_turns = max_turns
        self.history = []

    def add_turn(self, query, answer):
        """Add a completed conversation turn."""

        self.history.append(
            {
                "user": query,
                "assistant": answer,
            }
        )

        # Keep only the most recent N turns.
        self.history = self.history[
            -self.max_turns:
        ]

    def get_history(self):
        """Return the current conversation history."""

        return self.history

    def clear(self):
        """Clear the conversation."""

        self.history = []

    def is_empty(self):
        """Check whether memory contains any turns."""

        return len(self.history) == 0

    def format_history(self):
        """Format memory for use in an LLM prompt."""

        if not self.history:
            return "No previous conversation."

        parts = []

        for index, turn in enumerate(
            self.history,
            start=1,
        ):
            parts.append(
                f"Turn {index}\n"
                f"User: {turn['user']}\n"
                f"Assistant: {turn['assistant']}"
            )

        return "\n\n".join(parts)
# ============================================================
# QUERY UNDERSTANDING
# ============================================================

def analyze_query(query):
    """
    Use the LLM to understand the user's query.

    The LLM determines:
    - user intent
    - whether structured product filtering is needed
    - relevant product constraints
    - search queries for retrieval

    Python does not contain product/category-specific rules.
    """

    llm = create_llm()

    prompt = f"""
You are the query-understanding component of CatalogueIQ,
a RAG-powered product intelligence assistant for ShopSmart India.

Analyze the user's query and return ONLY valid JSON.

Your job is to understand the user's intent and identify
structured constraints that can be used for metadata filtering.

Possible intents include:
- product_lookup
- product_comparison
- product_recommendation
- policy_lookup
- seller_policy_lookup
- troubleshooting
- multi_hop
- general

Determine whether metadata filtering is useful.

For product-related queries, identify any constraints explicitly
stated or clearly implied by the user, such as:
- category
- subcategory
- price
- rating
- battery_hours
- anc
- returnable
- seller_verified
- warranty_months
- any other metadata field that exists in the product catalogue

Do NOT invent constraints.

For numeric constraints, represent them using:
- operator: "eq"
- operator: "lte"
- operator: "lt"
- operator: "gte"
- operator: "gt"

For example:

"under 2000 rupees"

becomes:

{{
    "field": "price",
    "operator": "lte",
    "value": 2000
}}

"at least 20 hours battery"

becomes:

{{
    "field": "battery_hours",
    "operator": "gte",
    "value": 20
}}

"with ANC"

becomes:

{{
    "field": "anc",
    "operator": "eq",
    "value": true
}}

Important:
- Do not add a constraint merely because a product category
  commonly has that attribute.
- Only extract constraints relevant to this query.
- Do not answer the user's question.
- Do not invent product names.
- Do not invent values.
- Do not add filters to policy/troubleshooting queries unless
  the user explicitly asks for product filtering.

Also generate exactly TWO alternative search queries that
preserve the user's meaning and constraints.

Return exactly this JSON structure:

{{
    "intent": "string",
    "needs_product_filter": true,
    "filters": [
        {{
            "field": "string",
            "operator": "eq|lte|lt|gte|gt",
            "value": "string or number or boolean"
        }}
    ],
    "search_queries": [
        "query 1",
        "query 2"
    ]
}}

USER QUERY:
{query}
"""

    response = llm.invoke(prompt)

    content = response.content.strip()

    # Handle accidental markdown code fences.
    content = re.sub(
        r"^```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )

    content = re.sub(
        r"\s*```$",
        "",
        content,
        flags=re.IGNORECASE,
    )

    try:
        analysis = json.loads(content)
    except json.JSONDecodeError:

        print("\nWarning: Query analysis returned invalid JSON.")
        print("Falling back to semantic retrieval.")

        return {
            "intent": "general",
            "needs_product_filter": False,
            "filters": [],
            "search_queries": [
                query,
                query,
                query,
            ],
        }

    # Validate expected fields.
    if not isinstance(analysis, dict):
        analysis = {}

    intent = analysis.get("intent", "general")

    needs_product_filter = analysis.get(
        "needs_product_filter",
        False,
    )

    filters = analysis.get("filters", [])

    search_queries = analysis.get(
        "search_queries",
        [],
    )

    if not isinstance(filters, list):
        filters = []

    if not isinstance(search_queries, list):
        search_queries = []

    # Keep exactly two alternative queries.
    search_queries = [
        str(item).strip()
        for item in search_queries
        if str(item).strip()
    ][:2]

    while len(search_queries) < 2:
        search_queries.append(query)

    return {
        "intent": intent,
        "needs_product_filter": bool(needs_product_filter),
        "filters": filters,
        "search_queries": search_queries,
    }


def rewrite_query_with_memory(
    query,
    memory,
):
    """
    Rewrite a follow-up question into a standalone query
    using conversation history.

    If the current query is already self-contained,
    it should remain essentially unchanged.
    """

    if memory is None or memory.is_empty():
        return query

    llm = create_llm()

    history = memory.format_history()

    prompt = f"""
You are the query-rewriting component of CatalogueIQ.

Rewrite the user's latest question into a standalone search
query using the previous conversation when necessary.

The rewritten query will be used for knowledge-base retrieval.

Rules:

1. Resolve references such as:
   - it
   - this product
   - that one
   - the second option
   - its price
   - its battery
   - that seller
   - the product mentioned earlier

2. Use previous conversation only to resolve references
   and missing context.

3. Do not answer the question.

4. Do not add requirements that the user did not ask for.

5. If the latest question is already self-contained,
   keep its meaning unchanged.

6. Return ONLY the rewritten standalone query.

PREVIOUS CONVERSATION:
{history}

LATEST USER QUESTION:
{query}
"""

    response = llm.invoke(prompt)

    rewritten_query = response.content.strip()

    return rewritten_query

# ============================================================
# CHROMA FILTER BUILDER
# ============================================================

def build_chroma_filter(filters):
    """
    Convert the LLM-generated generic filter representation
    into a ChromaDB filter.

    This function contains no product/category knowledge.
    """

    operator_map = {
        "eq": "$eq",
        "lte": "$lte",
        "lt": "$lt",
        "gte": "$gte",
        "gt": "$gt",
    }

    conditions = []

    invalid_values = {
        "",
        "unknown",
        "n/a",
        "na",
        "none",
        "null",
    }

    for item in filters:

        if not isinstance(item, dict):
            continue

        field = item.get("field")
        operator = item.get("operator")
        value = item.get("value")

        if not field or operator not in operator_map:
            continue

        if value is None:
            continue

        if isinstance(value, str):
            normalized_value = value.strip().lower()

            if normalized_value in invalid_values:
                continue

        chroma_operator = operator_map[operator]

        conditions.append(
            {
                field: {
                    chroma_operator: value
                }
            }
        )

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {
        "$and": conditions
    }

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query,
    vector_store,
    top_k=DEFAULT_TOP_K,
    filter=None,
):
    """
    Retrieve relevant chunks from ChromaDB.
    """

    if filter:
        return vector_store.similarity_search(
            query,
            k=top_k,
            filter=filter,
        )

    return vector_store.similarity_search(
        query,
        k=top_k,
    )

def deduplicate_documents(documents):
    """
    Remove duplicate chunks while preserving order.
    """

    unique_documents = []
    seen = set()

    for document in documents:

        metadata = document.metadata

        key = (
            document.page_content,
            metadata.get("source"),
            metadata.get("product_id"),
            metadata.get("section"),
            metadata.get("faq_id"),
        )

        if key not in seen:
            seen.add(key)
            unique_documents.append(document)

    return unique_documents

def retrieve_with_expansion(
    query,
    vector_store,
    top_k=DEFAULT_TOP_K,
):
    """
    Analyze the query using the LLM, then perform retrieval.

    Product comparison/recommendation queries can use
    LLM-generated metadata filters.

    Other queries use normal semantic retrieval.
    """

    print("\nAnalyzing query...")

    analysis = analyze_query(query)

    print(f"Intent: {analysis['intent']}")

    search_queries = [
        query,
        *analysis["search_queries"],
    ]

    print("\nSearch queries:")

    for index, search_query in enumerate(
        search_queries,
        start=1,
    ):
        print(f"  {index}. {search_query}")

    documents = []

    # --------------------------------------------------------
    # Metadata-filtered retrieval
    # --------------------------------------------------------

    if (
        analysis["needs_product_filter"]
        and analysis["filters"]
    ):

        chroma_filter = build_chroma_filter(
            analysis["filters"]
        )

        if chroma_filter:

            print("\nLLM-extracted product constraints:")

            for item in analysis["filters"]:
                print(
                    f"  {item.get('field')} "
                    f"{item.get('operator')} "
                    f"{item.get('value')}"
                )

            print("\nApplying metadata filtering...")

            for search_query in search_queries:

                results = retrieve(
                    search_query,
                    vector_store,
                    top_k=top_k,
                    filter=chroma_filter,
                )

                documents.extend(results)

            documents = deduplicate_documents(
                documents
            )

            documents = documents[:MAX_CONTEXT_CHUNKS]

            print(
                f"\nRetrieved {len(documents)} "
                "unique constraint-matching chunks."
            )

            if documents:
                return documents

            print(
                "No results after metadata filtering. "
                "Falling back to normal semantic retrieval..."
            )

    # --------------------------------------------------------
    # Normal semantic retrieval
    # --------------------------------------------------------

    for search_query in search_queries:

        results = retrieve(
            search_query,
            vector_store,
            top_k=top_k,
        )

        documents.extend(results)

    documents = deduplicate_documents(
        documents
    )

    documents = documents[:MAX_CONTEXT_CHUNKS]

    print(
        f"\nRetrieved {len(documents)} unique chunks."
    )

    return documents

# ============================================================
# CITATIONS
# ============================================================

def create_citation(document):
    """
    Create a deterministic citation from document metadata.
    """

    metadata = document.metadata

    source = metadata.get(
        "source",
        "unknown",
    )

    product_id = metadata.get(
        "product_id"
    )

    section = metadata.get(
        "section"
    )

    faq_id = metadata.get(
        "faq_id"
    )

    identifier = None

    if product_id:
        identifier = product_id

    elif faq_id:
        identifier = faq_id

    elif section:
        identifier = section

    if identifier:
        return f"{source} | {identifier}"

    return source

def format_context(documents):
    """
    Format retrieved documents using deterministic source
    numbers.
    """

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        citation = create_citation(
            document
        )

        context_parts.append(
            f"[SOURCE {index}]\n"
            f"Citation: {citation}\n"
            f"Content:\n{document.page_content}"
        )

    return "\n\n".join(context_parts)

# ============================================================
# CITATION POST-PROCESSING
# ============================================================

def replace_source_markers(
    answer,
    documents,
):
    """
    Replace [SOURCE n] with deterministic citations.
    """

    def replace_match(match):

        source_number = int(
            match.group(1)
        )

        if 1 <= source_number <= len(
            documents
        ):

            citation = create_citation(
                documents[
                    source_number - 1
                ]
            )

            return f"[{citation}]"

        return ""

    answer = re.sub(
        r"\[SOURCE\s+(\d+)\]",
        replace_match,
        answer,
        flags=re.IGNORECASE,
    )

    # Remove malformed citations if the
    # model still produces them.
    answer = re.sub(
        r"\[source\s*\|\s*identifier\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    return answer.strip()

# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    query,
    documents,
    persona_instruction="",
):
    """
    Generate a grounded answer using only retrieved context.
    """

    if not documents:
        return (
            "I could not find enough information in the "
            "knowledge base to answer this question."
        )

    llm = create_llm()

    context = format_context(
        documents
    )

    persona_block = ""

    if persona_instruction:
        persona_block = (
            "\nPERSONA INSTRUCTION:\n"
            f"{persona_instruction}\n"
        )

    prompt = f"""
You are CatalogueIQ, a product intelligence assistant
for ShopSmart India.

{persona_block}

Answer the user's question using ONLY the retrieved context.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not invent product specifications, prices, policies,
   warranties, or recommendations.
3. If the context does not contain enough information,
   clearly state that.
4. For recommendations, include only products supported
   by the retrieved context and satisfying the user's
   stated requirements.
5. Every factual claim must be supported by retrieved context.
6. Cite supporting context using ONLY [SOURCE n].
7. Do not invent citation numbers.
8. Do not write [source | identifier].
9. Keep the answer concise and directly relevant.
10. For multi-hop questions, combine information from
    multiple retrieved sources when necessary.

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}
"""

    response = llm.invoke(
        prompt
    )

    answer = response.content.strip()

    answer = replace_source_markers(
        answer,
        documents,
    )

    return answer

# ============================================================
# END-TO-END RAG PIPELINE
# ============================================================

def answer_query(
    query,
    vector_store=None,
):
    """
    Complete CatalogueIQ RAG pipeline.
    """

    if vector_store is None:
        vector_store = get_vector_store()

    documents = retrieve_with_expansion(
        query=query,
        vector_store=vector_store,
    )

    answer = generate_answer(
        query=query,
        documents=documents,
    )

    return answer


def answer_query(
    query,
    vector_store=None,
    memory=None,
    persona_instruction="",
):
    """
    Complete CatalogueIQ RAG pipeline with optional
    conversation memory.
    """

    if vector_store is None:
        vector_store = get_vector_store()

    # --------------------------------------------------------
    # Rewrite follow-up query using conversation history
    # --------------------------------------------------------

    retrieval_query = rewrite_query_with_memory(
        query=query,
        memory=memory,
    )

    if retrieval_query != query:
        print(
            f"\nRewritten query: {retrieval_query}"
        )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    documents = retrieve_with_expansion(
        query=retrieval_query,
        vector_store=vector_store,
    )

    # --------------------------------------------------------
    # Answer generation
    # --------------------------------------------------------

    answer = generate_answer(
        query=retrieval_query,
        documents=documents,
        persona_instruction=persona_instruction,
    )

    # --------------------------------------------------------
    # Store the ORIGINAL user question and answer.
    # --------------------------------------------------------

    if memory is not None:
        memory.add_turn(
            query=query,
            answer=answer,
        )

    return answer


def answer_query_with_trace(
    query,
    vector_store=None,
    memory=None,
    persona_instruction="",
):
    """
    Run the full pipeline and return both the answer and
    retrieved evidence for UI/debugging use cases.
    """

    if vector_store is None:
        vector_store = get_vector_store()

    retrieval_query = rewrite_query_with_memory(
        query=query,
        memory=memory,
    )

    documents = retrieve_with_expansion(
        query=retrieval_query,
        vector_store=vector_store,
    )

    answer = generate_answer(
        query=retrieval_query,
        documents=documents,
        persona_instruction=persona_instruction,
    )

    if memory is not None:
        memory.add_turn(
            query=query,
            answer=answer,
        )

    return {
        "answer": answer,
        "retrieval_query": retrieval_query,
        "documents": documents,
    }

# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    vector_store = get_vector_store()

    while True:

        query = input(
            "\nAsk CatalogueIQ a question "
            "(or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(
            answer_query(
                query=query,
                vector_store=vector_store,
            )
        )
