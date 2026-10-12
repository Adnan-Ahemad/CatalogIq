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
You are ProductLens, a product intelligence assistant
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
    Complete ProductLens RAG pipeline.
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
    Complete ProductLens RAG pipeline with optional
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
