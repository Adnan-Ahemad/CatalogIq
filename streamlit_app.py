"""
CatalogueIQ Streamlit UI

Run:
    streamlit run streamlit_app.py
"""

import streamlit as st

from rag import (
    ConversationMemory,
    answer_query_with_trace,
    create_citation,
    get_vector_store,
)

PERSONAS = {
    "Shopper": (
        "You are assisting a shopper. Prioritize product fit, value, and clear "
        "trade-offs in plain language while staying strictly grounded in cited "
        "context."
    ),
    "Seller": (
        "You are assisting a marketplace seller. Prioritize listing compliance, "
        "policy requirements, and operational clarity while staying strictly "
        "grounded in cited context."
    ),
    "Support Agent": (
        "You are assisting an internal support agent. Prioritize precise policy "
        "interpretation, eligibility checks, and stepwise resolution guidance "
        "while staying strictly grounded in cited context."
    ),
}

def _init_state():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vector_store()

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "persona" not in st.session_state:
        st.session_state.persona = "Shopper"

def _reset_chat():
    st.session_state.memory.clear()
    st.session_state.chat_history = []

def _render_retrieved_context(documents):
    st.markdown("### Retrieved Context")

    if not documents:
        st.info("No context chunks were retrieved.")
        return

    for index, document in enumerate(documents, start=1):
        citation = create_citation(document)
        with st.expander(f"Chunk {index}: {citation}"):
            st.markdown("**Metadata**")
            st.json(document.metadata)
            st.markdown("**Content**")
            st.write(document.page_content)

def main():
    st.set_page_config(
        page_title="CatalogueIQ",
        page_icon="🛍️",
        layout="wide",
    )

    _init_state()

    st.title("CatalogueIQ")
    st.caption("RAG-powered Product Intelligence Assistant for ShopSmart India")

    with st.sidebar:
        st.subheader("Session Controls")

        selected_persona = st.selectbox(
            "Choose Persona",
            list(PERSONAS.keys()),
            index=list(PERSONAS.keys()).index(st.session_state.persona),
        )

        if selected_persona != st.session_state.persona:
            st.session_state.persona = selected_persona

        st.markdown("**Active Persona Prompt**")
        st.info(PERSONAS[st.session_state.persona])

        if st.button("Clear Conversation"):
            _reset_chat()
            st.rerun()

        st.markdown("---")
        st.write("Memory turns retained:", len(st.session_state.memory.get_history()))

    left_col, right_col = st.columns([1.2, 1.0])

    with left_col:
        st.subheader("Chat")

        for item in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(item["question"])

            with st.chat_message("assistant"):
                st.write(item["answer"])

        user_query = st.chat_input("Ask a product, policy, seller, or support question")

        if user_query:
            with st.chat_message("user"):
                st.write(user_query)

            with st.spinner("Retrieving and generating answer..."):
                result = answer_query_with_trace(
                    query=user_query,
                    vector_store=st.session_state.vector_store,
                    memory=st.session_state.memory,
                    persona_instruction=PERSONAS[st.session_state.persona],
                )

            answer = result["answer"]

            st.session_state.chat_history.append(
                {
                    "question": user_query,
                    "answer": answer,
                    "retrieval_query": result["retrieval_query"],
                    "documents": result["documents"],
                }
            )

            with st.chat_message("assistant"):
                st.write(answer)

            st.rerun()

    with right_col:
        st.subheader("Evidence Panel")

        if not st.session_state.chat_history:
            st.info("Ask a question to view rewritten query and retrieved chunks.")
        else:
            latest = st.session_state.chat_history[-1]
            st.markdown("### Retrieval Query")
            st.write(latest["retrieval_query"])
            _render_retrieved_context(latest["documents"])

if __name__ == "__main__":
    main()
