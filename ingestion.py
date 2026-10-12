    return chunks

# -------------------------------------------------------------------
# Strategy 2 - Structure-aware chunking
# -------------------------------------------------------------------

def structure_aware_chunking(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Strategy 2:

    Use the natural structure of each source:

    Product → one product
    Review  → one review summary
    FAQ     → question + answer
    Markdown → heading/section

    Long Markdown sections are recursively split while retaining
    section metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for document in documents:

        document_type = document.metadata.get("document_type")

        # Product record → natural chunk.
        if document_type == "product":
            chunks.append(document)

        # Review summary → natural chunk.
        elif document_type == "review":
            chunks.append(document)

        # FAQ question + answer → natural chunk.
        elif document_type == "faq":
            chunks.append(document)

        # Markdown → heading-aware chunks.
        elif document_type == "markdown":
            sections = markdown_sections(document)

            for section in sections:

                # Keep small sections intact.
                if len(section.page_content) <= chunk_size:
                    chunks.append(section)

                else:
                    # Split large sections while retaining metadata.
                    split_sections = splitter.split_documents([section])
                    chunks.extend(split_sections)

    return chunks

# -------------------------------------------------------------------
# Load complete knowledge base
# -------------------------------------------------------------------

def load_all_documents() -> List[Document]:
    """
    Load every knowledge source used by ProductLens.
    """

    documents = []

    # Structured data
    documents.extend(load_products_csv())
    documents.extend(load_reviews_csv())

    # Markdown documents
    markdown_files = [
        DATA_DIR / "returns_policy.md",
        DATA_DIR / "seller_guide.md",
        DATA_DIR / "category_guides.md",
    ]

    for path in markdown_files:
        if path.exists():
            documents.append(load_markdown(path))

    # HTML
    faq_path = DATA_DIR / "buyer_faq.html"

    if faq_path.exists():
        documents.extend(load_html_faq(faq_path))

    return documents

# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------

def print_document_summary(documents: List[Document]) -> None:
    """
    Print a simple summary of loaded documents.
    """

    counts = {}

    for document in documents:
        doc_type = document.metadata.get("document_type", "unknown")
        counts[doc_type] = counts.get(doc_type, 0) + 1

    print("\nDocument Summary")
    print("-" * 40)

    for doc_type, count in sorted(counts.items()):
        print(f"{doc_type:15} : {count}")

    print(f"{'TOTAL':15} : {len(documents)}")

# -------------------------------------------------------------------
# Local test
# -------------------------------------------------------------------

if __name__ == "__main__":

    print("Loading ShopSmart India knowledge base...")

    documents = load_all_documents()

    print_document_summary(documents)

    print("\nApplying Strategy 1...")
    strategy_1 = recursive_chunking(documents)

    print(f"Strategy 1 chunks: {len(strategy_1)}")

    print("\nApplying Strategy 2...")
