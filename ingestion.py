"""
CatalogueIQ - Data Ingestion and Chunking

Responsibilities:
1. Load CSV, Markdown and HTML knowledge sources.
2. Convert them into LangChain Documents.
3. Provide two chunking strategies.
4. Preserve metadata required later for citations and retrieval.
"""

import csv
import re
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# -------------------------------------------------------------------
# File loaders
# -------------------------------------------------------------------

def load_products_csv(path: Path | None = None) -> List[Document]:
    """
    Load product records from products.csv.

    Each product row becomes one Document.

    Important product attributes are also stored as metadata
    so they can be used later for filtering.
    """

    path = path or DATA_DIR / "products.csv"

    documents = []

    with open(path, "r", encoding="utf-8", newline="") as file:

        reader = csv.DictReader(file)

        for row in reader:

            product_id = row.get("product_id", "")
            category = row.get("category", "")
            subcategory = row.get("subcategory", "")

            # -------------------------------------------------
            # Convert product row into readable text
            # -------------------------------------------------

            fields = []

            for key, value in row.items():

                if value is not None and value != "":
                    fields.append(
                        f"{key}: {value}"
                    )

            content = "\n".join(fields)

            # -------------------------------------------------
            # Product metadata
            # -------------------------------------------------

            metadata = {
                "source": path.name,
                "document_type": "product",
                "product_id": product_id,
                "category": category,
                "subcategory": subcategory,
            }

            # Numeric metadata
            if row.get("price"):
                metadata["price"] = float(row["price"])

            if row.get("rating"):
                metadata["rating"] = float(row["rating"])

            if row.get("battery_hours"):
                metadata["battery_hours"] = float(
                    row["battery_hours"]
                )

            # Boolean metadata
            if row.get("anc"):
                metadata["anc"] = (
                    row["anc"].strip().lower() == "true"
                )

            if row.get("returnable"):
                metadata["returnable"] = (
                    row["returnable"].strip().lower() == "true"
                )

            if row.get("seller_verified"):
                metadata["seller_verified"] = (
                    row["seller_verified"].strip().lower()
                    == "true"
                )

            # -------------------------------------------------
            # Create document
            # -------------------------------------------------

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

    return documents


def load_reviews_csv(path: Path | None = None) -> List[Document]:
    """
    Load product review summaries.

    Each review summary becomes one Document.
    """

    path = path or DATA_DIR / "reviews.csv"

    documents = []

    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            product_id = row.get("product_id", "")

            content = "\n".join(
                f"{key}: {value}"
                for key, value in row.items()
                if value is not None and value != ""
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": path.name,
                        "document_type": "review",
                        "product_id": product_id,
                    },
                )
            )

    return documents


def load_markdown(path: Path) -> Document:
    """
    Load an entire Markdown file as a single Document.

    This is the raw representation used by the generic chunking
    strategy.
    """

    content = path.read_text(encoding="utf-8")

    return Document(
        page_content=content,
        metadata={
            "source": path.name,
            "document_type": "markdown",
        },
    )


def load_html_faq(path: Path | None = None) -> List[Document]:
    """
    Load the Buyer FAQ.

    Each <section> containing a question and answer becomes one
    Document. This preserves the natural FAQ question-answer unit.
    """

    path = path or DATA_DIR / "buyer_faq.html"

    html_content = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    documents = []

    sections = soup.find_all("section")

    for index, section in enumerate(sections, start=1):
        heading = section.find("h2")
        answer = section.find("p")

        question = heading.get_text(" ", strip=True) if heading else ""
        answer_text = answer.get_text(" ", strip=True) if answer else ""

        content = f"Question: {question}\nAnswer: {answer_text}"

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "document_type": "faq",
                    "faq_id": f"FAQ{index:03d}",
                },
            )
        )

    return documents

# -------------------------------------------------------------------
# Markdown structure extraction
# -------------------------------------------------------------------

def markdown_sections(document: Document) -> List[Document]:
    """
    Split Markdown into heading-based sections.

    Each section retains the heading in its content and stores the
    heading as metadata.
    """

    lines = document.page_content.splitlines()

    sections = []

    current_heading = "Introduction"
    current_lines = []

    def save_section():
        if not current_lines:
            return

        content = "\n".join(current_lines).strip()

        if content:
            sections.append(
                Document(
                    page_content=content,
                    metadata={
                        **document.metadata,
                        "section": current_heading,
                    },
                )
            )

    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())

        if heading_match:
            save_section()

            current_heading = heading_match.group(2).strip()
            current_lines = [line]

        else:
            current_lines.append(line)

    save_section()

    return sections

# -------------------------------------------------------------------
# Strategy 1 - Recursive chunking
# -------------------------------------------------------------------

def recursive_chunking(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    Strategy 1:

    Generic recursive character splitting for unstructured text.

    Structured product, FAQ and review records are preserved as
    natural units because splitting those records would separate
    related attributes.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for document in documents:

        # Preserve structured records.
        if document.metadata.get("document_type") in {
            "product",
            "faq",
            "review",
        }:
            chunks.append(document)
            continue

        split_docs = splitter.split_documents([document])
        chunks.extend(split_docs)

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
    Load every knowledge source used by CatalogueIQ.
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
    strategy_2 = structure_aware_chunking(documents)

    print(f"Strategy 2 chunks: {len(strategy_2)}")

    print("\nExample chunk:")
    print("-" * 40)

    if strategy_2:
        example = strategy_2[0]

        print(example.page_content[:1000])
        print("\nMetadata:")
        print(example.metadata)
