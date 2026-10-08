"""
CatalogueIQ - RAGAS Evaluation

Evaluates the RAG pipeline using:
    1. Faithfulness
    2. Answer Relevancy
    3. Context Precision
    4. Context Recall

Run:

    python evaluation.py
"""

from datasets import Dataset

from rag import (
    get_vector_store,
    answer_query,
    retrieve_with_expansion,
    ConversationMemory,
    create_llm,
)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ============================================================
# EVALUATION DATASET
# ============================================================

EVALUATION_DATA = [

    {
        "question": "What is the battery life of the boAt Airdopes 141 earbuds?",
        "ground_truth": "The boAt Airdopes 141 earbuds have a battery life of 42 hours.",
    },

    {
        "question": "What is the price of the boAt Airdopes 141 earbuds?",
        "ground_truth": "The boAt Airdopes 141 earbuds cost 1499 rupees.",
    },

    {
        "question": "What is the rating of the boAt Airdopes 141 earbuds?",
        "ground_truth": "The boAt Airdopes 141 earbuds have a rating of 4.3.",
    },

    {
        "question": "How many reviews does the boAt Airdopes 141 have?",
        "ground_truth": "The boAt Airdopes 141 has 8074 reviews.",
    },

    {
        "question": "Does the boAt Airdopes 141 have ANC?",
        "ground_truth": "Yes, the boAt Airdopes 141 has ANC.",
    },

    {
        "question": "Can I return a fashion item bought on sale?",
        "ground_truth": "Yes, eligible fashion products bought on sale can generally be returned within 30 days if they satisfy the return conditions.",
    },

    {
        "question": "How long can eligible fashion products be returned?",
        "ground_truth": "Eligible fashion products can generally be returned within 30 days of delivery.",
    },

    {
        "question": "What condition must fashion items satisfy for return?",
        "ground_truth": "Fashion items should be unused, unwashed, and have their original tags and packaging intact.",
    },

    {
        "question": "What are the image requirements for electronics products?",
        "ground_truth": "Electronics products require clear high-resolution images, a complete primary product image on a clean background, additional detail images, and no misleading specifications or unrelated products.",
    },

    {
        "question": "What should the primary image of an electronics product show?",
        "ground_truth": "The primary image should show the complete product on a clean background.",
    },

    {
        "question": "What additional images should electronics sellers provide?",
        "ground_truth": "Additional images should highlight important ports, accessories, labels, or relevant product details.",
    },

    {
        "question": "Which wireless earbuds under 2000 rupees have ANC and at least 20 hours battery life?",
        "ground_truth": "Matching products include boAt Airdopes 141, Noise Buds Pro, and OnePlus Wireless Earbuds 967.",
    },

    {
        "question": "What is the battery life of the Noise Buds Pro?",
        "ground_truth": "The Noise Buds Pro has a battery life of 40 hours.",
    },

    {
        "question": "What is the price of the Noise Buds Pro?",
        "ground_truth": "The Noise Buds Pro costs 1899 rupees.",
    },

    {
        "question": "What is the rating of the Noise Buds Pro?",
        "ground_truth": "The Noise Buds Pro has a rating of 4.4.",
    },

    {
        "question": "What should I do if a smartwatch from a third-party seller stops working after 45 days?",
        "ground_truth": "After the normal return window, the buyer should check the product warranty and pursue an eligible warranty claim if applicable.",
    },

    {
        "question": "Does buying from a third-party seller change the warranty period?",
        "ground_truth": "No. Third-party seller status does not alter the applicable warranty period.",
    },

    {
        "question": "What should I check after the return window has expired?",
        "ground_truth": "The buyer should check the product's stated warranty and determine whether a warranty claim is available.",
    },

    {
        "question": "What is the warranty period of the Sony Watch 2?",
        "ground_truth": "The Sony Watch 2 has a warranty of 6 months.",
    },

    {
        "question": "What is the warranty period of the Sony Watch 6?",
        "ground_truth": "The Sony Watch 6 has a warranty of 12 months.",
    },
]

# ============================================================
# HELPERS
# ============================================================

def extract_scalar(value):
    """
    Convert a RAGAS metric result into a single float.

    Handles:
        float
        int
        numpy scalar
        lists
        tuples
        dictionaries
        pandas-like series
        None
    """

    if value is None:
        return 0.0

    # Normal scalar
    if isinstance(value, (int, float)):
        return float(value)

    # List / tuple
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0

        values = [
            extract_scalar(item)
            for item in value
        ]

        return sum(values) / len(values)

    # Dictionary
    if isinstance(value, dict):

        # Prefer common aggregate fields.
        for key in [
            "mean",
            "average",
            "score",
            "value",
        ]:
            if key in value:
                return extract_scalar(
                    value[key]
                )

        numeric_values = []

        for item in value.values():

            if isinstance(
                item,
                (int, float),
            ):
                numeric_values.append(
                    float(item)
                )

        if numeric_values:
            return (
                sum(numeric_values)
                / len(numeric_values)
            )

        return 0.0

    # Objects that expose a mean() method.
    if hasattr(value, "mean"):

        try:
            return float(value.mean())
        except Exception:
            pass

    # Objects with tolist()
    if hasattr(value, "tolist"):

        try:
            return extract_scalar(
                value.tolist()
            )
        except Exception:
            pass

    # Final fallback
    try:
        return float(value)
    except Exception:
        return 0.0


def clamp_score(value):
    """
    Guarantee a valid RAGAS score between 0 and 1.
    """

    score = extract_scalar(value)

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )

# ============================================================
# BUILD EVALUATION DATASET
# ============================================================

def build_evaluation_dataset(
    vector_store,
):
    """
    Run the CatalogueIQ RAG pipeline for every evaluation
    question and construct the dataset expected by RAGAS.

    Required columns:

        question
        answer
        contexts
        ground_truth
    """

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # Evaluation should use independent questions,
    # not conversation memory.
    memory = None

    total = len(
        EVALUATION_DATA
    )

    print("\nGenerating evaluation dataset...")
    print(
        f"Total evaluation questions: {total}"
    )

    for index, item in enumerate(
        EVALUATION_DATA,
        start=1,
    ):

        question = item["question"]

        print(
            f"\n[{index}/{total}] {question}"
        )

        # Retrieve independently so that RAGAS
        # receives the exact contexts used by the answer.
        retrieved_documents = retrieve_with_expansion(
            query=question,
            vector_store=vector_store,
        )

        answer = answer_query(
            query=question,
            vector_store=vector_store,
            memory=memory,
        )

        question_contexts = [
            document.page_content
            for document in retrieved_documents
        ]

        questions.append(
            question
        )

        answers.append(
            answer
        )

        contexts.append(
            question_contexts
        )

        ground_truths.append(
            item["ground_truth"]
        )

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    return dataset

# ============================================================
# RAGAS EVALUATION
# ============================================================

def run_ragas_evaluation(
    vector_store=None,
):
    """
    Run RAGAS evaluation.

    Returns exactly four scalar metrics:

        {
            "faithfulness": float,
            "answer_relevancy": float,
            "context_precision": float,
            "context_recall": float
        }
    """

    if vector_store is None:
        vector_store = get_vector_store()

    print("\n" + "=" * 70)
    print("CATALOGUEIQ RAGAS EVALUATION")
    print("=" * 70)

    dataset = build_evaluation_dataset(
        vector_store
    )

    print("\nEvaluation dataset created.")

    print(
        f"Questions: {len(dataset)}"
    )

    print(
        f"Columns: {dataset.column_names}"
    )

    print("\nRunning RAGAS metrics...")

    # GPT-4o-mini is used as the evaluation LLM.
    evaluator_llm = create_llm()

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
    )

    # --------------------------------------------------------
    # Normalize every metric to a scalar float.
    # --------------------------------------------------------

    scores = {}

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    for metric_name in metric_names:

        try:

            raw_score = result[
                metric_name
            ]

        except Exception:

            raw_score = 0.0

        scores[
            metric_name
        ] = clamp_score(
            raw_score
        )

    print("\n" + "=" * 70)
    print("RAGAS RESULTS")
    print("=" * 70)

    for metric_name, score in scores.items():

        print(
            f"{metric_name:20s}: "
            f"{score:.4f}"
        )

    return scores

# ============================================================
# MAIN
# ============================================================

def main():

    vector_store = get_vector_store()

    scores = run_ragas_evaluation(
        vector_store=vector_store
    )

    print("\nFinal metric dictionary:")
    print(scores)

if __name__ == "__main__":
    main()
