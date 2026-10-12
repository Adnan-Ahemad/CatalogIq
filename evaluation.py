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
    Run the ProductLens RAG pipeline for every evaluation
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
    print("PRODUCTLENS RAGAS EVALUATION")
    print("=" * 70)

    dataset = build_evaluation_dataset(
        vector_store
    )
