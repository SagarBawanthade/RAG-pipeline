
import json
from pathlib import Path

from rag import get_collection, retrieve, TOP_K

DATASET_PATH = Path("evaluation/questions.json")
THRESHOLDS = [0.55, 0.58, 0.60, 0.62, 0.65]


def load_questions():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate():
    questions = load_questions()
    collection = get_collection()

    # Retrieve results only once per question.
    # Reuse the same scores for every threshold.
    results = []

    print(f"Loading retrieval results for {len(questions)} questions...")

    for item in questions:
        hits = retrieve(item["question"], collection, k=TOP_K)
        results.append((item, hits))

    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Answerable':<16}"
        f"{'Source found':<16}"
        f"{'Rejected correctly':<20}"
    )
    print("-" * 70)

    answerable_questions = [
        (item, hits)
        for item, hits in results
        if item.get("expected_source") is not None
    ]

    unanswerable_questions = [
        (item, hits)
        for item, hits in results
        if item.get("expected_source") is None
    ]

    for threshold in THRESHOLDS:
        answerable_with_hits = 0
        expected_source_found = 0
        correctly_rejected = 0

        for item, hits in answerable_questions:
            accepted = [
                (score, chunk)
                for score, chunk in hits
                if score >= threshold
            ]

            if accepted:
                answerable_with_hits += 1

            if any(
                chunk["source"] == item["expected_source"]
                for score, chunk in accepted
            ):
                expected_source_found += 1

        for item, hits in unanswerable_questions:
            accepted = [
                (score, chunk)
                for score, chunk in hits
                if score >= threshold
            ]

            if not accepted:
                correctly_rejected += 1

        answerable_total = len(answerable_questions)
        unanswerable_total = len(unanswerable_questions)

        answerable_pct = (
            answerable_with_hits / answerable_total * 100
            if answerable_total else 0
        )
        source_pct = (
            expected_source_found / answerable_total * 100
            if answerable_total else 0
        )
        rejection_pct = (
            correctly_rejected / unanswerable_total * 100
            if unanswerable_total else 0
        )

        print(
            f"{threshold:<12.2f}"
            f"{answerable_with_hits}/{answerable_total} "
            f"({answerable_pct:.0f}%)     "
            f"{expected_source_found}/{answerable_total} "
            f"({source_pct:.0f}%)     "
            f"{correctly_rejected}/{unanswerable_total} "
            f"({rejection_pct:.0f}%)"
        )

    print("\nHigher thresholds reject more weak matches, but may")
    print("also reject useful chunks. Compare all three metrics.")


if __name__ == "__main__":
    evaluate()
