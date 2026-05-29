import json
import sys
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.graph import graph  # noqa: E402

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


def run():
    with open(DATASET_PATH) as f:
        eval_data = json.load(f)

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"Running graph on {len(eval_data)} questions...\n")
    for i, entry in enumerate(eval_data, 1):
        question = entry["question"]
        print(f"[{i}/{len(eval_data)}] {question[:70]}...")
        result = graph.invoke({"query": question, "retrieval_count": 0})
        questions.append(question)
        answers.append(result["generation"])
        ground_truths.append(entry["ground_truth"])
        docs = result.get("documents") or []
        contexts.append([doc.page_content for doc in docs])

    print("\nBuilding RAGAs dataset...")
    ragas_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    print("Running RAGAs evaluation (this calls OpenAI — takes ~2 min)...\n")
    scores = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n=== RAGAs Scores ===")
    print(scores)

    df = scores.to_pandas()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"results_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPer-question results saved to {out_path}")

    print("\n=== Summary ===")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric in df.columns:
            mean = df[metric].mean()
            status = "PASS" if mean >= 0.70 else "BELOW TARGET"
            print(f"  {metric:<25} {mean:.4f}  ({status})")


if __name__ == "__main__":
    run()
