import json
import argparse
import random
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict

# Define category order
CATEGORY_ORDER = [
    "detection",
    "classification",
    "localization",
    "comparison",
    "relationship",
    "diagnosis",
    "characterization",
]


def extract_letter_answer(answer: str) -> str:
    """Extract just the letter answer from various answer formats.

    Args:
        answer: The answer string to extract a letter from

    Returns:
        str: The extracted letter in uppercase, or empty string if no letter found
    """
    if not answer:
        return ""

    # Convert to string and clean
    answer = str(answer).strip()

    # If it's just a single letter A-F, return it
    if len(answer) == 1 and answer.upper() in "ABCDEF":
        return answer.upper()

    # Try to match patterns like "A)", "A.", "A ", etc.
    match = re.match(r"^([A-F])[).\s]", answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try to find any standalone A-F letters preceded by space or start of string
    # and followed by space, period, parenthesis or end of string
    matches = re.findall(r"(?:^|\s)([A-F])(?:[).\s]|$)", answer, re.IGNORECASE)
    if matches:
        return matches[0].upper()

    # Last resort: just find any A-F letter
    letters = re.findall(r"[A-F]", answer, re.IGNORECASE)
    if letters:
        return letters[0].upper()

    # If no letter found, return original (cleaned)
    return answer.strip().upper()


def parse_json_lines(file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse JSON Lines file and extract valid predictions.

    Args:
        file_path: Path to the JSON Lines file to parse

    Returns:
        Tuple containing:
            - str: Model name or file path if model name not found
            - List[Dict[str, Any]]: List of valid prediction entries
    """
    valid_predictions = []
    model_name = None

    # First try to parse as LLaVA format
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("model") == "llava-med-v1.5-mistral-7b":
                model_name = data["model"]
                for result in data.get("results", []):
                    if all(k in result for k in ["case_id", "question_id", "correct_answer"]):
                        # Extract answer with priority: model_answer > validated_answer > raw_output
                        model_answer = (
                            result.get("model_answer")
                            or result.get("validated_answer")
                            or result.get("raw_output", "")
                        )

                        # Add default categories for LLaVA results
                        prediction = {
                            "case_id": result["case_id"],
                            "question_id": result["question_id"],
                            "model_answer": model_answer,
                            "correct_answer": result["correct_answer"],
                            "input": {
                                "question_data": {
                                    "metadata": {
                                        "categories": [
                                            "detection",
                                            "classification",
                                            "localization",
                                            "comparison",
                                            "relationship",
                                            "diagnosis",
                                            "characterization",
                                        ]
                                    }
                                }
                            },
                        }
                        valid_predictions.append(prediction)
                return model_name, valid_predictions
    except (json.JSONDecodeError, KeyError):
        pass

    # If not LLaVA format, process as original format
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("HTTP Request:"):
                continue
            try:
                data = json.loads(line.strip())
                if "model" in data:
                    model_name = data["model"]
                if all(
                    k in data for k in ["model_answer", "correct_answer", "case_id", "question_id"]
                ):
                    valid_predictions.append(data)
            except json.JSONDecodeError:
                continue

    return model_name if model_name else file_path, valid_predictions


def filter_common_questions(
    predictions_list: List[List[Dict[str, Any]]]
) -> List[List[Dict[str, Any]]]:
    """Ensure only questions that exist across all models are evaluated.

    Args:
        predictions_list: List of prediction lists from different models

    Returns:
        List[List[Dict[str, Any]]]: Filtered predictions containing only common questions
    """
    question_sets = [
        set((p["case_id"], p["question_id"]) for p in preds) for preds in predictions_list
    ]
    common_questions = set.intersection(*question_sets)

    return [
        [p for p in preds if (p["case_id"], p["question_id"]) in common_questions]
        for preds in predictions_list
    ]


def calculate_accuracy(
    predictions: List[Dict[str, Any]]
) -> Tuple[float, int, int, Dict[str, Dict[str, float]]]:
    """Compute overall and category-level accuracy.

    Args:
        predictions: List of prediction entries to analyze

    Returns:
        Tuple containing:
            - float: Overall accuracy percentage
            - int: Number of correct predictions
            - int: Total number of predictions
            - Dict[str, Dict[str, float]]: Category-level accuracy statistics
    """
    if not predictions:
        return 0.0, 0, 0, {}

    category_performance = defaultdict(lambda: {"total": 0, "correct": 0})
    correct = 0
    total = 0
    sample_size = min(5, len(predictions))
    sampled_indices = random.sample(range(len(predictions)), sample_size)

    print("\nSample extracted answers:")
    for i in sampled_indices:
        pred = predictions[i]
        model_ans = extract_letter_answer(pred["model_answer"])
        correct_ans = extract_letter_answer(pred["correct_answer"])
        print(f"QID: {pred['question_id']}")
        print(f"  Raw Model Answer: {pred['model_answer']}")
        print(f"  Extracted Model Answer: {model_ans}")
        print(f"  Raw Correct Answer: {pred['correct_answer']}")
        print(f"  Extracted Correct Answer: {correct_ans}")
        print("-" * 80)

    for pred in predictions:
        try:
            model_ans = extract_letter_answer(pred["model_answer"])
            correct_ans = extract_letter_answer(pred["correct_answer"])
            categories = (
                pred.get("input", {})
                .get("question_data", {})
                .get("metadata", {})
                .get("categories", [])
            )

            if model_ans and correct_ans:
                total += 1
                is_correct = model_ans == correct_ans
                if is_correct:
                    correct += 1

                for category in categories:
                    category_performance[category]["total"] += 1
                    if is_correct:
                        category_performance[category]["correct"] += 1

        except KeyError:
            continue

    category_accuracies = {
        category: {
            "accuracy": (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0,
            "total": stats["total"],
            "correct": stats["correct"],
        }
        for category, stats in category_performance.items()
    }

    return (correct / total * 100 if total > 0 else 0.0, correct, total, category_accuracies)


def compare_models(file_paths: List[str]) -> None:
    """Compare accuracy between multiple model prediction files.

    Args:
        file_paths: List of paths to model prediction files to compare
    """
    # Parse all files
    parsed_results = [parse_json_lines(file_path) for file_path in file_paths]
    model_names, predictions_list = zip(*parsed_results)

    # Get initial stats
    print(f"\n📊 **Initial Accuracy**:")
    results = []
    category_results = []

    for preds, name in zip(predictions_list, model_names):
        acc, correct, total, category_acc = calculate_accuracy(preds)
        results.append((acc, correct, total, name))
        category_results.append(category_acc)
        print(f"{name}: Accuracy = {acc:.2f}% ({correct}/{total} correct)")

    # Get common questions across all models
    filtered_predictions = filter_common_questions(predictions_list)
    print(
        f"\nQuestions per model after ensuring common questions: {[len(p) for p in filtered_predictions]}"
    )

    # Compute accuracy on common questions
    print(f"\n📊 **Accuracy on Common Questions**:")
    filtered_results = []
    filtered_category_results = []

    for preds, name in zip(filtered_predictions, model_names):
        acc, correct, total, category_acc = calculate_accuracy(preds)
        filtered_results.append((acc, correct, total, name))
        filtered_category_results.append(category_acc)
        print(f"{name}: Accuracy = {acc:.2f}% ({correct}/{total} correct)")

    # Print category-wise accuracy
    print("\nCategory Performance (Common Questions):")
    for category in CATEGORY_ORDER:
        print(f"\n{category.capitalize()}:")
        for model_name, category_acc in zip(model_names, filtered_category_results):
            stats = category_acc.get(category, {"accuracy": 0, "total": 0, "correct": 0})
            print(f"  {model_name}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")


def main():
    parser = argparse.ArgumentParser(
        description="Compare accuracy across multiple model prediction files"
    )
    parser.add_argument("files", nargs="+", help="Paths to model prediction files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    args = parser.parse_args()
    random.seed(args.seed)

    compare_models(args.files)


if __name__ == "__main__":
    main()
