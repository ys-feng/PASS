from typing import Dict, List, Optional, Tuple, Union, Any
import json
import os
import sys
import argparse
from collections import defaultdict
from tqdm import tqdm

QUESTION_TYPES = {
    "Detailed Finding Analysis": ["detection", "localization", "characterization"],
    "Pattern Recognition & Relations": ["detection", "classification", "relationship"],
    "Spatial Understanding": ["localization", "comparison", "relationship"],
    "Clinical Decision Making": ["classification", "comparison", "diagnosis"],
    "Diagnostic Classification": ["classification", "characterization", "diagnosis"],
}


def extract_answer_letter(answer: Optional[Union[str, Any]]) -> Optional[str]:
    """
    Extract just the letter from various answer formats.

    Args:
        answer: The answer text to extract letter from

    Returns:
        Optional[str]: The extracted letter in uppercase, or None if no letter found
    """
    if not answer:
        return None

    # Convert to string and clean
    answer = str(answer).strip()

    # If it's just a single letter, return it
    if len(answer) == 1 and answer.isalpha():
        return answer.upper()

    # Try to extract letter from format like "A)" or "A."
    if len(answer) >= 2 and answer[0].isalpha() and answer[1] in ").:- ":
        return answer[0].upper()

    # Try to extract letter from format like "A) Some text"
    if answer.startswith(("A)", "B)", "C)", "D)", "E)", "F)")):
        return answer[0].upper()

    return None


def analyze_gpt4_results(
    results_file: str, max_questions: Optional[int] = None
) -> Tuple[float, Dict, Dict, List[str], List[str]]:
    """
    Analyze results in GPT-4 format.

    Args:
        results_file: Path to results file
        max_questions: Maximum number of questions to analyze

    Returns:
        Tuple containing:
            - overall_accuracy (float)
            - category_accuracies (Dict)
            - question_type_stats (Dict)
            - correct_ids (List[str])
            - incorrect_ids (List[str])
    """
    category_performance = defaultdict(lambda: {"total": 0, "correct": 0})
    all_questions = 0
    all_correct = 0
    correct_ids = []
    incorrect_ids = []

    with open(results_file, "r") as f:
        lines = f.readlines()

    processed_questions = 0

    for line in tqdm(lines, desc="Analyzing Benchmark Results"):
        # Check if we've hit the maximum questions
        if max_questions is not None and processed_questions >= max_questions:
            break
        if line.startswith("HTTP Request:"):
            continue

        try:
            entry = json.loads(line)
            metadata = entry.get("input", {}).get("question_data", {}).get("metadata", {})
            question_id = entry.get("question_id")

            model_letter = extract_answer_letter(entry.get("model_answer"))
            correct_letter = extract_answer_letter(entry.get("correct_answer"))

            if model_letter and correct_letter:
                all_questions += 1
                processed_questions += 1
                is_correct = model_letter == correct_letter

                if is_correct:
                    all_correct += 1
                    correct_ids.append(question_id)
                else:
                    incorrect_ids.append(question_id)

                for category in metadata.get("categories", []):
                    category_performance[category]["total"] += 1
                    if is_correct:
                        category_performance[category]["correct"] += 1

        except json.JSONDecodeError:
            continue

    return process_results(
        category_performance, all_questions, all_correct, correct_ids, incorrect_ids
    )


def analyze_llama_results(
    results_file: str, max_questions: Optional[int] = None
) -> Tuple[float, Dict, Dict, List[str], List[str]]:
    """
    Analyze results in Llama format.

    Args:
        results_file: Path to results file
        max_questions: Maximum number of questions to analyze

    Returns:
        Tuple containing:
            - overall_accuracy (float)
            - category_accuracies (Dict)
            - question_type_stats (Dict)
            - correct_ids (List[str])
            - incorrect_ids (List[str])
    """
    category_performance = defaultdict(lambda: {"total": 0, "correct": 0})
    all_questions = 0
    all_correct = 0
    correct_ids = []
    incorrect_ids = []

    with open(results_file, "r") as f:
        lines = f.readlines()

    # If max_questions is set, limit the number of lines processed
    if max_questions is not None:
        lines = lines[:max_questions]

    for line in tqdm(lines, desc="Analyzing Benchmark Results"):
        if line.startswith("HTTP Request:"):
            continue

        try:
            entry = json.loads(line)
            metadata = entry.get("input", {}).get("question_data", {}).get("metadata", {})
            question_id = entry.get("question_id")

            model_letter = extract_answer_letter(entry.get("model_answer"))
            correct_letter = extract_answer_letter(entry.get("correct_answer"))

            if model_letter and correct_letter:
                all_questions += 1
                is_correct = model_letter == correct_letter

                if is_correct:
                    all_correct += 1
                    correct_ids.append(question_id)
                else:
                    incorrect_ids.append(question_id)

                for category in metadata.get("categories", []):
                    category_performance[category]["total"] += 1
                    if is_correct:
                        category_performance[category]["correct"] += 1

        except json.JSONDecodeError:
            continue

    return process_results(
        category_performance, all_questions, all_correct, correct_ids, incorrect_ids
    )


def analyze_chexagent_results(
    results_file: str, max_questions: Optional[int] = None
) -> Tuple[float, Dict, Dict, List[str], List[str]]:
    """
    Analyze results in CheXagent format.

    Args:
        results_file: Path to results file
        max_questions: Maximum number of questions to analyze

    Returns:
        Tuple containing:
            - overall_accuracy (float)
            - category_accuracies (Dict)
            - question_type_stats (Dict)
            - correct_ids (List[str])
            - incorrect_ids (List[str])
    """
    category_performance = defaultdict(lambda: {"total": 0, "correct": 0})
    all_questions = 0
    all_correct = 0
    correct_ids = []
    incorrect_ids = []

    with open(results_file, "r") as f:
        lines = f.readlines()

    # If max_questions is set, limit the number of lines processed
    if max_questions is not None:
        lines = lines[:max_questions]

    for line in tqdm(lines, desc="Analyzing Benchmark Results"):
        try:
            entry = json.loads(line)
            metadata = entry.get("input", {}).get("question_data", {}).get("metadata", {})
            question_id = entry.get("question_id")

            model_letter = extract_answer_letter(entry.get("model_answer"))
            correct_letter = extract_answer_letter(entry.get("correct_answer"))

            if model_letter and correct_letter:
                all_questions += 1
                is_correct = model_letter == correct_letter

                if is_correct:
                    all_correct += 1
                    correct_ids.append(question_id)
                else:
                    incorrect_ids.append(question_id)

                for category in metadata.get("categories", []):
                    category_performance[category]["total"] += 1
                    if is_correct:
                        category_performance[category]["correct"] += 1

        except json.JSONDecodeError:
            continue

    return process_results(
        category_performance, all_questions, all_correct, correct_ids, incorrect_ids
    )


def process_results(
    category_performance: Dict,
    all_questions: int,
    all_correct: int,
    correct_ids: Optional[List[str]] = None,
    incorrect_ids: Optional[List[str]] = None,
) -> Tuple[float, Dict, Dict, List[str], List[str]]:
    """
    Process raw results into final statistics.

    Args:
        category_performance: Dict containing performance by category
        all_questions: Total number of questions
        all_correct: Total number of correct answers
        correct_ids: List of IDs for correctly answered questions
        incorrect_ids: List of IDs for incorrectly answered questions

    Returns:
        Tuple containing:
            - overall_accuracy (float)
            - category_accuracies (Dict)
            - question_type_stats (Dict)
            - correct_ids (List[str])
            - incorrect_ids (List[str])
    """
    category_accuracies = {
        category: {
            "accuracy": stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0,
            "total": stats["total"],
            "correct": stats["correct"],
        }
        for category, stats in category_performance.items()
    }

    question_type_stats = {}
    for qtype, categories in QUESTION_TYPES.items():
        total = sum(
            category_performance[cat]["total"] for cat in categories if cat in category_performance
        )
        correct = sum(
            category_performance[cat]["correct"]
            for cat in categories
            if cat in category_performance
        )

        question_type_stats[qtype] = {
            "accuracy": (correct / total * 100) if total > 0 else 0,
            "total": total,
            "correct": correct,
        }

    overall_accuracy = (all_correct / all_questions * 100) if all_questions > 0 else 0

    return (
        overall_accuracy,
        category_accuracies,
        question_type_stats,
        correct_ids or [],
        incorrect_ids or [],
    )


def print_analysis(
    overall_accuracy: float,
    category_accuracies: Dict,
    question_type_stats: Dict,
    correct_ids: List[str],
    incorrect_ids: List[str],
    model_name: str,
) -> None:
    """
    Print analysis results.

    Args:
        overall_accuracy: Overall accuracy percentage
        category_accuracies: Dict containing accuracy metrics by category
        question_type_stats: Dict containing stats by question type
        correct_ids: List of IDs for correctly answered questions
        incorrect_ids: List of IDs for incorrectly answered questions
        model_name: Name of the model being analyzed
    """
    total_questions = len(correct_ids) + len(incorrect_ids)
    print(
        f"\nOverall Accuracy: {overall_accuracy:.2f}% ({len(correct_ids)} correct out of {total_questions} questions)"
    )

    print("\nCategory Performance:")
    sorted_categories = sorted(
        category_accuracies.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )
    for category, metrics in sorted_categories:
        print(f"{category}:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Total Questions: {metrics['total']}")
        print(f"  Correct Questions: {metrics['correct']}")

    print("\nQuestion Type Performance:")
    sorted_types = sorted(question_type_stats.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for qtype, metrics in sorted_types:
        print(f"\n{qtype}:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Total Questions: {metrics['total']}")
        print(f"  Correct Questions: {metrics['correct']}")
        print(f"  Categories: {', '.join(QUESTION_TYPES[qtype])}")

    # Save question IDs to JSON
    question_ids = {"correct_ids": correct_ids, "incorrect_ids": incorrect_ids}

    output_filename = f"{model_name}_question_ids.json"
    with open(output_filename, "w") as f:
        json.dump(question_ids, f, indent=2)

    print(f"\nQuestion IDs have been saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("results_file", help="Path to results file")
    parser.add_argument("benchmark_dir", nargs="?", help="Path to benchmark questions directory")
    parser.add_argument(
        "--model",
        choices=["llava-med", "chexagent", "llama", "gpt4", "medrax"],
        default="gpt4",
        help="Specify model format (default: gpt4)",
    )
    parser.add_argument("--max-questions", type=int, help="Maximum number of questions to analyze")
    args = parser.parse_args()

    if args.model == "gpt4":
        results = analyze_gpt4_results(args.results_file, args.max_questions)
    elif args.model == "llama":
        results = analyze_llama_results(args.results_file, args.max_questions)
    elif args.model == "chexagent":
        results = analyze_chexagent_results(args.results_file, args.max_questions)
    elif args.model == "medrax":
        results = analyze_gpt4_results(args.results_file, args.max_questions)
    else:
        parser.error(f"Unsupported model: {args.model}")

    print_analysis(*results, args.model)
