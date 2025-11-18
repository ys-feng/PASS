from typing import Dict, List, Tuple, Optional
import json
import sys
import glob
from pathlib import Path
from collections import defaultdict


def get_latest_log() -> str:
    """Find the most recently modified log file in the current directory.

    Returns:
        str: Path to the most recently modified log file

    Raises:
        SystemExit: If no log files are found in current directory
    """
    log_pattern = "api_usage_*.json"
    logs = list(Path(".").glob(log_pattern))
    if not logs:
        print(f"No files matching pattern '{log_pattern}' found in current directory")
        sys.exit(1)
    return str(max(logs, key=lambda p: p.stat().st_mtime))


def analyze_log_file(filename: str) -> Tuple[List[Dict], List[Dict], Dict[str, List[str]]]:
    """Analyze a log file for entries missing images and errors.

    Args:
        filename: Path to the log file to analyze

    Returns:
        Tuple containing:
            - List of entries with no images
            - List of skipped/error entries
            - Dict of processing errors by type

    Raises:
        SystemExit: If file cannot be found or read
    """
    no_images = []
    errors = defaultdict(list)
    skipped = []

    try:
        with open(filename, "r") as f:
            for line_num, line in enumerate(f, 1):
                # Skip HTTP request logs
                if line.startswith("HTTP Request:") or line.strip() == "":
                    continue
                try:
                    # Try to parse the JSON line
                    if not line.strip().startswith("{"):
                        continue
                    entry = json.loads(line.strip())
                    case_id = entry.get("case_id")
                    question_id = entry.get("question_id")

                    # Skip if we can't identify the question
                    if not case_id or not question_id:
                        continue

                    # Check for explicit skip/error status
                    if entry.get("status") in ["skipped", "error"]:
                        skipped.append(
                            {
                                "case_id": case_id,
                                "question_id": question_id,
                                "reason": entry.get("reason"),
                                "status": entry.get("status"),
                            }
                        )
                        continue

                    # Check user content for images
                    messages = entry.get("input", {}).get("messages", [])
                    has_image = False
                    for msg in messages:
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "image_url":
                                    has_image = True
                                    break
                    if not has_image:
                        no_images.append(
                            {
                                "case_id": case_id,
                                "question_id": question_id,
                                "question": entry.get("input", {})
                                .get("question_data", {})
                                .get("question", "")[:100]
                                + "...",  # First 100 chars of question
                            }
                        )
                except json.JSONDecodeError:
                    errors["json_decode"].append(f"Line {line_num}: Invalid JSON")
                    continue
                except Exception as e:
                    errors["other"].append(f"Line {line_num}: Error processing entry: {str(e)}")
    except FileNotFoundError:
        print(f"Error: Could not find log file: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        sys.exit(1)

    return no_images, skipped, errors


def print_results(
    filename: str, no_images: List[Dict], skipped: List[Dict], errors: Dict[str, List[str]]
) -> None:
    """Print analysis results.

    Args:
        filename: Name of the analyzed log file
        no_images: List of entries with no images
        skipped: List of skipped/error entries
        errors: Dict of processing errors by type
    """
    print(f"\nAnalyzing log file: {filename}")
    print("\n=== Questions with No Images ===")
    if no_images:
        for entry in no_images:
            print(f"\nCase ID: {entry['case_id']}")
            print(f"Question ID: {entry['question_id']}")
            print(f"Question Preview: {entry['question']}")
    print(f"\nTotal questions without images: {len(no_images)}")

    print("\n=== Skipped/Error Questions ===")
    if skipped:
        for entry in skipped:
            print(f"\nCase ID: {entry['case_id']}")
            print(f"Question ID: {entry['question_id']}")
            print(f"Status: {entry['status']}")
            print(f"Reason: {entry.get('reason', 'unknown')}")
    print(f"\nTotal skipped/error questions: {len(skipped)}")

    if errors:
        print("\n=== Processing Errors ===")
        for error_type, messages in errors.items():
            if messages:
                print(f"\n{error_type}:")
                for msg in messages:
                    print(f"  {msg}")


def main() -> None:
    """Main entry point for log validation script."""
    # If a file is specified as an argument, use it; otherwise find the latest log
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = get_latest_log()

    no_images, skipped, errors = analyze_log_file(log_file)
    print_results(log_file, no_images, skipped, errors)


if __name__ == "__main__":
    main()
