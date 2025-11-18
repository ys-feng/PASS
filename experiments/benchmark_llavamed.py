import argparse
import json
import requests
import base64
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates
import time
import os
import glob
import logging
from datetime import datetime
from tqdm import tqdm
import re
from typing import Dict, List, Optional, Union, Any, Tuple


def process_image(image_path: str, target_size: int = 640) -> Image.Image:
    """Process and resize an image to match model requirements.

    Args:
        image_path: Path to the input image file
        target_size: Target size for both width and height in pixels

    Returns:
        PIL.Image: Processed and padded image with dimensions (target_size, target_size)
    """
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Calculate scaling to maintain aspect ratio
    ratio = min(target_size / image.width, target_size / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))

    # Resize image
    image = image.resize(new_size, Image.LANCZOS)

    # Create new image with padding
    new_image = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    # Paste resized image in center
    offset = ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2)
    new_image.paste(image, offset)

    return new_image


def validate_answer(response_text: str) -> Optional[str]:
    """Extract and validate a single-letter response from the model's output.
    Handles multiple response formats and edge cases.

    Args:
        response_text: The full text output from the model

    Returns:
        A single letter answer (A-F) or None if no valid answer found
    """
    if not response_text:
        return None

    # Clean the response text
    cleaned = response_text.strip()

    # Comprehensive set of patterns to extract the answer
    extraction_patterns = [
        # Strict format with explicit letter answer
        r"(?:THE\s*)?(?:SINGLE\s*)?LETTER\s*(?:ANSWER\s*)?(?:IS:?)\s*([A-F])\b",
        # Patterns for extracting from longer descriptions
        r"(?:correct\s+)?(?:answer|option)\s*(?:is\s*)?([A-F])\b",
        r"\b(?:answer|option)\s*([A-F])[):]\s*",
        # Patterns for extracting from descriptive sentences
        r"(?:most\s+likely\s+)?(?:answer|option)\s*(?:is\s*)?([A-F])\b",
        r"suggest[s]?\s+(?:that\s+)?(?:the\s+)?(?:answer\s+)?(?:is\s*)?([A-F])\b",
        # Patterns with contextual words
        r"characteriz[e]?d?\s+by\s+([A-F])\b",
        r"indicat[e]?s?\s+([A-F])\b",
        # Fallback to Option X or Letterr X formats
        r"Option\s*([A-F])\b",
        r"\b([A-F])\)\s*",
        # Fallback to standalone letter
        r"^\s*([A-F])\s*$",
    ]

    # Try each pattern
    for pattern in extraction_patterns:
        matches = re.findall(pattern, cleaned, re.IGNORECASE)
        for match in matches:
            # Ensure match is a single valid letter
            if isinstance(match, tuple):
                match = match[0] if match[0] in "ABCDEF" else None
            if match and match.upper() in "ABCDEF":
                return match.upper()

    # Final fallback: look for standalone letters in context
    context_matches = re.findall(r"\b([A-F])\b", cleaned.upper())
    context_letters = [m for m in context_matches if m in "ABCDEF"]
    if context_letters:
        return context_letters[0]

    # No valid answer found
    return None


def load_benchmark_questions(case_id: str) -> List[str]:
    """Find all question files for a given case ID.

    Args:
        case_id: The ID of the medical case

    Returns:
        List of paths to question JSON files
    """
    benchmark_dir = "MedMAX/benchmark/questions"
    return glob.glob(f"{benchmark_dir}/{case_id}/{case_id}_*.json")


def count_total_questions() -> Tuple[int, int]:
    """Count total number of cases and questions in benchmark.

    Returns:
        Tuple containing (total_cases, total_questions)
    """
    total_cases = len(glob.glob("MedMAX/benchmark/questions/*"))
    total_questions = sum(
        len(glob.glob(f"MedMAX/benchmark/questions/{case_id}/*.json"))
        for case_id in os.listdir("MedMAX/benchmark/questions")
    )
    return total_cases, total_questions


def create_inference_request(
    question_data: Dict[str, Any],
    case_details: Dict[str, Any],
    case_id: str,
    question_id: str,
    worker_addr: str,
    model_name: str,
    raw_output: bool = False,
) -> Union[Tuple[Optional[str], Optional[float]], Dict[str, Any]]:
    """Create and send inference request to worker.

    Args:
        question_data: Dictionary containing question details and figures
        case_details: Dictionary containing case information and figures
        case_id: Identifier for the medical case
        question_id: Identifier for the specific question
        worker_addr: Address of the worker endpoint
        model_name: Name of the model to use
        raw_output: Whether to return raw model output

    Returns:
        If raw_output is False: Tuple of (validated_answer, duration)
        If raw_output is True: Dictionary with full inference details
    """
    system_prompt = """You are a medical imaging expert. Your answer MUST be a SINGLE LETTER (A/B/C/D/E/F), provided in this format: 'The SINGLE LETTER answer is: X'.
"""

    prompt = f"""Given the following medical case:
Please answer this multiple choice question:
{question_data['question']}
Base your answer only on the provided images and case information. Respond with your SINGLE LETTER answer: """

    try:
        # Parse required figures
        if isinstance(question_data["figures"], str):
            try:
                required_figures = json.loads(question_data["figures"])
            except json.JSONDecodeError:
                required_figures = [question_data["figures"]]
        elif isinstance(question_data["figures"], list):
            required_figures = question_data["figures"]
        else:
            required_figures = [str(question_data["figures"])]
    except Exception as e:
        print(f"Error parsing figures: {e}")
        required_figures = []

    required_figures = [
        fig if fig.startswith("Figure ") else f"Figure {fig}" for fig in required_figures
    ]

    # Get image paths
    image_paths = []
    for figure in required_figures:
        base_figure_num = "".join(filter(str.isdigit, figure))
        figure_letter = "".join(filter(str.isalpha, figure.split()[-1])) or None

        matching_figures = [
            case_figure
            for case_figure in case_details.get("figures", [])
            if case_figure["number"] == f"Figure {base_figure_num}"
        ]

        for case_figure in matching_figures:
            subfigures = []
            if figure_letter:
                subfigures = [
                    subfig
                    for subfig in case_figure.get("subfigures", [])
                    if subfig.get("number", "").lower().endswith(figure_letter.lower())
                    or subfig.get("label", "").lower() == figure_letter.lower()
                ]
            else:
                subfigures = case_figure.get("subfigures", [])

            for subfig in subfigures:
                if "local_path" in subfig:
                    image_paths.append("MedMAX/data/" + subfig["local_path"])

    if not image_paths:
        print(f"No local images found for case {case_id}, question {question_id}")
        return "skipped", 0.0  # Return a special 'skipped' marker

    try:
        start_time = time.time()

        # Process each image
        processed_images = [process_image(path) for path in image_paths]

        # Create conversation
        conv = conv_templates["mistral_instruct"].copy()

        # Add image and message
        if "<image>" not in prompt:
            text = prompt + "\n<image>"
        else:
            text = prompt

        message = (text, processed_images[0], "Default")  # Currently handling first image
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        headers = {"User-Agent": "LLaVA-Med Client"}
        pload = {
            "model": model_name,
            "prompt": prompt,
            "max_new_tokens": 150,  # Reduce this since we only need one letter
            "temperature": 0.5,  # Lower temperature for more focused responses
            "stop": conv.sep2,
            "images": conv.get_images(),
            "top_p": 1,  # Lower top_p for more focused sampling
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        max_retries = 3
        retry_delay = 5
        response_text = None

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    worker_addr + "/worker_generate_stream",
                    headers=headers,
                    json=pload,
                    stream=True,
                    timeout=30,
                )

                complete_output = ""
                for chunk in response.iter_lines(
                    chunk_size=8192, decode_unicode=False, delimiter=b"\0"
                ):
                    if chunk:
                        data = json.loads(chunk.decode("utf-8"))
                        if data["error_code"] == 0:
                            output = data["text"].split("[/INST]")[-1]
                            complete_output = output
                        else:
                            print(f"\nError: {data['text']} (error_code: {data['error_code']})")
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                break
                            return None, None

                if complete_output:
                    response_text = complete_output
                    break

            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    print(f"\nNetwork error: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"\nFailed after {max_retries} attempts: {str(e)}")
                    return None, None

        duration = time.time() - start_time

        if raw_output:
            inference_details = {
                "raw_output": response_text,
                "validated_answer": validate_answer(response_text),
                "duration": duration,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "image_paths": image_paths,
                "payload": pload,
            }
            return inference_details

        return validate_answer(response_text), duration

    except Exception as e:
        print(f"Error in inference request: {str(e)}")
        return None, None


def clean_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Remove image-related and large data from the payload to keep the log lean.

    Args:
        payload: Original request payload dictionary

    Returns:
        Cleaned payload dictionary with large data removed
    """
    if not payload:
        return None

    # Create a copy of the payload to avoid modifying the original
    cleaned_payload = payload.copy()

    # Remove large or sensitive data
    if "images" in cleaned_payload:
        del cleaned_payload["images"]

    return cleaned_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--worker-address", type=str)
    parser.add_argument("--model-name", type=str, default="llava-med-v1.5-mistral-7b")
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    parser.add_argument(
        "--raw-output", action="store_true", help="Return raw model output without validation"
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        help="Number of cases to process if looking at raw outputs",
        default=2,
    )
    args = parser.parse_args()

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup live logging files
    live_log_filename = os.path.join(args.output_dir, f"live_benchmark_log_{timestamp}.json")
    final_results_filename = os.path.join(args.output_dir, f"final_results_{timestamp}.json")

    # Initialize live log file
    with open(live_log_filename, "w") as live_log_file:
        live_log_file.write("[\n")  # Start of JSON array

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, f"benchmark_{timestamp}.log"),
        level=logging.INFO,
        format="%(message)s",
    )

    # Get worker address
    if args.worker_address:
        worker_addr = args.worker_address
    else:
        try:
            requests.post(args.controller_address + "/refresh_all_workers")
            ret = requests.post(args.controller_address + "/list_models")
            models = ret.json()["models"]
            ret = requests.post(
                args.controller_address + "/get_worker_address", json={"model": args.model_name}
            )
            worker_addr = ret.json()["address"]
            print(f"Worker address: {worker_addr}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to controller: {e}")
            return

    if worker_addr == "":
        print("No available worker")
        return

    # Load cases with local paths
    with open("MedMAX/data/updated_cases.json", "r") as file:
        data = json.load(file)

    total_cases, total_questions = count_total_questions()
    print(f"\nStarting benchmark with {args.model_name}")
    print(f"Found {total_cases} cases with {total_questions} total questions")

    results = {
        "model": args.model_name,
        "timestamp": datetime.now().isoformat(),
        "total_cases": total_cases,
        "total_questions": total_questions,
        "results": [],
    }

    cases_processed = 0
    questions_processed = 0
    correct_answers = 0
    skipped_questions = 0
    total_processed_entries = 0

    # Process each case
    for case_id, case_details in tqdm(data.items(), desc="Processing cases"):
        question_files = load_benchmark_questions(case_id)
        if not question_files:
            continue

        cases_processed += 1
        for question_file in tqdm(
            question_files, desc=f"Processing questions for case {case_id}", leave=False
        ):
            with open(question_file, "r") as file:
                question_data = json.load(file)
                question_id = os.path.basename(question_file).split(".")[0]

            questions_processed += 1

            # Get model's answer
            inference_result = create_inference_request(
                question_data,
                case_details,
                case_id,
                question_id,
                worker_addr,
                args.model_name,
                raw_output=True,  # Always use raw output for detailed logging
            )

            # Handle skipped questions
            if inference_result == ("skipped", 0.0):
                skipped_questions += 1
                print(f"\nCase {case_id}, Question {question_id}: Skipped (No images)")

                # Log skipped question
                skipped_entry = {
                    "case_id": case_id,
                    "question_id": question_id,
                    "status": "skipped",
                    "reason": "No images found",
                }
                with open(live_log_filename, "a") as live_log_file:
                    json.dump(skipped_entry, live_log_file, indent=2)
                    live_log_file.write(",\n")  # Add comma for next entry

                continue

            # Extract information
            answer = inference_result["validated_answer"]
            duration = inference_result["duration"]

            # Prepare detailed logging entry
            log_entry = {
                "case_id": case_id,
                "question_id": question_id,
                "question": question_data["question"],
                "correct_answer": question_data["answer"],
                "raw_output": inference_result["raw_output"],
                "validated_answer": answer,
                "model_answer": answer,
                "is_correct": answer == question_data["answer"] if answer else False,
                "duration": duration,
                "system_prompt": inference_result["system_prompt"],
                "input_prompt": inference_result["prompt"],
                "image_paths": inference_result["image_paths"],
                "payload": clean_payload(inference_result["payload"]),
            }

            # Write to live log file
            with open(live_log_filename, "a") as live_log_file:
                json.dump(log_entry, live_log_file, indent=2)
                live_log_file.write(",\n")  # Add comma for next entry

            # Print to console
            print(f"\nCase {case_id}, Question {question_id}")
            print(f"Model Answer: {answer}")
            print(f"Correct Answer: {question_data['answer']}")
            print(f"Time taken: {duration:.2f}s")

            # Track correct answers
            if answer == question_data["answer"]:
                correct_answers += 1

            # Append to results
            results["results"].append(log_entry)
            total_processed_entries += 1

            # Optional: break if reached specified number of cases
            if args.raw_output and cases_processed == args.num_cases:
                break

        # Optional: break if reached specified number of cases
        if args.raw_output and cases_processed == args.num_cases:
            break

    # Close live log file
    with open(live_log_filename, "a") as live_log_file:
        # Remove trailing comma and close JSON array
        live_log_file.seek(live_log_file.tell() - 2, 0)  # Go back 2 chars to remove ',\n'
        live_log_file.write("\n]")

    # Calculate final statistics
    results["summary"] = {
        "cases_processed": cases_processed,
        "questions_processed": questions_processed,
        "total_processed_entries": total_processed_entries,
        "correct_answers": correct_answers,
        "skipped_questions": skipped_questions,
        "accuracy": (
            correct_answers / (questions_processed - skipped_questions)
            if (questions_processed - skipped_questions) > 0
            else 0
        ),
    }

    # Save final results
    with open(final_results_filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark Summary:")
    print(f"Total Cases Processed: {cases_processed}")
    print(f"Total Questions Processed: {questions_processed}")
    print(f"Total Processed Entries: {total_processed_entries}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Skipped Questions: {skipped_questions}")
    print(f"Accuracy: {(correct_answers / (questions_processed - skipped_questions) * 100):.2f}%")
    print(f"\nResults saved to {args.output_dir}")
    print(f"Live log: {live_log_filename}")
    print(f"Final results: {final_results_filename}")


if __name__ == "__main__":
    main()
