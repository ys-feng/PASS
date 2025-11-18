import json
import openai
import os
from datetime import datetime
import base64
import logging
from pathlib import Path
import time
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any

# Configuration constants
DEBUG_MODE = False
OUTPUT_DIR = "results"
MODEL_NAME = "gpt-4o-2024-05-13"
TEMPERATURE = 0.2
SUBSET = "Visual Question Answering"

# Set up logging configuration
logging_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_mime_type(file_path: str) -> str:
    """
    Determine MIME type based on file extension.

    Args:
        file_path (str): Path to the file

    Returns:
        str: MIME type string for the file
    """
    extension = os.path.splitext(file_path)[1].lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
    }
    return mime_types.get(extension, "application/octet-stream")


def encode_image(image_path: str) -> str:
    """
    Encode image to base64 with extensive error checking.

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Base64 encoded image string

    Raises:
        FileNotFoundError: If image file does not exist
        ValueError: If image file is empty or too large
        Exception: For other image processing errors
    """
    logger.debug(f"Attempting to read image from: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Add check for file size
    file_size = os.path.getsize(image_path)
    if file_size > 20 * 1024 * 1024:  # 20MB limit
        raise ValueError("Image file size exceeds 20MB limit")
    if file_size == 0:
        raise ValueError("Image file is empty")
    logger.debug(f"Image file size: {file_size / 1024:.2f} KB")

    try:
        from PIL import Image

        # Try to open and verify the image
        with Image.open(image_path) as img:
            # Get image details
            width, height = img.size
            format = img.format
            mode = img.mode
            logger.debug(
                f"Image verification - Format: {format}, Size: {width}x{height}, Mode: {mode}"
            )

            if format not in ["PNG", "JPEG", "GIF"]:
                raise ValueError(f"Unsupported image format: {format}")

        with open(image_path, "rb") as image_file:
            # Read the first few bytes to verify it's a valid PNG
            header = image_file.read(8)
            # if header != b'\x89PNG\r\n\x1a\n':
            #     logger.warning("File does not have a valid PNG signature")

            # Reset file pointer and read entire file
            image_file.seek(0)
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
            encoded_length = len(encoded)
            logger.debug(f"Base64 encoded length: {encoded_length} characters")

            # Verify the encoded string is not empty and starts correctly
            if encoded_length == 0:
                raise ValueError("Base64 encoding produced empty string")
            if not encoded.startswith("/9j/") and not encoded.startswith("iVBOR"):
                logger.warning("Base64 string doesn't start with expected JPEG or PNG header")

            return encoded
    except Exception as e:
        logger.error(f"Error reading/encoding image: {str(e)}")
        raise


def create_single_request(
    image_path: str, question: str, options: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Create a single API request with image and question.

    Args:
        image_path (str): Path to the image file
        question (str): Question text
        options (Dict[str, str]): Dictionary containing options with keys 'option_0' and 'option_1'

    Returns:
        List[Dict[str, Any]]: List of message dictionaries for the API request

    Raises:
        Exception: For errors in request creation
    """
    if DEBUG_MODE:
        logger.debug("Creating API request...")

    prompt = f"""Given the following medical examination question:
Please answer this multiple choice question:

Question: {question}

Options:
A) {options['option_0']}
B) {options['option_1']}

Base your answer only on the provided image and select either A or B."""

    try:
        encoded_image = encode_image(image_path)
        mime_type = get_mime_type(image_path)

        if DEBUG_MODE:
            logger.debug(f"Image encoded with MIME type: {mime_type}")

        messages = [
            {
                "role": "system",
                "content": "You are taking a medical exam. Answer ONLY with the letter (A/B) corresponding to your answer.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
                    },
                ],
            },
        ]

        if DEBUG_MODE:
            log_messages = json.loads(json.dumps(messages))
            log_messages[1]["content"][1]["image_url"][
                "url"
            ] = f"data:{mime_type};base64,[BASE64_IMAGE_TRUNCATED]"
            logger.debug(f"Complete API request payload:\n{json.dumps(log_messages, indent=2)}")

        return messages

    except Exception as e:
        logger.error(f"Error creating request: {str(e)}")
        raise


def check_answer(model_answer: str, correct_answer: int) -> bool:
    """
    Check if the model's answer matches the correct answer.

    Args:
        model_answer (str): The model's answer (A or B)
        correct_answer (int): The correct answer index (0 for A, 1 for B)

    Returns:
        bool: True if answer is correct, False otherwise
    """
    if not isinstance(model_answer, str):
        return False

    # Clean the model answer to get just the letter
    model_letter = model_answer.strip().upper()
    if model_letter.startswith("A"):
        model_index = 0
    elif model_letter.startswith("B"):
        model_index = 1
    else:
        return False

    return model_index == correct_answer


def save_results_to_json(results: List[Dict[str, Any]], output_dir: str) -> str:
    """
    Save results to a JSON file with timestamp.

    Args:
        results (List[Dict[str, Any]]): List of result dictionaries
        output_dir (str): Directory to save results

    Returns:
        str: Path to the saved file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"batch_results_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Batch results saved to {output_file}")
    return output_file


def calculate_accuracy(results: List[Dict[str, Any]]) -> tuple[float, int, int]:
    """
    Calculate accuracy from results, handling error cases.

    Args:
        results (List[Dict[str, Any]]): List of result dictionaries

    Returns:
        tuple[float, int, int]: Tuple containing (accuracy percentage, number correct, total)
    """
    if not results:
        return 0.0, 0, 0

    total = len(results)
    valid_results = [r for r in results if "output" in r]
    correct = sum(
        1 for result in valid_results if result.get("output", {}).get("is_correct", False)
    )

    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total


def calculate_batch_accuracy(results: List[Dict[str, Any]]) -> float:
    """
    Calculate accuracy for the current batch.

    Args:
        results (List[Dict[str, Any]]): List of result dictionaries

    Returns:
        float: Accuracy percentage for the batch
    """
    valid_results = [r for r in results if "output" in r]
    if not valid_results:
        return 0.0
    return sum(1 for r in valid_results if r["output"]["is_correct"]) / len(valid_results) * 100


def process_batch(
    data: List[Dict[str, Any]], client: openai.OpenAI, start_idx: int = 0, batch_size: int = 50
) -> List[Dict[str, Any]]:
    """
    Process a batch of examples and return results.

    Args:
        data (List[Dict[str, Any]]): List of data items to process
        client (openai.OpenAI): OpenAI client instance
        start_idx (int, optional): Starting index for batch. Defaults to 0
        batch_size (int, optional): Size of batch to process. Defaults to 50

    Returns:
        List[Dict[str, Any]]: List of processed results
    """
    batch_results = []
    end_idx = min(start_idx + batch_size, len(data))

    pbar = tqdm(
        range(start_idx, end_idx),
        desc=f"Processing batch {start_idx//batch_size + 1}",
        unit="example",
    )

    for index in pbar:
        vqa_item = data[index]
        options = {"option_0": vqa_item["option_0"], "option_1": vqa_item["option_1"]}

        try:
            messages = create_single_request(
                image_path=vqa_item["image_path"], question=vqa_item["question"], options=options
            )

            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, max_tokens=50, temperature=TEMPERATURE
            )

            model_answer = response.choices[0].message.content.strip()
            is_correct = check_answer(model_answer, vqa_item["answer"])

            result = {
                "timestamp": datetime.now().isoformat(),
                "example_index": index,
                "input": {
                    "question": vqa_item["question"],
                    "options": {"A": vqa_item["option_0"], "B": vqa_item["option_1"]},
                    "image_path": vqa_item["image_path"],
                },
                "output": {
                    "model_answer": model_answer,
                    "correct_answer": "A" if vqa_item["answer"] == 0 else "B",
                    "is_correct": is_correct,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                },
            }
            batch_results.append(result)

            # Update progress bar with current accuracy
            current_accuracy = calculate_batch_accuracy(batch_results)
            pbar.set_description(
                f"Batch {start_idx//batch_size + 1} - Accuracy: {current_accuracy:.2f}% "
                f"({len(batch_results)}/{index-start_idx+1} examples)"
            )

        except Exception as e:
            error_result = {
                "timestamp": datetime.now().isoformat(),
                "example_index": index,
                "error": str(e),
                "input": {
                    "question": vqa_item["question"],
                    "options": {"A": vqa_item["option_0"], "B": vqa_item["option_1"]},
                    "image_path": vqa_item["image_path"],
                },
            }
            batch_results.append(error_result)
            if DEBUG_MODE:
                pbar.write(f"Error processing example {index}: {str(e)}")

        time.sleep(1)  # Rate limiting

    return batch_results


def main() -> None:
    """
    Main function to process the entire dataset.

    Raises:
        ValueError: If OPENAI_API_KEY is not set
        Exception: For other processing errors
    """
    logger.info("Starting full dataset processing...")
    json_path = "../data/chexbench_updated.json"

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        client = openai.OpenAI(api_key=api_key)

        with open(json_path, "r") as f:
            data = json.load(f)

        subset_data = data[SUBSET]
        total_examples = len(subset_data)
        logger.info(f"Found {total_examples} examples in {SUBSET} subset")

        all_results = []
        batch_size = 50  # Process in batches of 50 examples

        # Process all examples in batches
        for start_idx in range(0, total_examples, batch_size):
            batch_results = process_batch(subset_data, client, start_idx, batch_size)
            all_results.extend(batch_results)

            # Save intermediate results after each batch
            output_file = save_results_to_json(all_results, OUTPUT_DIR)

            # Calculate and log overall progress
            overall_accuracy, correct, total = calculate_accuracy(all_results)
            logger.info(f"Overall Progress: {len(all_results)}/{total_examples} examples processed")
            logger.info(f"Current Accuracy: {overall_accuracy:.2f}% ({correct}/{total} correct)")

        logger.info("Processing completed!")
        logger.info(f"Final results saved to: {output_file}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
