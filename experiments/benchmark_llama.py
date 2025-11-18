from typing import Dict, List, Optional, Any, Union
import re
import json
import os
import glob
import time
import logging
import socket
import requests
import httpx
import backoff
from datetime import datetime
from tenacity import retry, wait_exponential, stop_after_attempt
from openai import OpenAI

# Configure model settings
MODEL_NAME = "meta-llama/llama-3.2-90b-vision-instruct"
temperature = 0.2

# Configure logging
log_filename = f"api_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(message)s")


def verify_dns() -> bool:
    """Verify DNS resolution and connectivity.

    Returns:
        bool: True if DNS resolution succeeds, False otherwise
    """
    try:
        # Try to resolve openrouter.ai
        socket.gethostbyname("openrouter.ai")
        return True
    except socket.gaierror:
        print("DNS resolution failed. Trying to use Google DNS (8.8.8.8)...")
        # Modify resolv.conf to use Google DNS
        try:
            with open("/etc/resolv.conf", "w") as f:
                f.write("nameserver 8.8.8.8\n")
            return True
        except Exception as e:
            print(f"Failed to update DNS settings: {e}")
            return False


def verify_connection() -> bool:
    """Verify connection to OpenRouter API.

    Returns:
        bool: True if connection succeeds, False otherwise
    """
    try:
        response = requests.get("https://openrouter.ai/api/v1/status", timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False


def initialize_client() -> OpenAI:
    """Initialize the OpenRouter client with proper timeout settings and connection verification.

    Returns:
        OpenAI: Configured OpenAI client for OpenRouter

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set
        ConnectionError: If DNS verification or connection test fails
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

    # Configure timeout settings for the client
    timeout_settings = 120  # Increased timeout for large images/responses

    # Verify DNS and connection
    if not verify_dns():
        raise ConnectionError("DNS verification failed. Please check your network settings.")

    if not verify_connection():
        raise ConnectionError(
            "Cannot connect to OpenRouter. Please check your internet connection."
        )

    # Set up client with retry and timeout settings
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=timeout_settings,
        http_client=httpx.Client(
            timeout=timeout_settings, transport=httpx.HTTPTransport(retries=3)
        ),
    )


@backoff.on_exception(
    backoff.expo,
    (ConnectionError, TimeoutError, socket.gaierror, httpx.ConnectError),
    max_tries=5,
    max_time=300,  # Maximum total time to try in seconds
)
def create_multimodal_request(
    question_data: Dict[str, Any],
    case_details: Dict[str, Any],
    case_id: str,
    question_id: str,
    client: OpenAI,
) -> Optional[Any]:
    """Create and send a multimodal request to the model.

    Args:
        question_data: Dictionary containing question details
        case_details: Dictionary containing case information
        case_id: ID of the medical case
        question_id: ID of the specific question
        client: OpenAI client instance

    Returns:
        Optional[Any]: Model response if successful, None if skipped

    Raises:
        ConnectionError: If connection fails
        TimeoutError: If request times out
        Exception: For other errors
    """

    system_prompt = """You are a medical imaging expert. Your task is to provide ONLY a single letter answer.
Rules:
1. Respond with exactly one uppercase letter (A/B/C/D/E/F)
2. Do not add periods, explanations, or any other text
3. Do not use markdown or formatting
4. Do not restate the question
5. Do not explain your reasoning

Examples of valid responses:
A
B
C

Examples of invalid responses:
"A."
"Answer: B"
"C) This shows..."
"The answer is D"
"""

    prompt = f"""Given the following medical case:
Please answer this multiple choice question:
{question_data['question']}
Base your answer only on the provided images and case information."""

    # Parse required figures
    try:
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

    # Process subfigures and prepare content
    content = [{"type": "text", "text": prompt}]
    image_urls = []
    image_captions = []

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
                if "url" in subfig:
                    content.append({"type": "image_url", "image_url": {"url": subfig["url"]}})
                    image_urls.append(subfig["url"])
                    image_captions.append(subfig.get("caption", ""))

    if len(content) == 1:  # Only the text prompt exists
        print(f"No images found for case {case_id}, question {question_id}")
        # Log the skipped question
        log_entry = {
            "case_id": case_id,
            "question_id": question_id,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "status": "skipped",
            "reason": "no_images",
            "input": {
                "question_data": {
                    "question": question_data["question"],
                    "explanation": question_data["explanation"],
                    "metadata": question_data.get("metadata", {}),
                    "figures": question_data["figures"],
                },
                "image_urls": image_urls,
            },
        }
        logging.info(json.dumps(log_entry))
        return None

    try:
        start_time = time.time()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        duration = time.time() - start_time

        # Get raw response
        raw_answer = response.choices[0].message.content

        # Validate and clean
        clean_answer = validate_answer(raw_answer)

        if not clean_answer:
            print(f"Warning: Invalid response format for case {case_id}, question {question_id}")
            print(f"Raw response: {raw_answer}")

        # Update response object with cleaned answer
        response.choices[0].message.content = clean_answer

        # Log response
        log_entry = {
            "case_id": case_id,
            "question_id": question_id,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "temperature": temperature,
            "duration": round(duration, 2),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "model_answer": response.choices[0].message.content,
            "correct_answer": question_data["answer"],
            "input": {
                "question_data": {
                    "question": question_data["question"],
                    "explanation": question_data["explanation"],
                    "metadata": question_data.get("metadata", {}),
                    "figures": question_data["figures"],
                },
                "image_urls": image_urls,
            },
        }
        logging.info(json.dumps(log_entry))
        return response

    except ConnectionError as e:
        print(f"Connection error for case {case_id}, question {question_id}: {str(e)}")
        print("Retrying after a longer delay...")
        time.sleep(30)  # Add a longer delay before retry
        raise
    except TimeoutError as e:
        print(f"Timeout error for case {case_id}, question {question_id}: {str(e)}")
        print("Retrying with increased timeout...")
        raise
    except Exception as e:
        # Log failed requests too
        log_entry = {
            "case_id": case_id,
            "question_id": question_id,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "temperature": temperature,
            "status": "error",
            "error": str(e),
            "input": {
                "question_data": {
                    "question": question_data["question"],
                    "explanation": question_data["explanation"],
                    "metadata": question_data.get("metadata", {}),
                    "figures": question_data["figures"],
                },
                "image_urls": image_urls,
            },
        }
        logging.info(json.dumps(log_entry))
        raise


def extract_answer(response_text: str) -> Optional[str]:
    """Extract single letter answer from model response.

    Args:
        response_text: Raw text response from model

    Returns:
        Optional[str]: Single letter answer if found, None otherwise
    """
    # Convert to uppercase and remove periods
    text = response_text.upper().replace(".", "")

    # Look for common patterns
    patterns = [
        r"ANSWER:\s*([A-F])",  # Matches "ANSWER: X"
        r"OPTION\s*([A-F])",  # Matches "OPTION X"
        r"([A-F])\)",  # Matches "X)"
        r"\b([A-F])\b",  # Matches single letter
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]

    return None


def validate_answer(response_text: str) -> Optional[str]:
    """Enforce strict single-letter response format.

    Args:
        response_text: Raw text response from model

    Returns:
        Optional[str]: Valid single letter answer if found, None otherwise
    """
    if not response_text:
        return None

    # Remove all whitespace and convert to uppercase
    cleaned = response_text.strip().upper()

    # Check if it's exactly one valid letter
    if len(cleaned) == 1 and cleaned in "ABCDEF":
        return cleaned

    # If not, try to extract just the letter
    match = re.search(r"([A-F])", cleaned)
    return match.group(1) if match else None


def load_benchmark_questions(case_id: str) -> List[str]:
    """Find all question files for a given case ID.

    Args:
        case_id: ID of the medical case

    Returns:
        List[str]: List of paths to question files
    """
    benchmark_dir = "../benchmark/questions"
    return glob.glob(f"{benchmark_dir}/{case_id}/{case_id}_*.json")


def count_total_questions() -> Tuple[int, int]:
    """Count total number of cases and questions.

    Returns:
        Tuple[int, int]: (total_cases, total_questions)
    """
    total_cases = len(glob.glob("../benchmark/questions/*"))
    total_questions = sum(
        len(glob.glob(f"../benchmark/questions/{case_id}/*.json"))
        for case_id in os.listdir("../benchmark/questions")
    )
    return total_cases, total_questions


def main():
    with open("../data/eurorad_metadata.json", "r") as file:
        data = json.load(file)

    client = initialize_client()
    total_cases, total_questions = count_total_questions()
    cases_processed = 0
    questions_processed = 0
    skipped_questions = 0

    print(f"Beginning benchmark evaluation for {MODEL_NAME} with temperature {temperature}")

    for case_id, case_details in data.items():
        question_files = load_benchmark_questions(case_id)
        if not question_files:
            continue

        cases_processed += 1
        for question_file in question_files:
            with open(question_file, "r") as file:
                question_data = json.load(file)
                question_id = os.path.basename(question_file).split(".")[0]

            questions_processed += 1
            response = create_multimodal_request(
                question_data, case_details, case_id, question_id, client
            )

            if response is None:
                skipped_questions += 1
                print(f"Skipped question: Case ID {case_id}, Question ID {question_id}")
                continue

            print(
                f"Progress: Case {cases_processed}/{total_cases}, Question {questions_processed}/{total_questions}"
            )
            print(f"Case ID: {case_id}")
            print(f"Question ID: {question_id}")
            print(f"Model Answer: {response.choices[0].message.content}")
            print(f"Correct Answer: {question_data['answer']}\n")

    print(f"\nBenchmark Summary:")
    print(f"Total Cases Processed: {cases_processed}")
    print(f"Total Questions Processed: {questions_processed}")
    print(f"Total Questions Skipped: {skipped_questions}")


if __name__ == "__main__":
    main()
