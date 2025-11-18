import json
import openai
import os
import glob
import time
import logging
from datetime import datetime
from tenacity import retry, wait_exponential, stop_after_attempt

model_name = "chatgpt-4o-latest"
temperature = 0.2
log_filename = f"api_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(message)s")


def calculate_cost(
    prompt_tokens: int, completion_tokens: int, model: str = "chatgpt-4o-latest"
) -> float:
    """Calculate the cost of API usage based on token counts.

    Args:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        model: Model name to use for pricing, defaults to chatgpt-4o-latest

    Returns:
        float: Cost in USD
    """
    pricing = {"chatgpt-4o-latest": {"prompt": 5.0, "completion": 15.0}}
    rates = pricing.get(model, {"prompt": 5.0, "completion": 15.0})
    return (prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]) / 1000000


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def create_multimodal_request(
    question_data: dict, case_details: dict, case_id: str, question_id: str, client: openai.OpenAI
) -> openai.types.chat.ChatCompletion:
    """Create and send a multimodal request to the OpenAI API.

    Args:
        question_data: Dictionary containing question details and figures
        case_details: Dictionary containing case information and figures
        case_id: Identifier for the medical case
        question_id: Identifier for the specific question
        client: OpenAI client instance

    Returns:
        openai.types.chat.ChatCompletion: API response object, or None if request fails
    """
    prompt = f"""Given the following medical case:
Please answer this multiple choice question:
{question_data['question']}
Base your answer only on the provided images and case information."""

    content = [{"type": "text", "text": prompt}]

    # Parse required figures
    try:
        # Try multiple ways of parsing figures
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

    # Ensure each figure starts with "Figure "
    required_figures = [
        fig if fig.startswith("Figure ") else f"Figure {fig}" for fig in required_figures
    ]

    subfigures = []
    for figure in required_figures:
        # Handle both regular figures and those with letter suffixes
        base_figure_num = "".join(filter(str.isdigit, figure))
        figure_letter = "".join(filter(str.isalpha, figure.split()[-1])) or None

        # Find matching figures in case details
        matching_figures = [
            case_figure
            for case_figure in case_details.get("figures", [])
            if case_figure["number"] == f"Figure {base_figure_num}"
        ]

        if not matching_figures:
            print(f"No matching figure found for {figure} in case {case_id}")
            continue

        for case_figure in matching_figures:
            # If a specific letter is specified, filter subfigures
            if figure_letter:
                matching_subfigures = [
                    subfig
                    for subfig in case_figure.get("subfigures", [])
                    if subfig.get("number", "").lower().endswith(figure_letter.lower())
                    or subfig.get("label", "").lower() == figure_letter.lower()
                ]
                subfigures.extend(matching_subfigures)
            else:
                # If no letter specified, add all subfigures
                subfigures.extend(case_figure.get("subfigures", []))

    # Add images to content
    for subfig in subfigures:
        if "url" in subfig:
            content.append({"type": "image_url", "image_url": {"url": subfig["url"]}})
        else:
            print(f"Subfigure missing URL: {subfig}")

    # If no images found, log and return None
    if len(content) == 1:  # Only the text prompt exists
        print(f"No images found for case {case_id}, question {question_id}")
        return None

    messages = [
        {
            "role": "system",
            "content": "You are a medical imaging expert. Provide only the letter corresponding to your answer choice (A/B/C/D/E/F).",
        },
        {"role": "user", "content": content},
    ]

    if len(content) == 1:  # Only the text prompt exists
        print(f"No images found for case {case_id}, question {question_id}")
        log_entry = {
            "case_id": case_id,
            "question_id": question_id,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "temperature": temperature,
            "status": "skipped",
            "reason": "no_images",
            "cost": 0,
            "input": {
                "messages": messages,
                "question_data": {
                    "question": question_data["question"],
                    "explanation": question_data["explanation"],
                    "metadata": question_data.get("metadata", {}),
                    "figures": question_data["figures"],
                },
                "image_urls": [subfig["url"] for subfig in subfigures if "url" in subfig],
                "image_captions": [subfig.get("caption", "") for subfig in subfigures],
            },
        }
        logging.info(json.dumps(log_entry))
        return None

    try:
        start_time = time.time()

        response = client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=50, temperature=temperature
        )
        duration = time.time() - start_time

        log_entry = {
            "case_id": case_id,
            "question_id": question_id,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "temperature": temperature,
            "duration": round(duration, 2),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "cost": calculate_cost(response.usage.prompt_tokens, response.usage.completion_tokens),
            "model_answer": response.choices[0].message.content,
            "correct_answer": question_data["answer"],
            "input": {
                "messages": messages,
                "question_data": {
                    "question": question_data["question"],
                    "explanation": question_data["explanation"],
                    "metadata": question_data.get("metadata", {}),
                    "figures": question_data["figures"],
                },
                "image_urls": [subfig["url"] for subfig in subfigures if "url" in subfig],
                "image_captions": [subfig.get("caption", "") for subfig in subfigures],
            },
        }
        logging.info(json.dumps(log_entry))
        return response

    except openai.RateLimitError:
        log_entry = {
            "case_id": case_id,
            "question_id": question_id,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "temperature": temperature,
            "status": "error",
            "reason": "rate_limit",
            "cost": 0,
            "input": {
                "messages": messages,
                "question_data": {
                    "question": question_data["question"],
                    "explanation": question_data["explanation"],
                    "metadata": question_data.get("metadata", {}),
                    "figures": question_data["figures"],
                },
                "image_urls": [subfig["url"] for subfig in subfigures if "url" in subfig],
                "image_captions": [subfig.get("caption", "") for subfig in subfigures],
            },
        }
        logging.info(json.dumps(log_entry))
        print(
            f"\nRate limit hit for case {case_id}, question {question_id}. Waiting 20s...",
            flush=True,
        )
        time.sleep(20)
        raise
    except Exception as e:
        log_entry = {
            "case_id": case_id,
            "question_id": question_id,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "temperature": temperature,
            "status": "error",
            "error": str(e),
            "cost": 0,
            "input": {
                "messages": messages,
                "question_data": {
                    "question": question_data["question"],
                    "explanation": question_data["explanation"],
                    "metadata": question_data.get("metadata", {}),
                    "figures": question_data["figures"],
                },
                "image_urls": [subfig["url"] for subfig in subfigures if "url" in subfig],
                "image_captions": [subfig.get("caption", "") for subfig in subfigures],
            },
        }
        logging.info(json.dumps(log_entry))
        print(f"Error processing case {case_id}, question {question_id}: {str(e)}")
        raise


def load_benchmark_questions(case_id: str) -> list:
    """Load benchmark questions for a given case.

    Args:
        case_id: Identifier for the medical case

    Returns:
        list: List of paths to question files
    """
    benchmark_dir = "../benchmark/questions"
    return glob.glob(f"{benchmark_dir}/{case_id}/{case_id}_*.json")


def count_total_questions() -> tuple[int, int]:
    """Count total number of cases and questions in benchmark.

    Returns:
        tuple: (total_cases, total_questions)
    """
    total_cases = len(glob.glob("../benchmark/questions/*"))
    total_questions = sum(
        len(glob.glob(f"../benchmark/questions/{case_id}/*.json"))
        for case_id in os.listdir("../benchmark/questions")
    )
    return total_cases, total_questions


def main() -> None:
    """Main function to run the benchmark evaluation."""
    with open("../data/eurorad_metadata.json", "r") as file:
        data = json.load(file)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    global client
    client = openai.OpenAI(api_key=api_key)

    total_cases, total_questions = count_total_questions()
    cases_processed = 0
    questions_processed = 0
    skipped_questions = 0

    print(f"Beginning benchmark evaluation for model {model_name} with temperature {temperature}")

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

            # Handle cases where response is None
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
