import re
import json
import os
import glob
import time
import logging
from datetime import datetime
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Configure model settings
MODEL_NAME = "StanfordAIMI/CheXagent-2-3b"
DTYPE = torch.bfloat16
DEVICE = "cuda"

# Configure logging
log_filename = f"model_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(message)s")


def initialize_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Initialize the CheXagent model and tokenizer.

    Returns:
        tuple containing:
            - AutoModelForCausalLM: The initialized CheXagent model
            - AutoTokenizer: The initialized tokenizer
    """
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", trust_remote_code=True
    )
    model = model.to(DTYPE)
    model.eval()
    return model, tokenizer


def create_inference_request(
    question_data: dict,
    case_details: dict,
    case_id: str,
    question_id: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> str | None:
    """Create and execute an inference request for the CheXagent model.

    Args:
        question_data: Dictionary containing question details and metadata
        case_details: Dictionary containing case information and image paths
        case_id: Unique identifier for the medical case
        question_id: Unique identifier for the question
        model: The initialized CheXagent model
        tokenizer: The initialized tokenizer

    Returns:
        str | None: Single letter answer (A-F) if successful, None if failed
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
                    image_paths.append("medrax/data/" + subfig["local_path"])

    if not image_paths:
        print(f"No local images found for case {case_id}, question {question_id}")
        return None

    try:
        start_time = time.time()

        # Prepare input for the model
        query = tokenizer.from_list_format(
            [*[{"image": path} for path in image_paths], {"text": prompt}]
        )
        conv = [{"from": "system", "value": system_prompt}, {"from": "human", "value": query}]
        input_ids = tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        )

        # Generate response
        with torch.no_grad():
            output = model.generate(
                input_ids.to(DEVICE),
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=512,
            )[0]

        response = tokenizer.decode(output[input_ids.size(1) : -1])
        duration = time.time() - start_time

        # Clean response
        clean_answer = validate_answer(response)

        # Log response
        log_entry = {
            "case_id": case_id,
            "question_id": question_id,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "duration": round(duration, 2),
            "model_answer": clean_answer,
            "correct_answer": question_data["answer"],
            "input": {
                "question_data": {
                    "question": question_data["question"],
                    "explanation": question_data["explanation"],
                    "metadata": question_data.get("metadata", {}),
                    "figures": question_data["figures"],
                },
                "image_paths": image_paths,
            },
        }
        logging.info(json.dumps(log_entry))
        return clean_answer

    except Exception as e:
        print(f"Error processing case {case_id}, question {question_id}: {str(e)}")
        log_entry = {
            "case_id": case_id,
            "question_id": question_id,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "status": "error",
            "error": str(e),
            "input": {
                "question_data": {
                    "question": question_data["question"],
                    "explanation": question_data["explanation"],
                    "metadata": question_data.get("metadata", {}),
                    "figures": question_data["figures"],
                },
                "image_paths": image_paths,
            },
        }
        logging.info(json.dumps(log_entry))
        return None


def validate_answer(response_text: str) -> str | None:
    """Enforce strict single-letter response format.

    Args:
        response_text: Raw response text from the model

    Returns:
        str | None: Single uppercase letter (A-F) if valid, None if invalid
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


def load_benchmark_questions(case_id: str) -> list[str]:
    """Find all question files for a given case ID.

    Args:
        case_id: Unique identifier for the medical case

    Returns:
        list[str]: List of paths to question JSON files
    """
    benchmark_dir = "../benchmark/questions"
    return glob.glob(f"{benchmark_dir}/{case_id}/{case_id}_*.json")


def count_total_questions() -> tuple[int, int]:
    """Count total number of cases and questions in benchmark.

    Returns:
        tuple containing:
            - int: Total number of cases
            - int: Total number of questions
    """
    total_cases = len(glob.glob("../benchmark/questions/*"))
    total_questions = sum(
        len(glob.glob(f"../benchmark/questions/{case_id}/*.json"))
        for case_id in os.listdir("../benchmark/questions")
    )
    return total_cases, total_questions


def main():
    # Load the cases with local paths
    with open("medrax/data/updated_cases.json", "r") as file:
        data = json.load(file)

    # Initialize model and tokenizer
    model, tokenizer = initialize_model()

    total_cases, total_questions = count_total_questions()
    cases_processed = 0
    questions_processed = 0
    skipped_questions = 0

    print(f"\nBeginning inference with {MODEL_NAME}")
    print(f"Found {total_cases} cases with {total_questions} total questions")

    # Process each case with progress bar
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
            answer = create_inference_request(
                question_data, case_details, case_id, question_id, model, tokenizer
            )

            if answer is None:
                skipped_questions += 1
                continue

            print(f"\nCase {case_id}, Question {question_id}")
            print(f"Model Answer: {answer}")
            print(f"Correct Answer: {question_data['answer']}")

    print(f"\nInference Summary:")
    print(f"Total Cases Processed: {cases_processed}")
    print(f"Total Questions Processed: {questions_processed}")
    print(f"Total Questions Skipped: {skipped_questions}")


if __name__ == "__main__":
    main()
