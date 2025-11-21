import os
import json
import logging
import argparse
import asyncio
import csv
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import base64
from openai import AsyncOpenAI
import time
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.tokenize import word_tokenize
import numpy as np

# Download required NLTK data if missing
for resource in ['tokenizers/punkt', 'corpora/wordnet', 'corpora/omw-1.4']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
for noisy_logger in ["httpx", "httpcore", "openai"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def setup_args():
    """Set up command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate GPT-o1 series models on VQA datasets")

    parser.add_argument("--data_root", type=str, default="data/Slake1.0", help="Dataset root directory")
    parser.add_argument("--test_split", type=str, default="validate.json", help="Test split filename")
    parser.add_argument("--output_dir", type=str, default="./outputs/gpt_o1_baseline", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=5, help="Maximum number of samples (for quick testing)")

    parser.add_argument("--dataset_type", type=str, choices=["open_ended", "multiple_choice"],
                        default="open_ended", help="Dataset type: open-ended or multiple-choice")

    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--model", type=str, default="o1-mini", help="Model to use (o1-preview, o1-mini)")
    parser.add_argument("--api_base", type=str, default=None, help="Custom API base URL")

    parser.add_argument("--max_concurrent_runs", type=int, default=3, help="Maximum concurrent requests")
    parser.add_argument("--similarity_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model for semantic similarity")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    parser.add_argument("--std_test", action="store_true", help="Enable standard deviation test mode")
    parser.add_argument("--std_runs", type=int, default=5, help="Number of runs for std test")
    parser.add_argument("--std_samples", type=int, default=10, help="Number of samples per std test run")

    return parser.parse_args()


def load_data(args):
    """Load dataset"""
    data_file = Path(args.data_root) / args.test_split
    logger.info(f"Loading dataset: {data_file}")
    logger.info(f"Dataset type: {args.dataset_type}")

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # Normalize question field
        if "question" in item and "query" not in item:
            item["query"] = item["question"]
        if "answer" not in item and "answer_cn" in item:
            item["answer"] = item["answer_cn"]

        # Normalize image path
        if "img_name" in item:
            if "../data/Slake1.0/imgs/" in item["img_name"]:
                parts = item["img_name"].split("/")
                img_dir = parts[-2]
                img_file = parts[-1]
                image_path = Path(args.data_root) / "imgs" / img_dir / img_file
            else:
                image_path = Path(args.data_root) / "imgs" / item["img_name"]
        elif "image" in item:
            image_path = Path(args.data_root) / item["image"]
        else:
            continue
        item["image_path"] = str(image_path)

    if args.max_samples is not None:
        data = data[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    logger.info(f"Loaded {len(data)} samples")
    return data


def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_answer_from_response(response_text, dataset_type):
    """Extract answer from model response"""
    if dataset_type == "multiple_choice":
        patterns = [
            r"(?:answer is|answer|option|correct answer is)\s*:?\s*([A-D]|\d+)",
            r"(?:^|\s)([A-D])\s*[).]",
            r"(?:^|\s)([A-D])(?:\s|$)",
            r"(?:^|\s)(\d+)(?:\s|$|[).])",
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        match = re.search(r'([A-D]|\d+)', response_text)
        if match:
            return match.group(1).upper()
        return response_text[:100].strip()
    else:
        return response_text.strip()


def calculate_accuracy(ground_truth, predicted_answer):
    """Calculate accuracy for multiple-choice questions"""
    gt = str(ground_truth).strip().upper()
    pred = str(predicted_answer).strip().upper()

    if gt == pred:
        return 1.0

    gt_letter = re.search(r'^([A-D])', gt)
    if gt_letter:
        if pred == gt_letter.group(1):
            return 1.0
        if len(pred) > 5:
            pred_letter = re.search(r'([A-D])', pred)
            if pred_letter and pred_letter.group(1) == gt_letter.group(1):
                return 1.0

    if len(pred) > 5:
        match = re.search(r'([A-D]|\d+)', pred)
        if match:
            extracted = match.group(1).upper()
            if gt == extracted or (len(gt) == 1 and extracted == gt):
                return 1.0
    return 0.0


async def query_gpt_o1_multiple_choice(client, item, model):
    """Query o1 model for multiple-choice questions"""
    query = item['query']
    image_path = item['image_path']

    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        base64_image = encode_image(image_path)

        user_prompt = f"""You are a medical imaging expert. Analyze the provided image and select the correct answer.

Important:
1. Carefully observe all details in the image
2. Read all options
3. Choose the most accurate answer based on image evidence
4. Respond with ONLY the option letter (A, B, C, D, etc.) or number
5. If you cannot analyze the image, say "Cannot analyze"

Format: Answer is [letter/number]

Question: {query}"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]

        start_time = time.time()
        response = await client.chat.completions.create(model=model, messages=messages)
        answer = response.choices[0].message.content
        processing_time = time.time() - start_time

        # Rejection fallback
        rejection_phrases = ["unable to view", "cannot see", "can't analyze", "i cannot", "i can't", "sorry"]
        if any(phrase in answer.lower() for phrase in rejection_phrases):
            logger.warning("Model refused to analyze image, trying simplified prompt...")
            simplified = f"Analyze this medical image and answer the multiple-choice question. {query}"
            simplified_messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": simplified},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
            resp = await client.chat.completions.create(model=model, messages=simplified_messages)
            answer = resp.choices[0].message.content
            processing_time = time.time() - start_time

        return answer, processing_time, None

    except Exception as e:
        logger.error(f"API call failed: {e}", exc_info=True)
        return f"Error: {str(e)}", 0, str(e)


async def query_gpt_o1(client, item, model):
    """Query o1 model for open-ended questions"""
    query = item['query']
    image_path = item['image_path']

    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        base64_image = encode_image(image_path)

        user_prompt = f"""Answer the question based on the image. If you cannot directly analyze it, describe the image first.

Question: {query}"""

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]

        start_time = time.time()
        response = await client.chat.completions.create(model=model, messages=messages)
        answer = response.choices[0].message.content
        processing_time = time.time() - start_time

        # Vision rejection fallback: describe first, then answer
        rejection_phrases = ["unable to view", "cannot see", "can't analyze", "i cannot", "i can't"]
        if any(phrase in answer.lower() for phrase in rejection_phrases):
            logger.info("Image rejected. Trying describe-then-answer strategy...")
            desc_resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "Briefly describe this medical image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}]
            )
            desc = desc_resp.choices[0].message.content
            if not any(phrase in desc.lower() for phrase in rejection_phrases):
                final_resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": f"""Based on this description: "{desc}"

Original question: {query}

Answer the question."""}]
                )
                answer = final_resp.choices[0].message.content
                processing_time = time.time() - start_time

        return answer, processing_time, None

    except Exception as e:
        logger.error(f"API call failed: {e}", exc_info=True)
        return f"Error: {str(e)}", 0, str(e)


# [All metric calculation functions remain unchanged, only comments translated]
def calculate_meteor_score(reference, candidate):
    try:
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        return meteor_score([ref_tokens], cand_tokens)
    except Exception as e:
        logger.warning(f"METEOR error: {e}")
        return 0.0


def calculate_rouge_scores(reference, candidate):
    try:
        rouge = Rouge()
        if not reference.strip() or not candidate.strip():
            return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
        return rouge.get_scores(candidate, reference)[0]
    except Exception as e:
        logger.warning(f"ROUGE error: {e}")
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}


def calculate_bleu_score(reference, candidate):
    try:
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=SmoothingFunction().method1)
    except Exception as e:
        logger.warning(f"BLEU error: {e}")
        return 0.0


def calculate_semantic_similarity(a1, a2, model):
    embeddings = model.encode([a1, a2])
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()


async def calculate_llm_judge_score(client, ground_truth, generated_answer, model):
    try:
        prompt = f"""You are an expert evaluator. Score the generated answer against the ground truth using:

1. Correctness  2. Completeness  3. Relevance  4. Coherence

Score only: 1.0, 0.9, 0.7, 0.5, or 0.0

Ground Truth: {ground_truth}
Generated Answer: {generated_answer}

Respond with ONLY the numeric score."""
        response = await client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        score = response.choices[0].message.content.strip()
        try:
            score_val = float(score)
            return score_val if score_val in [0.0, 0.5, 0.7, 0.9, 1.0] else 0.0
        except:
            return 0.0
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return 0.0


def convert_llm_judge_to_accuracy(score):
    return 1.0 if score > 0.6 else 0.0


async def main():
    args = setup_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.api_key:
        raise ValueError("OpenAI API key required")

    data = load_data(args)
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base)

    if args.std_test:
        await run_std_test(args, data)
    else:
        await run_evaluations(args, data)


if __name__ == "__main__":
    asyncio.run(main())