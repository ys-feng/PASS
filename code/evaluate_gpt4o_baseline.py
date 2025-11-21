import os
import json
import logging
import argparse
import asyncio
import csv
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
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

# Download NLTK data if missing
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

# Suppress noisy libraries
for lib in ["httpx", "httpcore", "openai"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


def setup_args():
    """Set up command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate GPT-4o on medical VQA datasets")

    parser.add_argument("--data_root", type=str, default="data/Slake1.0", help="Dataset root directory")
    parser.add_argument("--test_split", type=str, default="validate.json", help="Test split filename")
    parser.add_argument("--output_dir", type=str, default="./outputs/gpt4o_baseline", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (for quick testing)")

    parser.add_argument("--dataset_type", type=str, choices=["open_ended", "multiple_choice"],
                        default="multiple_choice", help="Dataset type")

    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06", help="Model name")
    parser.add_argument("--api_base", type=str, default=None, help="Custom API base URL")

    parser.add_argument("--max_concurrent_runs", type=int, default=3, help="Max concurrent requests")
    parser.add_argument("--similarity_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer for semantic similarity")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def load_data(args):
    """Load and normalize dataset"""
    data_file = Path(args.data_root) / args.test_split
    logger.info(f"Loading dataset: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "question" in item and "query" not in item:
            item["query"] = item["question"]
        if "answer" not in item and "answer_cn" in item:
            item["answer"] = item["answer_cn"]

        # Resolve image path
        if "img_name" in item:
            path = item["img_name"]
            if "../data/Slake1.0/imgs/" in path:
                parts = path.split("/")
                image_path = Path(args.data_root) / "imgs" / parts[-2] / parts[-1]
            else:
                image_path = Path(args.data_root) / "imgs" / path
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


def encode_image(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_answer_from_response(text: str, dataset_type: str) -> str:
    """Extract final answer (multiple-choice only)"""
    if dataset_type != "multiple_choice":
        return text.strip()

    patterns = [
        r"(?:answer is|answer|option|correct answer is)\s*:?\s*([A-D]|\d+)",
        r"(?:^|\s)([A-D])\s*[).]",
        r"(?:^|\s)([A-D])(?:\s|$)",
        r"(?:^|\s)(\d+)(?:\s|$|[).])",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    m = re.search(r'([A-D]|\d+)', text)
    return m.group(1).upper() if m else text[:100].strip()


def calculate_accuracy(gt: str, pred: str) -> float:
    """Exact or relaxed match for multiple-choice"""
    gt = str(gt).strip().upper()
    pred = str(pred).strip().upper()

    if gt == pred:
        return 1.0

    gt_letter = re.search(r'^([A-D])', gt)
    if gt_letter and (pred == gt_letter.group(1) or
                     (len(pred) > 5 and re.search(r'([A-D])', pred) and re.search(r'([A-D])', pred).group(1) == gt_letter.group(1))):
        return 1.0

    if len(pred) > 5:
        m = re.search(r'([A-D]|\d+)', pred)
        if m and (m.group(1).upper() == gt or (len(gt) == 1 and m.group(1).upper() == gt)):
            return 1.0

    return 0.0


async def query_gpt4o_multiple_choice(client, item, model):
    """Query GPT-4o for multiple-choice questions"""
    query = item["query"]
    image_path = item["image_path"]

    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image missing: {image_path}")

        base64_image = encode_image(image_path)

        system_prompt = """You are a medical imaging expert. Analyze the image and select the correct answer.
Answer with ONLY the option letter (A, B, C, D, etc.) or number. Do not include explanations."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]

        start = time.time()
        resp = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=500, temperature=0.1
        )
        answer = resp.choices[0].message.content
        elapsed = time.time() - start

        return answer, elapsed, None

    except Exception as e:
        logger.error(f"Multiple-choice query failed: {e}", exc_info=True)
        return f"Error: {e}", 0, str(e)


async def query_gpt4o(client, item, model):
    """Query GPT-4o for open-ended questions with fallback strategy"""
    query = item["query"]
    image_path = item["image_path"]

    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image missing: {image_path}")

        base64_image = encode_image(image_path)

        messages = [
            {"role": "system", "content": "Answer the question based on the medical image."},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]

        start = time.time()
        resp = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=1000, temperature=0.1
        )
        answer = resp.choices[0].message.content
        elapsed = time.time() - start

        # Vision rejection fallback (describe → answer → concise)
        rejection = any(p in answer.lower() for p in ["unable to view", "cannot see", "can't analyze", "i cannot"])
        if rejection:
            logger.info("Vision rejected. Using describe-then-answer-then-concise strategy...")
            desc_resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "Briefly describe this medical image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}],
                max_tokens=200
            )
            desc = desc_resp.choices[0].message.content

            if "cannot" not in desc.lower():
                detailed = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": f"Description: {desc}\nQuestion: {query}\nAnswer concisely."}],
                    max_tokens=500
                )
                answer = detailed.choices[0].message.content
                elapsed = time.time() - start

        return answer, elapsed, None

    except Exception as e:
        logger.error(f"Open-ended query failed: {e}", exc_info=True)
        return f"Error: {e}", 0, str(e)


# Metric functions (unchanged, only comments translated)
def calculate_meteor_score(ref, cand): ...
def calculate_rouge_scores(ref, cand): ...
def calculate_bleu_score(ref, cand): ...
def calculate_semantic_similarity(a, b, model): ...
async def calculate_llm_judge_score(client, gt, gen, model): ...
def convert_llm_judge_to_accuracy(score): return 1.0 if score > 0.6 else 0.0
def calculate_std_deviation(vals): return np.std(vals, ddof=1) if len(vals) >= 2 else 0.0
def calculate_confidence_interval(vals, conf=0.95): ...


async def evaluate_gpt4o_on_sample(client, item, encoder, model, dataset_type="open_ended"):
    """Evaluate one sample"""
    query = item["query"]
    gt = item.get("answer", "")

    try:
        if dataset_type == "multiple_choice":
            answer, proc_time, err = await query_gpt4o_multiple_choice(client, item, model)
        else:
            answer, proc_time, err = await query_gpt4o(client, item, model)

        if err:
            logger.warning(f"Query error: {err}")
            return {}, answer, {"query": query, "ground_truth": gt, "answer": answer, "status": "error", "processing_time": 0}

        if dataset_type == "multiple_choice":
            extracted = extract_answer_from_response(answer, dataset_type)
            acc = calculate_accuracy(gt, extracted)
            logger.info(f"Q: {query[:50]}... | GT: {gt} | Pred: {extracted} | Acc: {acc:.1f}")
            return {"accuracy": acc}, answer, {
                "query": query, "ground_truth": gt, "answer": answer,
                "extracted_answer": extracted, "accuracy": acc,
                "status": "success", "processing_time": proc_time
            }
        else:
            # Open-ended metrics
            sim = calculate_semantic_similarity(gt, answer, encoder)
            meteor = calculate_meteor_score(gt, answer)
            rouge = calculate_rouge_scores(gt, answer)
            bleu = calculate_bleu_score(gt, answer)
            judge = await calculate_llm_judge_score(client, gt, answer, model)
            judge_acc = convert_llm_judge_to_accuracy(judge)

            scores = {
                "semantic_similarity": sim, "meteor_score": meteor,
                "rouge_1_f": rouge["rouge-1"]["f"], "rouge_2_f": rouge["rouge-2"]["f"], "rouge_l_f": rouge["rouge-l"]["f"],
                "bleu_score": bleu, "llm_judge_score": judge, "llm_judge_accuracy": judge_acc
            }
            return scores, answer, {**scores, "query": query, "ground_truth": gt, "answer": answer,
                                   "status": "success", "processing_time": proc_time}

    except Exception as e:
        logger.error(f"Sample evaluation failed: {e}", exc_info=True)
        return {}, f"Error: {e}", {"query": query, "ground_truth": gt, "answer": f"Error: {e}", "status": "error"}


async def run_evaluations(args, data):
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base)
    encoder = SentenceTransformer(args.similarity_model) if args.dataset_type == "open_ended" else None

    sem = asyncio.Semaphore(args.max_concurrent_runs)
    async def wrapped(item):
        async with sem:
            await asyncio.sleep(0.5)  # gentle rate limiting
            return await evaluate_gpt4o_on_sample(client, item, encoder, args.model, args.dataset_type)

    tasks = [wrapped(item) for item in data]
    results = []
    success_count = 0
    total_scores = {"accuracy": 0.0} if args.dataset_type == "multiple_choice" else {
        "semantic_similarity": 0.0, "meteor_score": 0.0, "rouge_1_f": 0.0, "rouge_2_f": 0.0,
        "rouge_l_f": 0.0, "bleu_score": 0.0, "llm_judge_score": 0.0, "llm_judge_accuracy": 0.0
    }

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        scores_dict, _, result = await future
        results.append(result)
        if result["status"] == "success":
            success_count += 1
            for k, v in scores_dict.items():
                total_scores[k] += v

        if len(results) % 10 == 0 or len(results) == len(data):
            save_results(args, results, len(data), success_count, total_scores)

    save_results(args, results, len(data), success_count, total_scores)
    avg_scores = {k: total_scores[k] / success_count for k in total_scores} if success_count else {k: 0.0 for k in total_scores}
    avg_time = sum(r["processing_time"] for r in results if r["status"] == "success") / success_count if success_count else 0

    logger.info(f"Evaluation complete. Success: {success_count}/{len(data)}")
    for k, v in avg_scores.items():
        logger.info(f"  {k}: {v:.4f}{' (' + f'{v*100:.2f}%' + ')' if 'accuracy' in k else ''}")

    return results, avg_scores, success_count / len(data), avg_time



def save_results(args, results, total, success, total_scores):
    """Save CSV + JSON summary"""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_type == "multiple_choice":
        csv_file = out_dir / "gpt4o_mc_baseline_results.csv"
        summary_file = out_dir / "gpt4o_mc_baseline_summary.json"
        fields = ["query", "ground_truth", "answer", "extracted_answer", "accuracy", "status", "processing_time"]
    else:
        csv_file = out_dir / "gpt4o_baseline_results.csv"
        summary_file = out_dir / "gpt4o_baseline_summary.json"
        fields = ["query", "ground_truth", "answer", "semantic_similarity", "meteor_score", "rouge_1_f",
                  "rouge_2_f", "rouge_l_f", "bleu_score", "llm_judge_score", "llm_judge_accuracy",
                  "status", "processing_time"]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})

    success_results = [r for r in results if r["status"] == "success"]
    avg_scores = {k: sum(r.get(k, 0) for r in success_results) / len(success_results)
                  for k in total_scores.keys()} if success_results else {}
    stats = {
        "model": args.model,
        "dataset_type": args.dataset_type,
        "total_samples": total,
        "successful_samples": success,
        "average_scores": avg_scores,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {out_dir}")


async def main():
    args = setup_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if not args.api_key:
        raise ValueError("OpenAI API key required")

    data = load_data(args)


if __name__ == "__main__":
    asyncio.run(main())