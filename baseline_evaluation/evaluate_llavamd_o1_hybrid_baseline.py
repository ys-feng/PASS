import os
import json
import logging
import argparse
import asyncio
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.tokenize import word_tokenize
from openai import AsyncOpenAI
import numpy as np

# NLTK resources
for pkg in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
for lib in ["httpx", "httpcore", "openai"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# Import local tools
try:
    from tools import LlavaMedTool
    from core.agent_operators import LlaVAMed
except ImportError as e:
    logger.error("Required modules not found.")
    raise e


def setup_args():
    """Set up command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-Med + GPT-o1 Hybrid on medical VQA")

    parser.add_argument("--data_root", type=str, default="data/Slake1.0", help="Dataset root directory")
    parser.add_argument("--test_split", type=str, default="validate.json", help="Test split filename")
    parser.add_argument("--output_dir", type=str, default="./outputs/llavamd_o1_hybrid", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")

    parser.add_argument("--dataset_type", type=str, choices=["open_ended", "multiple_choice"],
                        default="multiple_choice", help="Dataset type")

    parser.add_argument("--model_dir", type=str, default=os.getenv("MODEL_DIR", "model-weights"),
                        help="LLaVA-Med weights directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--o1_model", type=str, default="o1-preview", help="GPT-o1 model (o1-preview or o1-mini)")
    parser.add_argument("--api_base", type=str, default=None, help="Custom OpenAI API base")

    parser.add_argument("--max_concurrent_runs", type=int, default=3, help="Max concurrent processing")
    parser.add_argument("--similarity_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer for semantic similarity")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def load_slake_dataset(data_root: str, split_file: str) -> List[Dict]:
    """Load and normalize SLAKE dataset"""
    json_path = Path(data_root) / split_file
    if not json_path.exists():
        logger.error(f"Dataset not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = []
    root = Path(data_root)

    for item in raw_data:
        query = item.get("question") or item.get("query")
        answer = item.get("answer_cn") or item.get("answer")
        img_name = item.get("img_name")

        if not (query and img_name):
            continue

        if "../data/Slake1.0/imgs/" in img_name:
            parts = img_name.split("/")
            image_path = root / "imgs" / parts[-2] / parts[-1]
        else:
            image_path = root / "imgs" / img_name

        if image_path.exists():
            processed.append({
                "query": query,
                "image_path": str(image_path),
                "answer": answer
            })
        else:
            logger.warning(f"Image not found: {image_path}")

    logger.info(f"Loaded {len(processed)} valid samples from {json_path}")
    return processed


def extract_answer_from_response(text: str, dataset_type: str) -> str:
    if dataset_type != "multiple_choice" or not text:
        return text.strip()

    patterns = [
        r"(?:answer is|answer|option|correct answer is)\s*:?\s*([A-D]|\d+)",
        r"^([A-D])\b",
        r"([A-D])(?:\s|$)",
        r"^(\d+)\b",
        r"(\d+)(?:\s|$)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).upper() if m.group(1).isalpha() else m.group(1)

    return text[:100].strip()


def calculate_accuracy(gt: str, pred: str) -> float:
    if not pred:
        return 0.0
    gt = str(gt).strip().upper()
    pred = str(pred).strip().upper()
    if gt == pred:
        return 1.0
    if len(pred) > 5:
        m = re.search(r'([A-D]|\d+)', pred)
        if m and m.group(1).upper() == gt:
            return 1.0
    return 0.0


async def get_image_description_from_llavamd(tool, image_path: str, query: str) -> str:
    """Get detailed image description using LLaVA-Med"""
    try:
        operator = LlaVAMed(tool, name="LlaVAMed")
        prompt = f"Describe this medical image in detail, focusing on findings relevant to: {query}"
        result = await operator(image_path=image_path, query=prompt)
        description = result.get("response", "").strip()

        if not description or "error" in description.lower():
            logger.warning("Primary description failed, falling back to simple prompt")
            result = await operator(image_path=image_path, query="Describe this medical image briefly.")
            description = result.get("response", "Unable to generate description")

        return description
    except Exception as e:
        logger.error(f"Image description failed: {e}", exc_info=True)
        return "Error generating image description"


async def query_o1_with_description(client: AsyncOpenAI, description: str, query: str,
                                   model: str, dataset_type: str):
    """Query GPT-o1 using text description only"""
    try:
        if dataset_type == "multiple_choice":
            prompt = f"""Based on this medical image description, answer the multiple-choice question.

Description: {description}

Question: {query}

Respond with ONLY the correct option letter (A, B, C, D, etc.) or number."""
        else:
            prompt = f"""Based on this medical image description, answer the question concisely.

Description: {description}

Question: {query}

Answer in a few words."""

        start = time.time()
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = resp.choices[0].message.content.strip()
        elapsed = time.time() - start
        return answer, elapsed, None

    except Exception as e:
        logger.error(f"GPT-o1 query failed: {e}", exc_info=True)
        return f"Error: {e}", 0, str(e)


async def evaluate_sample(tool, client: AsyncOpenAI, item: Dict, encoder: Optional[SentenceTransformer],
                         o1_model: str, dataset_type: str):
    query = item["query"]
    image_path = item["image_path"]
    gt = item.get("answer", "")

    try:
        start = time.time()
        description = await get_image_description_from_llavamd(tool, image_path, query)
        if "Error" in description:
            return {}, "Error", {"query": query, "ground_truth": gt, "status": "error"}
        answer, o1_time, err = await query_o1_with_description(client, description, query, o1_model, dataset_type)
        total_time = time.time() - start

        if err:
            return {}, answer, {"query": query, "ground_truth": gt, "status": "error"}

        if dataset_type == "multiple_choice":
            pred = extract_answer_from_response(answer, dataset_type)
            acc = calculate_accuracy(gt, pred)
            logger.info(f"Q: {query[:50]}... | GT: {gt} | Pred: {pred} | Acc: {acc:.1f}")
            return {"accuracy": acc}, answer, {
                "query": query, "ground_truth": gt, "image_description": description,
                "answer": answer, "extracted_answer": pred, "accuracy": acc,
                "status": "success", "processing_time": total_time
            }
        else:
            sim = calculate_semantic_similarity(gt, answer, encoder) if encoder else 0.0
            judge = await calculate_llm_judge_score(client, gt, answer, o1_model)
            judge_acc = 1.0 if judge > 0.6 else 0.0

            scores = {
                "semantic_similarity": sim,
                "llm_judge_score": judge,
                "llm_judge_accuracy": judge_acc
            }
            return scores, answer, {**scores, "query": query, "ground_truth": gt,
                                   "image_description": description, "answer": answer,
                                   "status": "success", "processing_time": total_time}

    except Exception as e:
        logger.error(f"Sample evaluation failed: {e}", exc_info=True)
        return {}, f"Error: {e}", {"query": query, "ground_truth": gt, "status": "error"}


async def run_evaluations(args, data):
    tool = LlavaMedTool(
        cache_dir=args.model_dir,
        device=args.device,
        load_in_8bit=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base)
    encoder = SentenceTransformer(args.similarity_model, device=args.device) if args.dataset_type == "open_ended" else None

    sem = asyncio.Semaphore(args.max_concurrent_runs)
    async def task(item):
        async with sem:
            return await evaluate_sample(tool, client, item, encoder, args.o1_model, args.dataset_type)

    tasks = [task(item) for item in data]
    results = []
    success_count = 0
    total_scores = {"accuracy": 0.0} if args.dataset_type == "multiple_choice" else {
        "semantic_similarity": 0.0, "llm_judge_score": 0.0, "llm_judge_accuracy": 0.0
    }

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Hybrid Evaluation"):
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

    logger.info(f"Hybrid evaluation complete. Success: {success_count}/{len(data)}")
    for k, v in avg_scores.items():
        logger.info(f"  {k}: {v:.4f}{f' ({v*100:.2f}%)' if 'accuracy' in k else ''}")

    return results, avg_scores


def save_results(args, results, total, success, total_scores):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "hybrid"
    if args.dataset_type == "multiple_choice":
        csv_file = out_dir / f"{prefix}_mc_results.csv"
        summary_file = out_dir / f"{prefix}_mc_summary.json"
        fields = ["query", "ground_truth", "image_description", "answer", "extracted_answer", "accuracy", "status", "processing_time"]
    else:
        csv_file = out_dir / f"{prefix}_results.csv"
        summary_file = out_dir / f"{prefix}_summary.json"
        fields = ["query", "ground_truth", "image_description", "answer", "semantic_similarity",
                  "llm_judge_score", "llm_judge_accuracy", "status", "processing_time"]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fields})

    success_results = [r for r in results if r["status"] == "success"]
    avg_scores = {k: sum(r.get(k, 0) for r in success_results) / len(success_results)
                  for k in total_scores.keys()} if success_results else {}

    stats = {
        "method": "LLaVA-Med + GPT-o1 Hybrid",
        "o1_model": args.o1_model,
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

    data = load_slake_dataset(args.data_root, args.test_split)
    if not data:
        return

    if args.max_samples:
        data = data[:args.max_samples]

    await run_evaluations(args, data)


if __name__ == "__main__":
    asyncio.run(main())