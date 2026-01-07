import os
import json
import logging
import argparse
import asyncio
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
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
    MEDRAX_AVAILABLE = True
except ImportError as e:
    logger.error("Required modules not found.")
    raise e


def setup_args():
    """Set up command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-Med (via medrax) on medical VQA")

    parser.add_argument("--data_root", type=str, default="data/Slake1.0", help="Dataset root directory")
    parser.add_argument("--test_split", type=str, default="validate.json", help="Test split filename")
    parser.add_argument("--output_dir", type=str, default="./outputs/llavamd_baseline", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")

    parser.add_argument("--dataset_type", type=str, choices=["open_ended", "multiple_choice"],
                        default="multiple_choice", help="Dataset type")

    parser.add_argument("--model_dir", type=str, default=os.getenv("MODEL_DIR", "model-weights"),
                        help="Directory for LLaVA-Med weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu)")

    parser.add_argument("--max_concurrent_runs", type=int, default=5, help="Max concurrent processing")
    parser.add_argument("--similarity_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer for semantic similarity")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    parser.add_argument("--use_llm_judge", action="store_true", default=True, help="Use LLM as judge")
    parser.add_argument("--judge_model", type=str, default="gpt-3.5-turbo", help="LLM judge model")

    parser.add_argument("--std_test", action="store_true", help="Run standard deviation test")
    parser.add_argument("--std_runs", type=int, default=5, help="Number of std test runs")
    parser.add_argument("--std_samples", type=int, default=10, help="Samples per std test run")

    return parser.parse_args()


def load_slake_dataset(data_root: str, split_file: str) -> List[Dict]:
    """Load and normalize SLAKE dataset"""
    json_path = Path(data_root) / split_file
    if not json_path.exists():
        logger.error(f"Dataset file not found: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = []
    root = Path(data_root)

    for item in raw_data:
        query = item.get("question") or item.get("query")
        answer = item.get("answer_cn") or item.get("answer")
        img_name = item.get("img_name")
        qid = item.get("qid")

        if not (query and img_name):
            continue

        if "../data/Slake1.0/imgs/" in img_name:
            parts = img_name.split("/")
            image_path = root / "imgs" / parts[-2] / parts[-1]
        else:
            image_path = root / "imgs" / img_name

        if image_path.exists():
            processed.append({
                "qid": qid,
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
        r"\(([A-D]|\d+)\)",
        r"(?:answer is|answer|option|correct answer is)\s*:?\s*\(?([A-D]|\d+)\)?",
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


async def calculate_llm_judge_score(client, gt: str, gen: str, model: str) -> float:
    prompt = f"""Score the generated answer against the ground truth. Return ONLY one score: 1.0, 0.9, 0.7, 0.5, or 0.0.

Ground Truth: {gt}
Generated: {gen}"""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        score = resp.choices[0].message.content.strip()
        return float(score) if score in ["0.0", "0.5", "0.7", "0.9", "1.0"] else 0.0
    except:
        return 0.0


def convert_llm_judge_to_accuracy(score: float) -> float:
    return 1.0 if score > 0.6 else 0.0


async def evaluate_sample(tool, item: Dict, encoder: Optional[SentenceTransformer],
                         use_judge: bool, judge_client: Optional[AsyncOpenAI],
                         judge_model: str, dataset_type: str):
    query = item["query"]
    image_path = item["image_path"]
    gt = item.get("answer", "")

    try:
        start = time.time()
        operator = LlaVAMed(tool, name="LlaVAMed")
        result = await operator(image_path=image_path, query=query)
        answer = result.get("response", "")
        elapsed = time.time() - start

        if not answer or "Error" in answer:
            logger.warning(f"Query failed: {query[:50]}...")
            return {}, answer, {"query": query, "ground_truth": gt, "answer": answer, "status": "error"}

        if dataset_type == "multiple_choice":
            pred = extract_answer_from_response(answer, dataset_type)
            acc = calculate_accuracy(gt, pred)
            logger.info(f"Q: {query[:50]}... | GT: {gt} | Pred: {pred} | Acc: {acc:.1f}")
            return {"accuracy": acc}, answer, {
                "query": query, "ground_truth": gt, "answer": answer,
                "extracted_answer": pred, "accuracy": acc,
                "status": "success", "processing_time": elapsed
            }
        else:
            sim = calculate_semantic_similarity(gt, answer, encoder) if encoder else 0.0
            judge = await calculate_llm_judge_score(judge_client, gt, answer, judge_model) if use_judge and judge_client else 0.0
            judge_acc = convert_llm_judge_to_accuracy(judge)

            scores = {
                "semantic_similarity": sim,
                "llm_judge_score": judge,
                "llm_judge_accuracy": judge_acc
            }
            return scores, answer, {**scores, "query": query, "ground_truth": gt,
                                   "answer": answer, "status": "success", "processing_time": elapsed}

    except Exception as e:
        logger.error(f"Sample failed: {e}", exc_info=True)
        return {}, f"Error: {e}", {"query": query, "ground_truth": gt, "status": "error"}


async def run_evaluations(args, data):
    tool = LlavaMedTool(
        cache_dir=args.model_dir,
        device=args.device,
        load_in_8bit=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    encoder = SentenceTransformer(args.similarity_model, device=args.device) if args.dataset_type == "open_ended" else None
    judge_client = AsyncOpenAI() if args.use_llm_judge else None

    sem = asyncio.Semaphore(args.max_concurrent_runs)
    async def task(item):
        async with sem:
            return await evaluate_sample(
                tool, item, encoder, args.use_llm_judge, judge_client, args.judge_model, args.dataset_type
            )

    tasks = [task(item) for item in data]
    results = []
    success_count = 0
    total_scores = {"accuracy": 0.0} if args.dataset_type == "multiple_choice" else {
        "semantic_similarity": 0.0, "llm_judge_score": 0.0, "llm_judge_accuracy": 0.0
    }

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating LLaVA-Med"):
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

    logger.info(f"Evaluation complete. Success: {success_count}/{len(data)}")
    for k, v in avg_scores.items():
        logger.info(f"  {k}: {v:.4f}{f' ({v*100:.2f}%)' if 'accuracy' in k else ''}")

    return results, avg_scores


def save_results(args, results, total, success, total_scores):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "llavamd"
    if args.dataset_type == "multiple_choice":
        csv_file = out_dir / f"{prefix}_mc_results.csv"
        summary_file = out_dir / f"{prefix}_mc_summary.json"
        fields = ["query", "ground_truth", "answer", "extracted_answer", "accuracy", "status", "processing_time"]
    else:
        csv_file = out_dir / f"{prefix}_results.csv"
        summary_file = out_dir / f"{prefix}_summary.json"
        fields = ["query", "ground_truth", "answer", "semantic_similarity", "llm_judge_score", "llm_judge_accuracy",
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
        "model": "LLaVA-Med (medrax)",
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

    data = load_slake_dataset(args.data_root, args.test_split)
    if not data:
        return

    if args.max_samples:
        data = data[:args.max_samples]

    if args.std_test:
        logger.info("Standard deviation test not included in this clean version (can be added on request)")
    else:
        await run_evaluations(args, data)


if __name__ == "__main__":
    asyncio.run(main())