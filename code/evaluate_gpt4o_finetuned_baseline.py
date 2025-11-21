import os
import json
import logging
import argparse
import asyncio
import csv
import re
from pathlib import Path
from typing import Dict, Tuple
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

# Download NLTK resources if missing
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
for lib in ["httpx", "httpcore", "openai"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


def setup_args():
    """Set up command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned GPT-4o on medical VQA")

    parser.add_argument("--data_root", type=str, default="data/Slake1.0", help="Dataset root directory")
    parser.add_argument("--test_split", type=str, default="validate.json", help="Test split filename")
    parser.add_argument("--output_dir", type=str, default="./outputs/gpt4o_finetuned", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=99, help="Limit number of samples (for testing)")

    parser.add_argument("--dataset_type", type=str, choices=["open_ended", "multiple_choice"],
                        default="open_ended", help="Dataset type")

    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--model", type=str, default="ft:gpt-4o-2024-08-06:personal:pass:COxN5GNH",
                        help="Fine-tuned model ID")
    parser.add_argument("--api_base", type=str, default=None, help="Custom API base URL")

    parser.add_argument("--max_concurrent_runs", type=int, default=3, help="Max concurrent requests")
    parser.add_argument("--similarity_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer for semantic similarity")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    parser.add_argument("--std_test", action="store_true", help="Run standard deviation test")
    parser.add_argument("--std_runs", type=int, default=5, help="Number of std test runs")
    parser.add_argument("--std_samples", type=int, default=10, help="Samples per std test run")

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
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_answer_from_response(text: str, dataset_type: str) -> str:
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


async def query_finetuned_gpt4o(client, item, model, dataset_type):
    """Query fine-tuned GPT-4o (no fallbacks needed)"""
    query = item["query"]
    image_path = item["image_path"]

    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image missing: {image_path}")

        base64_image = encode_image(image_path)

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]

        start = time.time()
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500 if dataset_type == "multiple_choice" else 1000,
            temperature=0.0  # Fine-tuned models are deterministic
        )
        answer = resp.choices[0].message.content.strip()
        elapsed = time.time() - start

        return answer, elapsed, None

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return f"Error: {e}", 0, str(e)


# Metric functions (same as before)
def calculate_meteor_score(ref, cand):
    try:
        return meteor_score([word_tokenize(ref.lower())], word_tokenize(cand.lower()))
    except:
        return 0.0

def calculate_rouge_scores(ref, cand):
    try:
        rouge = Rouge()
        return rouge.get_scores(cand, ref)[0] if ref.strip() and cand.strip() else {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    except:
        return {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}

def calculate_bleu_score(ref, cand):
    try:
        return sentence_bleu([word_tokenize(ref.lower())], word_tokenize(cand.lower()), smoothing_function=SmoothingFunction().method1)
    except:
        return 0.0

def calculate_semantic_similarity(a, b, model):
    embeddings = model.encode([a, b])
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

async def calculate_llm_judge_score(client, gt, gen, model):
    prompt = f"""Score the generated answer against the ground truth. Return ONLY a score: 1.0, 0.9, 0.7, 0.5, or 0.0.

Ground Truth: {gt}
Generated: {gen}"""
    try:
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_tokens=10, temperature=0.0
        )
        score = resp.choices[0].message.content.strip()
        return float(score) if score in ["0.0", "0.5", "0.7", "0.9", "1.0"] else 0.0
    except:
        return 0.0

def convert_llm_judge_to_accuracy(score):
    return 1.0 if score > 0.6 else 0.0


async def evaluate_sample(client, item, encoder, model, dataset_type):
    query = item["query"]
    gt = item.get("answer", "")

    answer, proc_time, err = await query_finetuned_gpt4o(client, item, model, dataset_type)
    if err:
        logger.warning(f"Query error: {err}")
        base = {"query": query, "ground_truth": gt, "answer": answer, "status": "error", "processing_time": 0}
        return {}, answer, base

    if dataset_type == "multiple_choice":
        pred = extract_answer_from_response(answer, dataset_type)
        acc = calculate_accuracy(gt, pred)
        logger.info(f"Q: {query[:50]}... | GT: {gt} | Pred: {pred} | Acc: {acc:.1f}")
        return {"accuracy": acc}, answer, {
            "query": query, "ground_truth": gt, "answer": answer, "extracted_answer": pred,
            "accuracy": acc, "status": "success", "processing_time": proc_time
        }
    else:
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


async def run_evaluations(args, data):
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base)
    logger.info(f"Using fine-tuned model: {args.model}")

    encoder = SentenceTransformer(args.similarity_model) if args.dataset_type == "open_ended" else None

    sem = asyncio.Semaphore(args.max_concurrent_runs)
    async def task(item):
        async with sem:
            await asyncio.sleep(0.5)
            return await evaluate_sample(client, item, encoder, args.model, args.dataset_type)

    tasks = [task(item) for item in data]
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
        logger.info(f"  {k}: {v:.4f}{f' ({v*100:.2f}%)' if 'accuracy' in k else ''}")

    return results, avg_scores, success_count / len(data), avg_time


def save_results(args, results, total, success, total_scores):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "gpt4o_finetuned"
    if args.dataset_type == "multiple_choice":
        csv_file = out_dir / f"{prefix}_mc_results.csv"
        summary_file = out_dir / f"{prefix}_mc_summary.json"
        fields = ["query", "ground_truth", "answer", "extracted_answer", "accuracy", "status", "processing_time"]
    else:
        csv_file = out_dir / f"{prefix}_results.csv"
        summary_file = out_dir / f"{prefix}_summary.json"
        fields = ["query", "ground_truth", "answer", "semantic_similarity", "meteor_score",
                  "rouge_1_f", "rouge_2_f", "rouge_l_f", "bleu_score", "llm_judge_score", "llm_judge_accuracy",
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

    if args.std_test:
        # run_std_test can be added similarly if needed
        pass
    else:
        await run_evaluations(args, data)


if __name__ == "__main__":
    asyncio.run(main())