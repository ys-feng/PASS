import os
import json
import logging
import argparse
import asyncio
import csv
import re
import time
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
from collections import Counter
import base64
from openai import AsyncOpenAI
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Download NLTK resources
for pkg in ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
for lib in ["httpx", "httpcore", "openai"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


def setup_args():
    """Set up command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate GPT-4o with Self-Consistency on medical VQA")

    parser.add_argument("--data_root", type=str, default="data/Slake1.0", help="Dataset root directory")
    parser.add_argument("--test_split", type=str, default="mc_validate_new.json", help="Test split filename")
    parser.add_argument("--output_dir", type=str, default="./outputs/gpt4o_sc_baseline", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")

    parser.add_argument("--dataset_type", type=str, choices=["open_ended", "multiple_choice"],
                        default="multiple_choice", help="Dataset type")

    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents for self-consistency")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06", help="Model name")
    parser.add_argument("--api_base", type=str, default=None, help="Custom API base URL")

    parser.add_argument("--max_concurrent_runs", type=int, default=2, help="Max concurrent sample processing")
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

    logger.info(f"Loaded {len(data)} samples | Agents: {args.num_agents} | Temp: {args.temperature}")
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
    return text[:50].strip()


def calculate_accuracy(gt: str, pred: str) -> float:
    gt = str(gt).strip().upper()
    pred = str(pred).strip().upper()
    if gt == pred:
        return 1.0
    if len(pred) > 5:
        m = re.search(r'([A-D]|\d+)', pred)
        if m and m.group(1).upper() == gt:
            return 1.0
    return 0.0


def majority_vote(answers: list, dataset_type: str = "multiple_choice"):
    if not answers:
        return "ERROR", {}

    if dataset_type == "multiple_choice":
        counts = Counter(answers)
        winner, count = counts.most_common(1)[0]
        strength = count / len(answers)
        return winner, {
            "votes": dict(counts),
            "total_agents": len(answers),
            "consensus_strength": strength,
            "final_answer": winner
        }
    else:
        unique = set(answers)
        if len(unique) == 1:
            return answers[0], {"consensus_strength": 1.0, "final_answer": answers[0]}
        return answers[0], {"all_answers": answers, "consensus_strength": 0.0, "final_answer": answers[0]}


# Diverse system prompts for different reasoning personas
SYSTEM_PROMPTS = [
    "You are an experienced medical imaging expert. Analyze the image carefully and select the correct answer. "
    "Think step-by-step, eliminate wrong options, and output ONLY the final letter (A, B, C, D).",

    "You are a senior radiologist. Examine the image systematically, identify key features, "
    "and choose the most accurate answer. Final answer format: [letter]",

    "As a diagnostic imaging specialist, perform structured analysis: observe → identify pathology → match to options. "
    "Output only the correct option letter.",

    "You are a medical school professor teaching radiology. Use evidence-based reasoning from the image. "
    "Final answer: [A/B/C/D]",

    "You are an AI-assisted diagnostic expert. Maximize accuracy by pattern matching and clinical correlation. "
    "Respond with just the answer letter."
]


async def query_single_agent(client, item, model, agent_id, temperature, dataset_type):
    query = item["query"]
    image_path = item["image_path"]

    try:
        base64_image = encode_image(image_path)
        system_msg = SYSTEM_PROMPTS[agent_id % len(SYSTEM_PROMPTS)]

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]

        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=300,
            temperature=temperature
        )
        raw = resp.choices[0].message.content
        extracted = extract_answer_from_response(raw, dataset_type)
        return extracted, raw

    except Exception as e:
        logger.error(f"Agent {agent_id+1} failed: {e}")
        return "ERROR", f"Error: {e}"


async def query_gpt4o_self_consistency(client, item, model, num_agents, temperature, dataset_type):
    tasks = [
        query_single_agent(client, item, model, i, temperature, dataset_type)
        for i in range(num_agents)
    ]
    results = await asyncio.gather(*tasks)

    extracted_answers = []
    raw_responses = []
    for extracted, raw in results:
        extracted_answers.append(extracted)
        raw_responses.append(raw)

    final_answer, vote_details = majority_vote(extracted_answers, dataset_type)
    return final_answer, vote_details, raw_responses


# Metric functions (same as previous clean versions)
def calculate_semantic_similarity(a, b, encoder):
    try:
        embeddings = encoder.encode([a, b])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    except:
        return 0.0

async def calculate_llm_judge_score(client, gt, gen, model):
    prompt = f"""Score the generated answer against the ground truth. Return ONLY one score: 1.0, 0.9, 0.7, 0.5, or 0.0.

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


async def evaluate_sample(client, item, model, num_agents, temperature, dataset_type, encoder=None):
    query = item["query"]
    gt = item.get("answer", "")

    start = time.time()
    final_answer, vote_details, raw_responses = await query_gpt4o_self_consistency(
        client, item, model, num_agents, temperature, dataset_type
    )
    elapsed = time.time() - start

    consensus = vote_details.get("consensus_strength", 0.0)

    if final_answer == "ERROR":
        return {}, "ERROR", {"query": query, "ground_truth": gt, "status": "error", "processing_time": elapsed}

    if dataset_type == "multiple_choice":
        acc = calculate_accuracy(gt, final_answer)
        logger.info(f"Q: {query[:50]}... | GT: {gt} | SC: {final_answer} | Acc: {acc:.1f} | Consensus: {consensus:.2f}")
        return {"accuracy": acc}, final_answer, {
            "query": query, "ground_truth": gt, "final_answer": final_answer,
            "accuracy": acc, "vote_details": vote_details, "consensus_strength": consensus,
            "status": "success", "processing_time": elapsed
        }
    else:
        # Open-ended metrics
        sim = calculate_semantic_similarity(gt, final_answer, encoder) if encoder else 0.0
        judge = await calculate_llm_judge_score(client, gt, final_answer, model)
        judge_acc = convert_llm_judge_to_accuracy(judge)

        scores = {
            "semantic_similarity": sim,
            "llm_judge_score": judge,
            "llm_judge_accuracy": judge_acc
        }
        return scores, final_answer, {**scores, "query": query, "ground_truth": gt,
                                     "final_answer": final_answer, "vote_details": vote_details,
                                     "consensus_strength": consensus, "status": "success", "processing_time": elapsed}


async def run_evaluations(args, data):
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base)
    encoder = SentenceTransformer("all-MiniLM-L6-v2") if args.dataset_type == "open_ended" else None

    sem = asyncio.Semaphore(args.max_concurrent_runs)
    async def task(item):
        async with sem:
            await asyncio.sleep(1.5)
            return await evaluate_sample(client, item, args.model, args.num_agents,
                                        args.temperature, args.dataset_type, encoder)

    tasks = [task(item) for item in data]
    results = []
    success_count = 0
    total_scores = {"accuracy": 0.0} if args.dataset_type == "multiple_choice" else {
        "semantic_similarity": 0.0, "llm_judge_score": 0.0, "llm_judge_accuracy": 0.0
    }
    consensus_list = []

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Self-Consistency Evaluation"):
        scores_dict, _, result = await future
        results.append(result)
        if result["status"] == "success":
            success_count += 1
            for k, v in scores_dict.items():
                total_scores[k] += v
            consensus_list.append(result.get("consensus_strength", 0))

        if len(results) % 5 == 0 or len(results) == len(data):
            save_results(args, results, len(data), success_count, total_scores)

    save_results(args, results, len(data), success_count, total_scores)

    avg_scores = {k: total_scores[k] / success_count for k in total_scores} if success_count else {k: 0.0 for k in total_scores}
    avg_time = sum(r["processing_time"] for r in results if r["status"] == "success") / success_count if success_count else 0
    avg_consensus = sum(consensus_list) / len(consensus_list) if consensus_list else 0

    logger.info(f"Evaluation complete. Success: {success_count}/{len(data)} | Avg Consensus: {avg_consensus:.3f}")
    for k, v in avg_scores.items():
        logger.info(f"  {k}: {v:.4f}{f' ({v*100:.2f}%)' if 'accuracy' in k else ''}")

    return results, avg_scores, success_count / len(data), avg_time, avg_consensus


def save_results(args, results, total, success, total_scores):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "gpt4o_sc"
    if args.dataset_type == "multiple_choice":
        csv_file = out_dir / f"{prefix}_mc_results.csv"
        summary_file = out_dir / f"{prefix}_mc_summary.json"
        fields = ["query", "ground_truth", "final_answer", "accuracy", "consensus_strength", "vote_details", "status", "processing_time"]
    else:
        csv_file = out_dir / f"{prefix}_open_results.csv"
        summary_file = out_dir / f"{prefix}_open_summary.json"
        fields = ["query", "ground_truth", "final_answer", "semantic_similarity", "llm_judge_score", "llm_judge_accuracy",
                  "consensus_strength", "vote_details", "status", "processing_time"]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fields}
            if "vote_details" in row:
                row["vote_details"] = json.dumps(row["vote_details"], ensure_ascii=False)
            writer.writerow(row)

    success_results = [r for r in results if r["status"] == "success"]
    avg_scores = {k: sum(r.get(k, 0) for r in success_results) / len(success_results)
                  for k in total_scores.keys()} if success_results else {}
    avg_consensus = sum(r.get("consensus_strength", 0) for r in success_results) / len(success_results) if success_results else 0

    stats = {
        "method": "Self-Consistency",
        "model": args.model,
        "num_agents": args.num_agents,
        "temperature": args.temperature,
        "dataset_type": args.dataset_type,
        "total_samples": total,
        "successful_samples": success,
        "average_scores": avg_scores,
        "average_consensus_strength": avg_consensus,
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
        logger.info("Standard deviation test mode not included in this clean version (can be added on request)")
    else:
        await run_evaluations(args, data)


if __name__ == "__main__":
    asyncio.run(main())