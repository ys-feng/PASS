import os
import json
import logging
import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, Tuple, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
from PIL import Image
import nltk

# NLTK resources
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
for lib in ["httpx", "httpcore"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# Optional imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False


def setup_args():
    """Set up command-line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLaVA-Med model")

    parser.add_argument("--data_root", type=str, default="data/Slake1.0", help="Dataset root directory")
    parser.add_argument("--test_split", type=str, default="validate.json", help="Test split filename")
    parser.add_argument("--output_dir", type=str, default="./outputs/llava_med_finetuned", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")

    parser.add_argument("--dataset_type", type=str, choices=["open_ended", "multiple_choice"],
                        default="open_ended", help="Dataset type")

    parser.add_argument("--model_name", type=str, default="Veda0718/llava-med-v1.5-mistral-7b-finetuned",
                        help="Hugging Face model path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device")
    parser.add_argument("--conv_mode", type=str, default="mistral_instruct", help="Conversation template")

    parser.add_argument("--similarity_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer for semantic similarity")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--print_answers", action="store_true", help="Print each answer")

    # LLM Judge (optional)
    parser.add_argument("--llm_judge_enable", action="store_true", help="Enable LLM Judge scoring")
    parser.add_argument("--llm_judge_model", type=str, default="ft:gpt-4o-2024-08-06:personal:pass:COxN5GNH")
    parser.add_argument("--llm_judge_api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--llm_judge_api_base", type=str, default=None)

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


def extract_answer_from_response(text: str, dataset_type: str) -> str:
    if dataset_type != "multiple_choice" or not text:
        return text.strip()

    patterns = [
        r"(?:answer is|answer|option|correct answer is)\s*:?\s*([A-D]|\d+)",
        r"^([A-D])(?:[.)]|$)",
        r"([A-D])(?:\s|$|\.|,)",
        r"^(\d+)(?:[.)]|$)",
        r"(\d+)(?:\s|$|\.|,)",
        r"\(([A-D]|\d+)\)",
        r"\[([A-D]|\d+)\]",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            ans = m.group(1)
            return ans.upper() if ans.isalpha() else ans

    # Fallback: first letter or number
    m = re.search(r'\b([A-D])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'\b(\d+)\b', text)
    if m and len(m.group(1)) <= 2:
        return m.group(1)

    return text[:100].strip()


def calculate_accuracy(gt: str, pred: str) -> float:
    if not pred:
        return 0.0
    gt = str(gt).strip()
    pred = str(pred).strip()

    if gt == pred:
        return 1.0

    # Letter match
    gt_letter = re.search(r'^([A-D])', gt, re.IGNORECASE)
    if gt_letter:
        gl = gt_letter.group(1).upper()
        if pred.upper() == gl or re.search(r'\b([A-D])\b', pred.upper()) == gl:
            return 1.0

    # Number match
    gt_num = re.search(r'^(\d+)', gt)
    if gt_num:
        gn = gt_num.group(1)
        if pred == gn or re.search(r'\b(\d+)\b', pred) == gn:
            return 1.0

    return 0.0


def query_llava_model(model, tokenizer, image_processor, item, args, dataset_type):
    """Query LLaVA-Med model"""
    query = item["query"]
    image_path = item["image_path"]

    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if dataset_type == "multiple_choice":
            prompt = f"You are a medical imaging expert. Answer with only the letter (A, B, C, D).\nQuestion: {query}"
        else:
            prompt = f"You are a medical imaging expert. Answer concisely in a few words.\nQuestion: {query}"

        conv = conv_templates[args.conv_mode].copy()
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16 if model.device.type == "cuda" else torch.float32)

        input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
            )

        generated = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        elapsed = time.time() - start

        return generated or "", elapsed, None

    except Exception as e:
        logger.error(f"LLaVA query failed: {e}", exc_info=True)
        return f"Error: {e}", 0, str(e)


async def calculate_llm_judge_score(client, gt: str, gen: str, model_name: str) -> float:
    prompt = f"""Score the generated answer against the ground truth. Return ONLY one score: 1.0, 0.9, 0.7, 0.5, or 0.0.

Ground Truth: {gt}
Generated: {gen}"""
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        score = resp.choices[0].message.content.strip()
        return float(score) if score in ["0.0", "0.5", "0.7", "0.9", "1.0"] else 0.0
    except:
        return 0.0


def evaluate_sample(model, tokenizer, image_processor, item, similarity_encoder, args, dataset_type, judge_client=None):
    query = item["query"]
    gt = item.get("answer", "")

    answer, proc_time, err = query_llava_model(model, tokenizer, image_processor, item, args, dataset_type)
    if err or not answer.strip():
        status = "error" if err else "skipped"
        base = {"query": query, "ground_truth": gt, "answer": answer, "status": status, "processing_time": proc_time or 0}
        return {}, answer, base

    if dataset_type == "multiple_choice":
        extracted = extract_answer_from_response(answer, dataset_type)
        acc = calculate_accuracy(gt, extracted)
        logger.info(f"Q: {query[:50]}... | GT: {gt} | Pred: {extracted} | Acc: {acc:.1f}")
        return {"accuracy": acc}, answer, {
            "query": query, "ground_truth": gt, "answer": answer, "extracted_answer": extracted,
            "accuracy": acc, "status": "success", "processing_time": proc_time
        }
    else:
        sim = calculate_semantic_similarity(gt, answer, similarity_encoder)
        judge = 0.0
        if args.llm_judge_enable and judge_client:
            judge = await calculate_llm_judge_score(judge_client, gt, answer, args.llm_judge_model)

        scores = {"semantic_similarity": sim, "llm_judge_score": judge, "llm_judge_accuracy": 1.0 if judge > 0.6 else 0.0}
        return scores, answer, {**scores, "query": query, "ground_truth": gt, "answer": answer,
                               "status": "success", "processing_time": proc_time}


def run_evaluations(args, data):
    if not LLAVA_AVAILABLE:
        raise ImportError("LLaVA library required. Install from https://github.com/haotian-liu/LLaVA")

    logger.info(f"Loading model: {args.model_name} on {args.device}")
    model_name = get_model_name_from_path(args.model_name)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_name,
        model_base=None,
        model_name=model_name,
        device=args.device,
        device_map="auto"
    )
    model.eval()

    similarity_encoder = SentenceTransformer(args.similarity_model) if args.dataset_type == "open_ended" else None
    judge_client = OpenAI(api_key=args.llm_judge_api_key, base_url=args.llm_judge_api_base) if args.llm_judge_enable and OPENAI_AVAILABLE else None

    results = []
    success_count = 0
    total_scores = {"accuracy": 0.0} if args.dataset_type == "multiple_choice" else {
        "semantic_similarity": 0.0, "llm_judge_score": 0.0, "llm_judge_accuracy": 0.0
    }

    for i, item in enumerate(tqdm(data, desc="Evaluating")):
        scores_dict, answer, result = evaluate_sample(
            model, tokenizer, image_processor, item, similarity_encoder, args, args.dataset_type, judge_client
        )
        results.append(result)

        if args.print_answers:
            print(f"[{i+1}] Q: {item['query'][:100]}")
            print(f"    A: {answer}")
            if args.dataset_type == "multiple_choice":
                print(f"    GT: {result['ground_truth']} | Pred: {result.get('extracted_answer')} | Acc: {result.get('accuracy', 0):.1f}")
            print("-" * 60)

        if result["status"] == "success":
            success_count += 1
            for k, v in scores_dict.items():
                total_scores[k] += v

        if (i + 1) % 10 == 0 or (i + 1) == len(data):
            save_results(args, results, len(data), success_count, total_scores)

    save_results(args, results, len(data), success_count, total_scores)
    avg_scores = {k: total_scores[k] / success_count for k in total_scores} if success_count else {k: 0.0 for k in total_scores}

    logger.info(f"Evaluation complete. Success: {success_count}/{len(data)}")
    for k, v in avg_scores.items():
        logger.info(f"  {k}: {v:.4f}{' (' + f'{v*100:.2f}%' + ')' if 'accuracy' in k else ''}")

    return results, avg_scores


def save_results(args, results, total, success, total_scores):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "llava_med_finetuned"
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
        "model": args.model_name,
        "dataset_type": args.dataset_type,
        "total_samples": total,
        "successful_samples": success,
        "average_scores": avg_scores,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {out_dir}")


def main():
    args = setup_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    data = load_data(args)
    run_evaluations(args, data)


if __name__ == "__main__":
    main()