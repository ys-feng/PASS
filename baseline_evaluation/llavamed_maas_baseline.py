"""
LLaVA-Med + MAAS Baseline
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import openai
from sentence_transformers import SentenceTransformer

from core.agent_workflow import AgentWorkflow as MaaSWorkflow
from core.agent_controller import MultiLayerController
from core.agent_utils import calculate_cost
from tools import (
    ChestXRayClassifierTool,
    ChestXRaySegmentationTool,
    ChestXRayReportGeneratorTool,
    XRayVQATool,
    XRayPhraseGroundingTool,
    LlavaMedTool
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import time

try:
    from nltk.translate.meteor_score import meteor_score
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False
    logger.warning("METEOR not available, will use 0.0 as default value")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key":
    logger.warning("⚠️  OpenAI API Key not set or invalid, please set environment variable OPENAI_API_KEY")
    logger.warning("⚠️  Evaluation will continue but cannot call GPT API for scoring")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

os.environ['http_proxy'] = "socks5://hkumedai:bestpaper@66.42.72.8:54321"
os.environ['https_proxy'] = "socks5://hkumedai:bestpaper@66.42.72.8:54321"

sync_openai_client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

async_openai_client = openai.AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

similarity_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def aggregate_info_for_prompt(query: str, workflow_result: dict) -> str:
    prompt_parts = [f"Question: {query}\n"]
    prompt_parts.append("Based on the following medical image analysis results, please provide a concise answer:\n")
    
    layer_results = workflow_result.get("layer_results", [])
    for layer_idx, layer in enumerate(layer_results):
        for step in layer:
            operator = step.get("operator", "Unknown")
            result = step.get("result", {})
            response = result.get("response", "")
            
            if response:
                prompt_parts.append(f"\n[{operator}]: {response}")
    
    prompt_parts.append("\n\nPlease provide a direct, concise answer to the question based on the above information.")
    return "\n".join(prompt_parts)


async def get_gpt_answer(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> Optional[str]:
    try:
        response = await async_openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"GPT API call failed: {e}")
        return None


def llm_as_judge(ground_truth: str, prediction: str, model: str = "gpt-4o-mini") -> float:
    judge_prompt = f"""You are an expert medical evaluator. Compare the predicted answer with the ground truth answer.

Ground Truth: {ground_truth}
Predicted Answer: {prediction}

Rate the prediction's accuracy on a scale:
- 0.0: Completely wrong or contradictory
- 0.33: Partially correct but missing key information
- 0.67: Mostly correct with minor errors
- 0.9: Correct with very minor imperfections
- 1.0: Perfectly correct

Respond with ONLY the numeric score (0.0, 0.33, 0.67, 0.9, or 1.0)."""

    try:
        response = sync_openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=10
        )
        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.error(f"LLM judge scoring failed: {e}")
        return 0.0


def detect_hallucination(ground_truth: str, prediction: str, model: str = "gpt-4o-mini") -> str:
    hallucination_prompt = f"""You are a medical fact-checker. Determine if the predicted answer contains hallucinations compared to the ground truth.

Ground Truth: {ground_truth}
Predicted Answer: {prediction}

Classify the hallucination level:
- "No Hallucination": The prediction is factually consistent with the ground truth
- "Minor Extrapolation": The prediction adds plausible medical details not in ground truth but not contradictory
- "Major Hallucination": The prediction contains significant factual errors or contradictions

Respond with ONLY one of these three exact phrases."""

    try:
        response = sync_openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": hallucination_prompt}],
            temperature=0.0,
            max_tokens=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Hallucination detection failed: {e}")
        return "Unknown"


def calculate_bleu(reference: str, prediction: str) -> float:
    try:
        reference_tokens = reference.lower().split()
        prediction_tokens = prediction.lower().split()
        
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([reference_tokens], prediction_tokens, 
                            smoothing_function=smoothing)
        return score
    except Exception as e:
        logger.error(f"BLEU calculation failed: {e}")
        return 0.0


def calculate_meteor(reference: str, prediction: str) -> float:
    if not METEOR_AVAILABLE:
        return 0.0
    
    try:
        reference_tokens = reference.lower().split()
        prediction_tokens = prediction.lower().split()
        score = meteor_score([reference_tokens], prediction_tokens)
        return score
    except Exception as e:
        logger.error(f"METEOR calculation failed: {e}")
        return 0.0


def calculate_rouge(reference: str, prediction: str) -> float:
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure
    except Exception as e:
        logger.error(f"ROUGE calculation failed: {e}")
        return 0.0


def calculate_semantic_similarity(reference: str, prediction: str) -> float:
    try:
        embeddings = similarity_encoder.encode([reference, prediction])
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        return float(similarity)
    except Exception as e:
        logger.error(f"Semantic similarity calculation failed: {e}")
        return 0.0


async def evaluate_single_item(
    workflow: MaaSWorkflow,
    item: Dict[str, Any],
    base_image_path: str
) -> Dict[str, Any]:
    query = item.get("query") or item.get("question", "")
    image_file = item.get("image") or item.get("img_name", "")
    ground_truth = item.get("answer", "")
    
    image_path = os.path.join(base_image_path, image_file)
    
    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}")
        return {
            "query": query,
            "image": image_file,
            "ground_truth": ground_truth,
            "architecture": [],
            "generated_answer": "Error: Image not found",
            "score": 0.0,
            "llm_judge": 0.0,
            "bleu": 0.0,
            "meteor": 0.0,
            "rouge_l": 0.0,
            "similarity": 0.0,
            "cost": 0.0,
            "latency": 0.0,
            "hallucination": "Unknown",
            "status": "error"
        }
    
    try:
        start_time = time.time()
        
        result = await workflow.process_query(query, image_path)
        
        latency = time.time() - start_time
        
        cost = calculate_cost(result)
        
        prompt = aggregate_info_for_prompt(query, result)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant answering medical questions based on provided context."},
            {"role": "user", "content": prompt}
        ]
        generated_answer = await get_gpt_answer(messages)
        
        llm_judge_score = 0.0
        bleu_score = 0.0
        meteor_score_val = 0.0
        rouge_l_score = 0.0
        similarity_score = 0.0
        hallucination = "Unknown"
        
        if generated_answer and ground_truth:
            llm_judge_score = llm_as_judge(ground_truth, generated_answer)
            
            bleu_score = calculate_bleu(ground_truth, generated_answer)
            
            meteor_score_val = calculate_meteor(ground_truth, generated_answer)
            
            rouge_l_score = calculate_rouge(ground_truth, generated_answer)
            
            similarity_score = calculate_semantic_similarity(ground_truth, generated_answer)
            
            hallucination = detect_hallucination(ground_truth, generated_answer)
        
        logger.info(f"Query: {query[:50]}... | LLM-J: {llm_judge_score:.2f} | BLEU: {bleu_score:.2f} | Sim: {similarity_score:.2f}")
        
        return {
            "query": query,
            "image": image_file,
            "ground_truth": ground_truth,
            "architecture": result.get("architecture", []),
            "generated_answer": generated_answer or "No answer generated",
            "score": llm_judge_score,
            "llm_judge": llm_judge_score,
            "bleu": bleu_score,
            "meteor": meteor_score_val,
            "rouge_l": rouge_l_score,
            "similarity": similarity_score,
            "cost": cost,
            "latency": latency,
            "hallucination": hallucination,
            "status": result.get("status", "unknown"),
            "workflow_result": result
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return {
            "query": query,
            "image": image_file,
            "ground_truth": ground_truth,
            "architecture": [],
            "generated_answer": f"Error: {str(e)}",
            "score": 0.0,
            "llm_judge": 0.0,
            "bleu": 0.0,
            "meteor": 0.0,
            "rouge_l": 0.0,
            "similarity": 0.0,
            "cost": 0.0,
            "latency": 0.0,
            "hallucination": "Unknown",
            "status": "error"
        }


async def evaluate_dataset(
    workflow: MaaSWorkflow,
    data_path: str,
    base_image_path: str,
    output_dir: str,
    max_samples: Optional[int] = None
):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    logger.info(f"Starting evaluation on {len(data)} samples...")
    
    semaphore = asyncio.Semaphore(3)
    
    async def eval_with_semaphore(item):
        async with semaphore:
            return await evaluate_single_item(workflow, item, base_image_path)
    
    tasks = [eval_with_semaphore(item) for item in data]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Sample {i} evaluation error: {result}")
        else:
            valid_results.append(result)
    
    scores = [r["score"] for r in valid_results if r["status"] != "error"]
    costs = [r["cost"] for r in valid_results if r["status"] != "error"]
    hallucinations = [r["hallucination"] for r in valid_results if r["status"] != "error"]
    
    metrics = {
        "total_samples": len(data),
        "successful_samples": len([r for r in valid_results if r["status"] != "error"]),
        "average_score": sum(scores) / len(scores) if scores else 0.0,
        "average_cost": sum(costs) / len(costs) if costs else 0.0,
        "hallucination_rate": len([h for h in hallucinations if h == "Major Hallucination"]) / len(hallucinations) * 100 if hallucinations else 0.0,
        "no_hallucination_rate": len([h for h in hallucinations if h == "No Hallucination"]) / len(hallucinations) * 100 if hallucinations else 0.0,
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = os.path.join(output_dir, f"llavamed_maas_results_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "method": "LLaVA-Med + MAAS",
                "data_path": data_path,
                "timestamp": timestamp,
                "controller_checkpoint": "trained MAAS controller"
            },
            "metrics": metrics,
            "results": valid_results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation completed!")
    logger.info(f"Total samples: {metrics['total_samples']}")
    logger.info(f"Successful samples: {metrics['successful_samples']}")
    logger.info(f"Average score: {metrics['average_score']:.4f}")
    logger.info(f"Average cost: {metrics['average_cost']:.4f} CU")
    logger.info(f"Hallucination rate: {metrics['hallucination_rate']:.2f}%")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"{'='*60}\n")


def load_trained_controller(checkpoint_path: str, device: torch.device) -> MultiLayerController:
    logger.info(f"Loading trained controller from {checkpoint_path}...")
    
    controller = MultiLayerController(device=device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['controller_state_dict']
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            logger.info("Detected DataParallel model, removed 'module.' prefix")
        
        controller.load_state_dict(state_dict)
        controller.eval()
        logger.info("Controller loaded successfully!")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}, using untrained controller")
    
    return controller


async def evaluate_on_gpu(
    gpu_id: int,
    data_shard: List[Dict],
    checkpoint_path: str,
    image_dir: str,
    output_dir: str
) -> List[Dict]:
    device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"[GPU {gpu_id}] Starting evaluation on {len(data_shard)} samples")
    
    tools_dict = {
        'ChestXRayClassifierTool': ChestXRayClassifierTool(),
        'ChestXRaySegmentationTool': ChestXRaySegmentationTool(),
        'ChestXRayReportGeneratorTool': ChestXRayReportGeneratorTool(),
        'XRayVQATool': XRayVQATool(),
        'LlavaMedTool': LlavaMedTool()
    }
    
    logger.info(f"[GPU {gpu_id}] Loaded {len(tools_dict)} tools (skipped MAIRA-2)")
    
    controller = load_trained_controller(checkpoint_path, device)
    
    workflow = MaaSWorkflow(
        tools_dict=tools_dict,
        controller=controller,
        log_tools=True,
        log_dir=f"/userhome/cs3/fengys/EVA-RAX/logs/llavamed_maas_gpu{gpu_id}",
        device=device
    )
    
    results = []
    for idx, item in enumerate(data_shard):
        try:
            result = await evaluate_single_item(workflow, item, image_dir)
            results.append(result)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"[GPU {gpu_id}] Progress: {idx + 1}/{len(data_shard)}")
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Sample {idx} evaluation failed: {e}")
            results.append({
                "query": item.get("query", ""),
                "image": item.get("image", ""),
                "ground_truth": item.get("answer", ""),
                "architecture": [],
                "generated_answer": f"Error: {str(e)}",
                "score": 0.0,
                "cost": 0.0,
                "hallucination": "Unknown",
                "status": "error"
            })
    
    logger.info(f"[GPU {gpu_id}] Evaluation completed! Collected {len(results)} results")
    return results


def save_combined_results(
    results: List[Dict],
    output_dir: str,
    data_path: str
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    successful_results = [r for r in results if r["status"] != "error"]
    
    llm_judge_scores = [r.get("llm_judge", 0.0) for r in successful_results]
    bleu_scores = [r.get("bleu", 0.0) for r in successful_results]
    meteor_scores = [r.get("meteor", 0.0) for r in successful_results]
    rouge_l_scores = [r.get("rouge_l", 0.0) for r in successful_results]
    similarity_scores = [r.get("similarity", 0.0) for r in successful_results]
    latencies = [r.get("latency", 0.0) for r in successful_results]
    costs = [r.get("cost", 0.0) for r in successful_results]
    hallucinations = [r.get("hallucination", "Unknown") for r in successful_results]
    
    metrics = {
        "total_samples": len(results),
        "successful_samples": len(successful_results),
        
        "accuracy": sum(llm_judge_scores) / len(llm_judge_scores) * 100 if llm_judge_scores else 0.0,
        
        "llm_judge": sum(llm_judge_scores) / len(llm_judge_scores) * 100 if llm_judge_scores else 0.0,
        
        "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        
        "meteor": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0,
        
        "rouge_l": sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0,
        
        "similarity": sum(similarity_scores) / len(similarity_scores) * 100 if similarity_scores else 0.0,
        
        "latency": sum(latencies) / len(latencies) if latencies else 0.0,
        
        "average_cost": sum(costs) / len(costs) if costs else 0.0,
        "hallucination_rate": len([h for h in hallucinations if h == "Major Hallucination"]) / len(hallucinations) * 100 if hallucinations else 0.0,
        "no_hallucination_rate": len([h for h in hallucinations if h == "No Hallucination"]) / len(hallucinations) * 100 if hallucinations else 0.0,
    }
    
    results_file = os.path.join(output_dir, f"llavamed_maas_results_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "method": "LLaVA-Med + MAAS",
                "data_path": data_path,
                "timestamp": timestamp,
                "num_gpus": torch.cuda.device_count()
            },
            "metrics": metrics,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*80}")
    logger.info("🎉 All GPU evaluations completed!")
    logger.info(f"{'='*80}")
    logger.info("\n📊 formatted results:")
    logger.info(f"{'Method':<30} {'Acc.↑':<10} {'LLM-J.↑':<10} {'BLEU↑':<10} {'METEOR↑':<10} {'ROUGE-L↑':<10} {'Sim.↑':<10} {'Lat.↓':<10}")
    logger.info(f"{'-'*110}")
    logger.info(
        f"{'LLaVA-Med + MAAS':<30} "
        f"{metrics['accuracy']:<10.2f} "
        f"{metrics['llm_judge']:<10.2f} "
        f"{metrics['bleu']:<10.4f} "
        f"{metrics['meteor']:<10.4f} "
        f"{metrics['rouge_l']:<10.4f} "
        f"{metrics['similarity']:<10.2f} "
        f"{metrics['latency']:<10.2f}"
    )
    logger.info("\nOther metrics:")
    logger.info(f"  Total samples: {metrics['total_samples']}")
    logger.info(f"  Successful samples: {metrics['successful_samples']}")
    logger.info(f"  Average cost: {metrics['average_cost']:.4f} CU")
    logger.info(f"  Hallucination rate: {metrics['hallucination_rate']:.2f}%")
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"{'='*80}\n")


async def main():
    parser = argparse.ArgumentParser(description="LLaVA-Med + MAAS Baseline Evaluation")
    parser.add_argument("--data_path", type=str, 
                       default="data/CAB-E/train.json",
                       help="Dataset path")
    parser.add_argument("--image_dir", type=str,
                       default="data/Slake1.0/imgs",
                       help="Image directory path")
    parser.add_argument("--checkpoint", type=str,
                       default="outputs/train_agent/best_model.pth",
                       help="Trained MAAS controller checkpoint path")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/llavamed_maas",
                       help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of evaluation samples (for testing)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Computation device")
    parser.add_argument("--num_gpus", type=int, default=4,
                       help="Number of GPUs to use (defaults to all available GPUs)")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini",
                       help="LLM model name")
    
    args = parser.parse_args()
    
    num_gpus = torch.cuda.device_count()
    if args.num_gpus > num_gpus:
        logger.warning(f"Requested {args.num_gpus} GPUs, but only {num_gpus} are available. Using {num_gpus} GPUs.")
        args.num_gpus = num_gpus
    
    logger.info(f"🚀 Using {args.num_gpus} GPUs for parallel evaluation")
    
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if args.max_samples:
        data = data[:args.max_samples]
    
    total_samples = len(data)
    logger.info(f"Total samples: {total_samples}")
    
    samples_per_gpu = (total_samples + args.num_gpus - 1) // args.num_gpus
    data_shards = [data[i*samples_per_gpu:(i+1)*samples_per_gpu] for i in range(args.num_gpus)]
    
    logger.info("Data sharding completed:")
    for i, shard in enumerate(data_shards):
        logger.info(f"  GPU {i}: {len(shard)} samples")
    
    all_results = []
    for gpu_id in range(args.num_gpus):
        if len(data_shards[gpu_id]) == 0:
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting evaluation on GPU {gpu_id} with {len(data_shards[gpu_id])} samples...")
        logger.info(f"{'='*60}")
        
        try:
            gpu_results = await evaluate_on_gpu(
                gpu_id=gpu_id,
                data_shard=data_shards[gpu_id],
                checkpoint_path=args.checkpoint,
                image_dir=args.image_dir,
                output_dir=args.output_dir
            )
            all_results.append(gpu_results)
            
            torch.cuda.empty_cache()
            logger.info(f"GPU {gpu_id} evaluation completed, memory cleared\n")
            
        except Exception as e:
            logger.error(f"GPU {gpu_id} Evaluation failed: {e}", exc_info=True)
            all_results.append([])
    
    combined_results = []
    for gpu_results in all_results:
        if isinstance(gpu_results, Exception):
            logger.error(f"GPU evaluation error: {gpu_results}")
        else:
            combined_results.extend(gpu_results)
    
    logger.info(f"All GPU evaluations completed! Collected {len(combined_results)} results")
    
    save_combined_results(combined_results, args.output_dir, args.data_path)


if __name__ == "__main__":
    asyncio.run(main())
