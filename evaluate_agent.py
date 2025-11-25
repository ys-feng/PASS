import os
import json
import time
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import csv
import openai
import tiktoken
import re
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import httpx

# Import Agent-based components
from core.agent_controller import MultiLayerController, SentenceEncoder
from core.agent_workflow import AgentWorkflow
from core.agent_utils import calculate_cost
from tools import (
    ChestXRayClassifierTool, ChestXRaySegmentationTool,
    ChestXRayReportGeneratorTool, XRayVQATool, LlavaMedTool,
    ImageVisualizerTool
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_args():
    parser = argparse.ArgumentParser(description="Evaluate the Agent controller.")
    
    parser.add_argument("--data_root", type=str, default="data", help="Root directory of the dataset.")
    parser.add_argument("--eval_split", type=str, default="new_validate.json", help="Evaluation set filename.")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_agent", help="Output directory.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to the controller checkpoint file (.pth).")
    
    parser.add_argument("--device", type=str, default="cuda", help="Evaluation device (cuda or cpu).")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Specify GPU IDs to use, comma-separated.")
    parser.add_argument("--distributed_tools", action='store_true', help="Whether to distribute tools to different GPUs.")
    parser.add_argument("--offload_large_tools", action='store_true', help="Offload large tools to other GPUs.")
    parser.add_argument("--large_tool_threshold", type=float, default=4.0, help="VRAM threshold (GB) for large tools.")
    
    parser.add_argument("--controller_hidden_dim", type=int, default=32, help="Controller hidden layer dimension.")
    parser.add_argument("--controller_num_layers", type=int, default=4, help="Number of controller layers.")
    parser.add_argument("--encoder_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Sentence encoder model.")
    
    parser.add_argument("--log_tools", action='store_true', help="Whether to log tool executions.")
    parser.add_argument("--log_dir", type=str, default="logs/agent_eval_executions", help="Directory for workflow execution logs.")
    parser.add_argument("--max_concurrent_runs", type=int, default=10, help="Maximum number of concurrent workflow executions.")
    parser.add_argument("--llm_eval_model", type=str, default="gpt-4o-mini", help="Model for LLM-as-judge evaluation.")
    parser.add_argument("--model_dir", type=str, default=os.getenv("MODEL_DIR", "model-weights"), help="Directory for tool model weights.")
    parser.add_argument("--temp_dir", type=str, default=os.getenv("TEMP_DIR", "temp"), help="Directory for temporary files.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (for testing).")

    return parser.parse_args()

def load_slake_dataset(data_root: str, split_file: str) -> List[Dict]:
    json_path = Path(data_root) / split_file
    img_dir = Path(data_root) / "Slake1.0" / "imgs"
    
    if not json_path.exists():
        logger.error(f"Dataset file not found: {json_path}")
        return []
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        processed_data = []
        for item in raw_data:
            img_name = item.get("img_name")
            question = item.get("question")
            answer = item.get("answer")
            qid = item.get("qid")
            if img_name and question:
                full_image_path = img_dir / img_name 
                if full_image_path.exists():
                    processed_data.append({
                        "qid": qid,
                        "query": question,
                        "image_path": str(full_image_path),
                        "answer": answer
                    })
                else:
                    logger.warning(f"Image file not found: {full_image_path}")
            else:
                 logger.warning(f"Skipping incomplete entry: {item}")
                 
        logger.info(f"Loaded {len(processed_data)} valid samples from {json_path}.")
        return processed_data
        
    except Exception as e:
        logger.error(f"Failed to load dataset {json_path}: {e}", exc_info=True)
        return []

# Proxy management functions
def disable_proxy():
    proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'all_proxy']
    saved_proxies = {}
    for var in proxy_vars:
        if var in os.environ:
            saved_proxies[var] = os.environ.pop(var)
    return saved_proxies

def restore_proxy(saved_proxies):
    for var, value in saved_proxies.items():
        os.environ[var] = value

# Setup OpenAI clients
socks5_proxy = os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
try:
    import httpx_socks
    SOCKS_SUPPORTED = True
except ImportError:
    SOCKS_SUPPORTED = False
    logger.warning("httpx[socks] not installed. SOCKS5 proxy support will be disabled.")

if socks5_proxy and SOCKS_SUPPORTED:
    logger.info(f"Using SOCKS5 proxy for OpenAI: {socks5_proxy}")
    transport = httpx_socks.AsyncProxyTransport.from_url(socks5_proxy)
    async_http_client = httpx.AsyncClient(transport=transport)
else:
    async_http_client = httpx.AsyncClient()

try:
    async_openai_client = AsyncOpenAI(http_client=async_http_client)
    if socks5_proxy and SOCKS_SUPPORTED:
        sync_http_client = httpx.Client(proxy=socks5_proxy)
    else:
        sync_http_client = httpx.Client()
    sync_openai_client = openai.OpenAI(http_client=sync_http_client)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    async_openai_client = None
    sync_openai_client = None

similarity_encoder = None

async def get_gpt_answer(messages: list, model: str = "gpt-4o-mini") -> Optional[str]:
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set!")
            return None

        proxy_url = os.getenv("ALL_PROXY")
        transport = httpx.AsyncHTTPTransport(proxy=proxy_url)
        http_client = httpx.AsyncClient(transport=transport, timeout=120.0)
        client = openai.AsyncOpenAI(api_key=api_key, http_client=http_client)
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=200 
        )
        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()
        else:
            logger.warning(f"GPT API returned an invalid response")
            return None
    except Exception as e:
        logger.error(f"Failed to get answer from GPT-{model}: {e}")
        return None
    finally:
        if 'http_client' in locals() and not http_client.is_closed:
            await http_client.aclose()

def llm_as_judge(text1: str, text2: str, model: str = "gpt-4o-mini") -> float:
    if not sync_openai_client:
        logger.error("Sync OpenAI client not initialized.")
        return 0.0
        
    prompt = f"""You are an expert evaluator assessing the quality of generated answers against the ground truth. 
Evaluate text2 (Model Generated Answer) against text1 (Ground Truth Answer) using these criteria:
1. **Correctness**: Accuracy of the information.
2. **Completeness**: Coverage of key points from text1.
3. **Relevance**: Pertinence to the question/context.
4. **Coherence**: Clarity and logical flow.
**SCORING RUBRIC**:
- **1.00 (Clinically Correct)**: Clinically correct, captures all major key points.
- **0.90 (Mostly Correct)**: Generally correct, covers most key points.
- **0.67 (Partially Correct)**: Captures some key points, notable omissions.
- **0.33 (Partially Incorrect)**: Limited understanding with major errors.
- **0.00 (Incorrect)**: Completely incorrect or irrelevant.
Provide a single numeric score: 0.0, 0.33, 0.67, 0.9, or 1.0. Respond with ONLY the numeric score.
Ground Truth Answer: {text1}
Model Generated Answer: {text2}
"""
    try:
        response = sync_openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        score_text = response.choices[0].message.content.strip()
        score_match = re.search(r'([0-9]*\.?[0-9]+)', score_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            logger.error(f"Could not extract score from response: {score_text}")
            return 0.0
        final_score = max(0.0, min(1.0, score))
        return final_score
    except Exception as e:
        logger.error(f"LLM evaluation call failed: {e}")
        return 0.0

def aggregate_info_for_prompt(query: str, result: Dict) -> str:
    gathered_info_parts = []
    final_result = result.get("final_result")
    if final_result and isinstance(final_result, dict):
        final_response = final_result.get("response")
        if final_response:
            if isinstance(final_response, dict):
                final_response_text = final_response.get('text', str(final_response))
            else:
                final_response_text = str(final_response)
            gathered_info_parts.append(f"Agent's Conclusion: {final_response_text[:400]}")

    if not gathered_info_parts:
        layer_results = result.get("layer_results", [])
        tool_summaries = []
        for layer in layer_results:
            for op_result in layer:
                op_name = op_result.get("operator")
                op_output = op_result.get("result", {})
                summary = str(op_output)[:250]
                tool_summaries.append(f"- {op_name}: {summary}")

        if tool_summaries:
            gathered_info_parts.append("\n".join(tool_summaries))
    
    gathered_info = "\n".join(gathered_info_parts) if gathered_info_parts else "No information gathered."
    MAX_CHARS = 2000
    if len(gathered_info) > MAX_CHARS:
        gathered_info = gathered_info[:MAX_CHARS] + "..."

    prompt = f"""Use the following info to answer the query in 200 tokens. If info is not enough, say so.
Query: {query}
Info: {gathered_info}
Answer:"""
    return prompt

async def run_workflow_and_get_feedback(
    workflow: AgentWorkflow, 
    query: str, 
    image_path: str,
    ground_truth_answer: Optional[str] = None
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[float]]:
    score = 0.0 
    cost = 0.0 
    gpt_answer = None
    latency = 0.0

    try:
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                workflow.process_query(query, image_path), 
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Workflow execution timed out (30s) for query: {query[:50]}...")
            return None, None, None, None
        
        cost = calculate_cost(result)

        if ground_truth_answer and async_openai_client:
            prompt = aggregate_info_for_prompt(query, result)
            gpt_answer = await get_gpt_answer([
                {"role": "system", "content": "You are a helpful assistant answering medical questions."},
                {"role": "user", "content": prompt}
            ])

            if gpt_answer:
                score = llm_as_judge(ground_truth_answer, gpt_answer)
            else:
                logger.warning("Failed to get answer from GPT")
                score = 0.0
        else:
            score = 0.0 
        
        latency = time.time() - start_time
        
        return score, cost, gpt_answer, latency

    except Exception as e:
        logger.error(f"Workflow execution failed for query '{query}': {e}")
        return None, None, None, None

async def evaluate_dataset(
    workflow: AgentWorkflow, 
    eval_data: List[Dict], 
    max_concurrent_runs: int,
    args: argparse.Namespace
) -> Dict[str, Any]:
    workflow.controller.eval()
    total_score, total_cost, total_latency = 0.0, 0.0, 0.0
    evaluated_count, total_items = 0, len(eval_data)
    evaluation_results = []
    
    semaphore = asyncio.Semaphore(max_concurrent_runs)

    async def eval_item(item_data, item_idx, pbar):
        async with semaphore:
            query = item_data['query']
            image_path = item_data['image_path']
            answer = item_data.get('answer')
            try:
                with torch.no_grad():
                    feedback = await run_workflow_and_get_feedback(
                         workflow, query, image_path, answer
                     )
                    if feedback is not None:
                        score, cost, gpt_answer, latency = feedback
                        pbar.update(1)
                        return score, cost, gpt_answer, latency, item_data
                    else:
                        pbar.update(1)
                        return None 
            except Exception as e:
                logger.error(f"Evaluation failed for '{query[:50]}...': {e}")
                pbar.update(1)
                return None

    tasks = []
    with tqdm(total=total_items, desc="Evaluating") as pbar:
        for idx, item in enumerate(eval_data):
            tasks.append(eval_item(item, idx, pbar))
        eval_results_raw = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_eval_results = []
    for res in eval_results_raw:
        if res is not None and not isinstance(res, Exception) and res[0] is not None:
            valid_eval_results.append(res)

    if valid_eval_results:
        total_score = sum(res[0] for res in valid_eval_results)
        total_cost = sum(res[1] for res in valid_eval_results if res[1] is not None)
        total_latency = sum(res[3] for res in valid_eval_results if res[3] is not None)
        evaluated_count = len(valid_eval_results)
        
        for score, cost, gpt_answer, latency, item_data in valid_eval_results:
            model_prediction = gpt_answer if gpt_answer is not None else "[FAILED]"
            ground_truth = item_data.get('answer', '')
            query = item_data.get('query', '')
            qid = item_data.get('qid', '')
            evaluation_results.append({
                "qid": qid,
                "query": query,
                "ground_truth": ground_truth,
                "prediction": model_prediction,
                "score": score,
                "cost": cost,
                "latency": latency
            })

    avg_score = total_score / evaluated_count if evaluated_count > 0 else 0.0
    avg_cost = total_cost / evaluated_count if evaluated_count > 0 else 0.0
    avg_latency = total_latency / evaluated_count if evaluated_count > 0 else 0.0
    failure_rate = (total_items - evaluated_count) / total_items if total_items > 0 else 0.0

    results = {
        "average_score": avg_score,
        "average_cost": avg_cost, 
        "failure_rate": failure_rate,
        "average_latency": avg_latency,
        "total_evaluated": evaluated_count,
        "total_samples": total_items,
        "detailed_results": evaluation_results
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Evaluated: {evaluated_count}/{total_items} samples")
    logger.info(f"Failure rate: {failure_rate:.2%}")
    logger.info(f"Average Score (Accuracy): {avg_score:.4f}")
    logger.info(f"Average Cost: {avg_cost:.4f}")
    logger.info(f"Average Latency: {avg_latency:.2f}s")
    logger.info(f"{'='*60}\n")

    return results

def initialize_tools(model_dir, temp_dir, device, gpu_ids=None, distributed_tools=False, args=None) -> Dict[str, Any]:
    if args is None:
        class DefaultArgs:
            offload_large_tools = False
            large_tool_threshold = 4.0
        args = DefaultArgs()
    
    logger.info("Temporarily disabling proxy for Hugging Face downloads...")
    saved_proxies = disable_proxy()
    
    tools_dict = {}
    tool_classes = {
        "ChestXRayClassifierTool": ChestXRayClassifierTool,
        "ChestXRaySegmentationTool": ChestXRaySegmentationTool,
        "ChestXRayReportGeneratorTool": ChestXRayReportGeneratorTool,
        "XRayVQATool": XRayVQATool,
        "LlavaMedTool": LlavaMedTool,
        "ImageVisualizerTool": ImageVisualizerTool 
    }
    
    tool_complexity = {
        "ChestXRayReportGeneratorTool": {"memory_gb": 8.0, "priority": "high"},
        "LlavaMedTool": {"memory_gb": 6.0, "priority": "high"},
        "XRayVQATool": {"memory_gb": 4.0, "priority": "medium"},
        "ChestXRaySegmentationTool": {"memory_gb": 3.0, "priority": "medium"},
        "ChestXRayClassifierTool": {"memory_gb": 2.0, "priority": "low"},
        "ImageVisualizerTool": {"memory_gb": 0.1, "priority": "cpu"},
    }
    
    gpu_assignment = {name: device for name in tool_classes.keys()}
    if distributed_tools and gpu_ids and len(gpu_ids) > 1:
        logger.info(f"Distributing tools across GPUs: {gpu_ids}")
        gpu_load = {gpu_id: 0.0 for gpu_id in gpu_ids}
        for tool_name in tool_classes.keys():
            tool_info = tool_complexity.get(tool_name, {"memory_gb": 2.0, "priority": "medium"})
            if tool_info["priority"] == "cpu":
                gpu_assignment[tool_name] = None
            else:
                best_gpu = min(gpu_ids, key=lambda x: gpu_load[x])
                gpu_assignment[tool_name] = best_gpu
                gpu_load[best_gpu] += tool_info["memory_gb"]
    
    for name, tool_class in tool_classes.items():
        try:
            tool_device = gpu_assignment.get(name, device)
            if tool_device is not None and isinstance(tool_device, int):
                tool_device = torch.device(f"cuda:{tool_device}")
            elif tool_device is None:
                tool_device = torch.device("cpu")
            
            kwargs = {"device": tool_device} if name != "ImageVisualizerTool" else {}
            
            if name in ["ChestXRayReportGeneratorTool", "XRayVQATool", "LlavaMedTool"]:
                kwargs["cache_dir"] = model_dir
            
            if name != "ImageVisualizerTool":    
                tools_dict[name] = tool_class(**kwargs)
            else:
                tools_dict[name] = tool_class()
            
            logger.info(f"Tool {name} initialized")
            
        except Exception as e:
             logger.error(f"Failed to initialize tool {name}: {e}")
    
    logger.info(f"Successfully initialized {len(tools_dict)}/{len(tool_classes)} tools")
    
    logger.info("Restoring proxy settings...")
    restore_proxy(saved_proxies)
    
    return tools_dict

async def main():
    args = setup_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if args.gpu_ids and torch.cuda.is_available():
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    else:
        gpu_ids = [0] if torch.cuda.is_available() else []
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Loading evaluation data from {args.data_root}...")
    eval_data = load_slake_dataset(args.data_root, args.eval_split)
    
    if not eval_data:
        logger.error("Failed to load evaluation data. Exiting.")
        return
    
    if args.max_samples:
        eval_data = eval_data[:args.max_samples]
        logger.info(f"Limited to {len(eval_data)} samples for evaluation")
        
    try:
        logger.info(f"Initializing sentence encoder: {args.encoder_model}")
        sentence_encoder = SentenceEncoder(model_name=args.encoder_model, device=device)
        _dummy_emb = sentence_encoder.encode("test")
        input_dim = _dummy_emb.shape[0]
        logger.info(f"Sentence encoder input dimension: {input_dim}")
    except Exception as e:
        logger.error(f"Failed to initialize sentence encoder: {e}")
        input_dim = 384

    logger.info("Initializing controller...")
    controller = MultiLayerController(
        input_dim=input_dim,
        hidden_dim=args.controller_hidden_dim,
        num_layers=args.controller_num_layers,
        device=device
    ).to(device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
        return
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['controller_state_dict']
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            logger.info("Removed 'module.' prefix from checkpoint")
        
        controller.load_state_dict(state_dict)
        controller.eval()
        logger.info("Checkpoint loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return

    logger.info("Initializing tools...")
    tools_dict = initialize_tools(
        args.model_dir, args.temp_dir, device, 
        gpu_ids, args.distributed_tools, args
    )
    if not tools_dict:
         logger.error("Failed to initialize any tools. Exiting.")
         return

    logger.info("Initializing Agent workflow...")
    workflow = AgentWorkflow(
        tools_dict=tools_dict,
        controller=controller,
        log_tools=args.log_tools,
        log_dir=args.log_dir,
        device=device
    )
    
    logger.info(f"Starting evaluation on {len(eval_data)} samples...")
    results = await evaluate_dataset(
        workflow, eval_data, args.max_concurrent_runs, args
    )
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    results_file = output_dir / f"evaluation_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "checkpoint": str(checkpoint_path),
                "eval_split": args.eval_split,
                "timestamp": timestamp,
                "total_samples": results["total_samples"],
                "evaluated_samples": results["total_evaluated"]
            },
            "metrics": {
                "average_score": results["average_score"],
                "average_cost": results["average_cost"],
                "average_latency": results["average_latency"],
                "failure_rate": results["failure_rate"]
            },
            "detailed_results": results["detailed_results"]
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {results_file}")
    
    csv_file = output_dir / f"evaluation_summary_{timestamp}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["QID", "Query", "Ground Truth", "Prediction", "Score", "Cost", "Latency"])
        for item in results["detailed_results"]:
            writer.writerow([
                item["qid"],
                item["query"],
                item["ground_truth"],
                item["prediction"],
                item["score"],
                item["cost"],
                item["latency"]
            ])
    
    logger.info(f"CSV summary saved to: {csv_file}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
    except Exception as e:
        logger.critical(f"An uncaught error occurred during evaluation: {e}", exc_info=True)
