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
import wandb
import openai
import tiktoken
import re
import gc
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import httpx

# Import Agent-based components
from core.agent_controller import MultiLayerController, SentenceEncoder
from core.agent_operators import get_operator_descriptions, build_operator_mapping
from core.agent_workflow import AgentWorkflow
from core.agent_optimizer import AgentOptimizer
from core.agent_utils import get_operator_embeddings, calculate_score, calculate_cost
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

# GPU memory optimization
def optimize_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        try:
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_per_process_memory_fraction(0.85, device=i)
                    logger.info(f"GPU {i}: Setting memory usage limit to 85%")
            
            if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, 'set_allocator_max_split_size_mb'):
                torch.cuda.memory.set_allocator_max_split_size_mb(256)
            elif hasattr(torch.cuda, 'set_allocator_max_split_size_mb'):
                torch.cuda.set_allocator_max_split_size_mb(256)
            
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info("GPU memory optimization setup complete.")
            
            for i in range(torch.cuda.device_count()):
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
                reserved_memory = torch.cuda.memory_reserved(i) / 1024**3
                free_memory = total_memory - reserved_memory
                logger.info(f"GPU {i}: Total={total_memory:.2f}GB, Allocated={allocated_memory:.2f}GB, "
                          f"Reserved={reserved_memory:.2f}GB, Free={free_memory:.2f}GB")
                
        except Exception as e:
            logger.warning(f"GPU memory optimization setup failed, continuing with default settings: {e}")
    else:
        logger.info("No CUDA device detected, skipping GPU memory optimization.")

def json_serializer_default(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float16, np.float32, 
                        np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
         return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def setup_args():
    parser = argparse.ArgumentParser(description="Train the Agent controller.")
    
    parser.add_argument("--data_root", type=str, default="data", help="Root directory of the dataset.")
    parser.add_argument("--train_split", type=str, default="CAB-E/train.json", help="Training set filename.")
    parser.add_argument("--val_split", type=str, default="CAB-E/test.json", help="Validation set filename.")
    parser.add_argument("--output_dir", type=str, default="./outputs/train_agent", help="Output directory.")

    parser.add_argument("--cpr_epochs", type=int, default=2, help="Number of epochs for Contrastive Path Ranking (CPR).")
    parser.add_argument("--cpr_lr", type=float, default=1e-5, help="Learning rate for the CPR phase.")

    parser.add_argument("--checkpoint_to_load", type=str, default=None, help="Path to the controller checkpoint file to load.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=48, help="RL batch size, adjust based on GPU memory.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda or cpu).")
    parser.add_argument("--use_multi_gpu", action='store_true', help="Whether to use multi-GPU parallel training.")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Specify GPU IDs to use, comma-separated (e.g., 0,1,2).")
    parser.add_argument("--distributed_tools", action='store_true', help="Whether to distribute different tools to different GPUs.")
    parser.add_argument("--offload_large_tools", action='store_true', help="Offload only large tools (VQA, LLM, etc.) to other GPUs.")
    parser.add_argument("--large_tool_threshold", type=float, default=4.0, help="VRAM threshold (GB) for large tools.")
    parser.add_argument("--controller_hidden_dim", type=int, default=32, help="Controller hidden layer dimension.")
    parser.add_argument("--controller_num_layers", type=int, default=4, help="Number of controller layers.")
    parser.add_argument("--encoder_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Sentence encoder model.")
    parser.add_argument("--cost_weight", type=float, default=0.0003, help="Weight of cost in the utility function.")
    parser.add_argument("--log_tools", action='store_true', help="Whether to log tool executions.")
    parser.add_argument("--log_dir", type=str, default="logs/agent_train_executions", help="Directory for workflow execution logs.")
    parser.add_argument("--max_concurrent_runs", type=int, default=10, help="Maximum number of concurrent workflow executions.")
    parser.add_argument("--warmup_size", type=int, default=500, help="Number of samples for supervised warm-up.")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of epochs for supervised warm-up.")
    parser.add_argument("--llm_eval_batch_size", type=int, default=20, help="Batch size for batch LLM evaluation.")
    parser.add_argument("--llm_eval_model", type=str, default="gpt-4o-mini", help="Model for LLM-as-judge evaluation.")
    parser.add_argument("--model_dir", type=str, default=os.getenv("MODEL_DIR", "model-weights"), help="Directory for tool model weights.")
    parser.add_argument("--temp_dir", type=str, default=os.getenv("TEMP_DIR", "temp"), help="Directory for temporary files.")
    parser.add_argument("--eval_every_n_epochs", type=int, default=1, help="Evaluate every N epochs.")
    parser.add_argument("--save_every_n_epochs", type=float, default=1, help="Save a checkpoint every N epochs.")
    parser.add_argument("--use_wandb", action='store_true', help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="", help="Wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name (optional).")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name (optional).")
    parser.add_argument("--log_every_n_batches", type=int, default=1, help="Log training metrics every N batches.")
    parser.add_argument("--disable_warmup", action='store_true', help="Disable the supervised warm-up phase.")

    return parser.parse_args()

def setup_multi_gpu(args):
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, will use CPU.")
        return torch.device("cpu"), [], False
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} GPUs.")
    
    if not args.use_multi_gpu or num_gpus <= 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using single GPU/CPU mode: {device}")
        return device, [0] if torch.cuda.is_available() else [], False
    
    if args.gpu_ids is not None:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
            invalid_ids = [gpu_id for gpu_id in gpu_ids if gpu_id >= num_gpus or gpu_id < 0]
            if invalid_ids:
                logger.error(f"Invalid GPU IDs: {invalid_ids}. Available GPU IDs: 0-{num_gpus-1}")
                return torch.device("cuda:0"), [0], False
        except ValueError:
            logger.error(f"Invalid GPU ID format: {args.gpu_ids}")
            return torch.device("cuda:0"), [0], False
    else:
        gpu_ids = list(range(num_gpus))
    
    main_device = torch.device(f"cuda:{gpu_ids[0]}")
    logger.info(f"Using multi-GPU parallel training: GPUs {gpu_ids}, Main device: {main_device}")
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    return main_device, gpu_ids, True

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
            tools = item.get("tools")
            if img_name and question:
                full_image_path = img_dir / img_name 
                if full_image_path.exists():
                    processed_data.append({
                        "qid": qid,
                        "query": question,
                        "image_path": str(full_image_path),
                        "answer": answer,
                        "tools": tools 
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
    if socks5_proxy and not SOCKS_SUPPORTED:
        logger.warning("SOCKS proxy environment variable is set, but required library is missing.")
    async_http_client = httpx.AsyncClient()

try:
    async_openai_client = AsyncOpenAI(http_client=async_http_client)
    if socks5_proxy and SOCKS_SUPPORTED:
        sync_http_client = httpx.Client(proxy=socks5_proxy)
    else:
        sync_http_client = httpx.Client()
    sync_openai_client = openai.OpenAI(http_client=sync_http_client)
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
    async_openai_client = None
    sync_openai_client = None

similarity_model_name = "all-MiniLM-L6-v2" 
similarity_encoder = None

try:
    tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
except:
    logger.warning("Could not initialize tiktoken encoder, will use simple word count as a fallback.")
    tiktoken_encoder = None

async def get_gpt_answer(messages: list, model: str = "gpt-4o-mini") -> Optional[str]:
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set!")
            return None

        proxy_url = os.getenv("ALL_PROXY")
        http_client = None

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
            logger.warning(f"GPT API returned an invalid response: {response}")
            return None
    except Exception as e:
        logger.error(f"Failed to get answer from GPT-{model}: {e}", exc_info=True)
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
        logger.info(f"LLM-as-judge assigned score: {final_score:.2f}")
        return final_score
    except Exception as e:
        logger.error(f"LLM evaluation call failed: {e}", exc_info=True)
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
                if op_name and isinstance(op_output, dict):
                    summary = str(op_output)[:250]
                else:
                    summary = str(op_result)
                tool_summaries.append(f"- {op_name}: {summary[:250]}")  

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

def create_parallel_controller(controller, gpu_ids, use_multi_gpu):
    if use_multi_gpu and len(gpu_ids) > 1:
        logger.info(f"Wrapping controller with DataParallel on GPU devices: {gpu_ids}")
        device_ids = list(range(len(gpu_ids)))
        parallel_controller = torch.nn.DataParallel(controller, device_ids=device_ids)
        main_device = torch.device(f"cuda:{device_ids[0]}")
        return parallel_controller, main_device
    else:
        logger.info("Using single GPU controller.")
        return controller, controller.device

async def run_workflow_and_get_feedback(
    workflow: AgentWorkflow, 
    query: str, 
    image_path: str,
    ground_truth_answer: Optional[str] = None,
    openai_client: Optional[openai.AsyncOpenAI] = async_openai_client, 
    similarity_encoder_instance: Optional[SentenceTransformer] = similarity_encoder 
) -> Tuple[Optional[float], Optional[float], Optional[torch.Tensor], Optional[str], Optional[float]]: 
    score = 0.0 
    cost = 0.0 
    if isinstance(workflow.controller, torch.nn.DataParallel):
        controller_device = workflow.controller.module.device
    else:
        controller_device = workflow.controller.device
    
    log_prob = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=controller_device) 
    gpt_answer = None
    latency = 0.0

    try:
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.set_device(controller_device)
            torch.cuda.synchronize(controller_device)
        
        try:
            result = await asyncio.wait_for(
                workflow.process_query(query, image_path), 
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Workflow execution timed out (30s) for query: {query[:50]}...")
            return None, None, None, None, None
        
        cost = calculate_cost(result)

        if ground_truth_answer and openai_client:
            prompt = aggregate_info_for_prompt(query, result)
            gpt_answer = await get_gpt_answer([{"role": "system", "content": "You are a helpful assistant answering medical questions based on provided context."}, {"role": "user", "content": prompt}]) 
            logger.info(f"Query: {query[:50]}... | GT: {ground_truth_answer} | GPT: {gpt_answer}") 

            if gpt_answer:
                score = llm_as_judge(ground_truth_answer, gpt_answer)
                chosen_architecture = result.get("architecture", [])
                used_operators = {op for layer in chosen_architecture for op in layer}
                if "LlaVAMed" in used_operators and score > 0.5:
                    bonus = 0.05
                    original_score = score
                    score = min(1.0, score + bonus)
                    logger.info(f"LlaVAMed was used effectively, score boosted from {original_score:.2f} to {score:.2f}")
            else:
                logger.warning("Failed to get an answer from GPT, score is 0.")
                score = 0.0
        else:
            score = 0.0 
            logger.info(f"Query: {query[:50]}... | GT: {ground_truth_answer} | Score: {score} (GPT scoring skipped)")
        
        latency = time.time() - start_time

        log_probs_list = result.get("log_probs", [])
        if not log_probs_list:
            log_prob = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=controller_device)
        else:
             if all(isinstance(lp, torch.Tensor) for lp in log_probs_list):
                  try:
                      log_probs_on_device = [lp.to(controller_device) for lp in log_probs_list]
                      log_prob = torch.stack(log_probs_on_device).sum().detach().clone().requires_grad_(True)
                  except RuntimeError as e:
                      logger.error(f"Failed to stack log_probs: {log_probs_list} - Error: {e}")
                      return None, None, None, None, None
             else:
                  logger.warning(f"log_probs list contains non-Tensor elements: {log_probs_list}")
                  return None, None, None, None, None
        
        return score, cost, log_prob, gpt_answer, latency

    except Exception as e:
        logger.error(f"Workflow execution or feedback calculation failed for query '{query}': {e}", exc_info=True)
        return None, None, None, None, None

def parse_tool_sequence(tool_string: str, operator_names: List[str]) -> List[str]:
    lines = tool_string.strip().splitlines()
    sequence = []
    TOOL_TO_OPERATOR_MAP = {
        "ChestXRayClassify": "ChestXRayClassify",
        "ChestXRaySegment": "ChestXRaySegment",
        "ChestXRayReport": "ChestXRayReport",
        "VQAnalyze": "VQAnalyze",
        "LlaVAMed": "LlaVAMed",
    }
    for line in lines:
        tool_line = line.strip()
        if '.' in tool_line:
            _, tool_name = tool_line.split('.', 1)
            tool_name = tool_name.strip()
        else:
            tool_name = tool_line
        operator = TOOL_TO_OPERATOR_MAP.get(tool_name)
        if operator and operator in operator_names:
            sequence.append(operator)
    return sequence

def supervised_warmup(
    controller: MultiLayerController,
    sentence_encoder: SentenceEncoder,
    train_data: List[Dict],
    optimizer: AgentOptimizer,
    device: str,
    args: argparse.Namespace,
    warmup_size: int = 500,
    warmup_epochs: int = 3 
):
    logger.info(f"Starting supervised warm-up (first {warmup_size} samples, for {warmup_epochs} epochs)...")
    controller.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(warmup_epochs):
        total_loss = 0.0
        count = 0
        
        logger.info(f"Warmup Epoch {epoch+1}/{warmup_epochs}")
        pbar = tqdm(train_data[:warmup_size], desc=f"Warmup Epoch {epoch+1}/{warmup_epochs}")
        for i, item in enumerate(pbar):
            query = item["query"]
            tool_sequence_str = item["tools"]
            operator_sequence = parse_tool_sequence(tool_sequence_str, optimizer.operator_names)
            
            if not operator_sequence:
                continue

            query_embedding = sentence_encoder.encode(query)
            query_tensor = torch.tensor(query_embedding, dtype=torch.float32).to(device)

            if isinstance(controller, torch.nn.DataParallel):
                actual_controller = controller.module
            else:
                actual_controller = controller
                
            probs_list = actual_controller.predict_operator_sequence(
                query=query,
                operator_embeddings=optimizer.operator_embeddings,
                operator_names=optimizer.operator_names,
                target_sequence=operator_sequence
            )

            loss = 0.0
            valid_steps = 0

            for i, probs in enumerate(probs_list):
                if i >= len(operator_sequence):
                    break
                target_op = operator_sequence[i]
                if target_op not in optimizer.operator_names:
                    continue
                target_idx = optimizer.operator_names.index(target_op)
                target_tensor = torch.tensor([target_idx], dtype=torch.long, device=device)
                
                loss += loss_fn(probs.unsqueeze(0), target_tensor)
                valid_steps += 1

            if valid_steps > 0:
                loss = loss / valid_steps
                optimizer.warmup_optimizer.zero_grad()
                loss.backward()
                optimizer.warmup_optimizer.step()

                total_loss += loss.item()
                count += 1
                pbar.set_postfix({"loss": f"{total_loss/count:.4f}"})

        avg_loss = total_loss / count if count > 0 else 0.0
        if args.use_wandb:
            wandb.log({"warmup/epoch": epoch+1, "warmup/avg_loss": avg_loss})
        logger.info(f"Warmup Epoch {epoch+1}/{warmup_epochs} complete, average loss: {avg_loss:.4f}")
    logger.info(f"Supervised warm-up complete, total epochs: {warmup_epochs}")

async def contrastive_path_ranking_phase(
    controller: MultiLayerController,
    workflow: AgentWorkflow,
    train_data: List[Dict],
    optimizer: AgentOptimizer,
    device: str,
    args: argparse.Namespace
):
    logger.info(f"Total epochs: {args.cpr_epochs}, Learning rate: {args.cpr_lr}")
    
    cpr_optimizer = torch.optim.Adam(controller.parameters(), lr=args.cpr_lr)
    controller.train()

    actual_controller = controller.module if isinstance(controller, torch.nn.DataParallel) else controller

    for epoch in range(args.cpr_epochs):
        total_loss = 0.0
        processed_pairs = 0
        
        pbar = tqdm(train_data, desc=f"CPR Epoch {epoch+1}/{args.cpr_epochs}")
        for item in pbar:
            query = item["query"]
            image_path = item["image_path"]
            ground_truth_answer = item["answer"]
            tool_sequence_str = item["tools"]

            positive_path = parse_tool_sequence(tool_sequence_str, optimizer.operator_names)
            if len(positive_path) < 2:
                continue

            negative_path = positive_path.copy()
            random.shuffle(negative_path)
            if negative_path == positive_path:
                continue

            try:
                positive_result = await workflow.execute_fixed_path(query, image_path, positive_path)
                negative_result = await workflow.execute_fixed_path(query, image_path, negative_path)
            except Exception as e:
                logger.warning(f"Error during fixed path execution: {e}, skipping sample.")
                continue

            pos_prompt = aggregate_info_for_prompt(query, positive_result)
            neg_prompt = aggregate_info_for_prompt(query, negative_result)
            
            pos_gpt_answer = await get_gpt_answer([{"role": "user", "content": pos_prompt}])
            neg_gpt_answer = await get_gpt_answer([{"role": "user", "content": neg_prompt}])

            if not pos_gpt_answer or not neg_gpt_answer:
                continue

            positive_score = llm_as_judge(ground_truth_answer, pos_gpt_answer)
            negative_score = llm_as_judge(ground_truth_answer, neg_gpt_answer)

            log_prob_pos = actual_controller.get_path_log_prob(query, workflow.operator_embeddings, workflow.operator_names, positive_path)
            log_prob_neg = actual_controller.get_path_log_prob(query, workflow.operator_embeddings, workflow.operator_names, negative_path)

            if positive_score > negative_score:
                loss = -torch.log(torch.sigmoid(log_prob_pos - log_prob_neg) + 1e-9)
                
                cpr_optimizer.zero_grad()
                loss.backward()
                cpr_optimizer.step()
                
                total_loss += loss.item()
                processed_pairs += 1
                pbar.set_postfix({"loss": f"{total_loss/processed_pairs:.4f}"})

        avg_loss = total_loss / processed_pairs if processed_pairs > 0 else 0.0
        logger.info(f"CPR Epoch {epoch+1} complete, average loss: {avg_loss:.4f}")
        if args.use_wandb:
            wandb.log({"cpr/epoch": epoch+1, "cpr/avg_loss": avg_loss})

    logger.info("Contrastive Path Ranking (Phase II) complete")

async def train_epoch_rl(
    controller: MultiLayerController,
    workflow: AgentWorkflow,
    optimizer: AgentOptimizer,
    train_data: List[Dict],
    batch_size: int,
    device: str,
    max_concurrent_runs: int,
    epoch_num: int,
    args: argparse.Namespace,
    global_step: int
) -> Tuple[float, int]:
    controller.train()
    total_loss = 0.0
    batches_processed = 0
    total_items = len(train_data)
    
    if isinstance(controller, torch.nn.DataParallel):
        max_concurrent_runs = max(1, max_concurrent_runs // 4)
        logger.info(f"DataParallel detected, adjusting concurrent runs to: {max_concurrent_runs}")
    
    semaphore = asyncio.Semaphore(max_concurrent_runs)

    async def process_item(item_data, item_idx, pbar):
        async with semaphore:
            try:
                query = item_data['query']
                image_path = item_data['image_path']
                answer = item_data.get('answer')
                
                max_retries = 2
                for retry in range(max_retries):
                    try:
                        feedback = await run_workflow_and_get_feedback(
                            workflow, query, image_path, answer
                        )
                        if feedback is not None:
                            pbar.update(1)
                            return feedback
                        else:
                            if retry < max_retries - 1:
                                await asyncio.sleep(1)
                            else:
                                pbar.update(1)
                                return None
                    except Exception as e:
                        logger.error(f"Retry {retry} failed for item {item_idx}: {e}")
                        if retry < max_retries - 1:
                            await asyncio.sleep(1)
                        else:
                            pbar.update(1)
                            return None
                
                pbar.update(1)
                return None
            except Exception as e:
                logger.error(f"Outer exception while processing item {item_idx}: {e}", exc_info=True)
                pbar.update(1)
                return None

    tasks = []
    with tqdm(total=total_items, desc=f"Epoch {epoch_num} Training RL") as pbar:
        for idx, item in enumerate(train_data):
            tasks.append(process_item(item, idx, pbar))

        processed_items_count = 0
        for i in range(0, total_items, batch_size):
            batch_tasks = tasks[i:min(i + batch_size, total_items)]
            batch_results_raw = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            valid_results = []
            for res in batch_results_raw:
                if isinstance(res, Exception):
                    logger.error(f"Async task failed with exception: {res}")
                elif res is not None:
                    valid_results.append(res)
            
            processed_items_count += len(batch_results_raw)
            
            if valid_results:
                scores_batch = [s for s, c, lp, ga, lat in valid_results if s is not None]
                costs_batch = [c for s, c, lp, ga, lat in valid_results if c is not None]
                log_probs_batch = [lp for s, c, lp, ga, lat in valid_results if lp is not None] 
                
                if not all(isinstance(lp, torch.Tensor) for lp in log_probs_batch):
                     logger.warning(f"Batch {i // batch_size} contains invalid log_prob types, skipping update.")
                     loss = None
                elif log_probs_batch:
                    loss = optimizer.update(log_probs_batch, scores_batch, costs_batch) 
                else:
                    logger.warning(f"Batch {i // batch_size} has no valid log_probs, skipping optimizer update.")
                    loss = None

                if loss is not None:
                    total_loss += loss
                    batches_processed += 1
                    current_avg_loss = total_loss / batches_processed
                    pbar.set_postfix({"Batch Loss": f"{loss:.4f}", "Avg Loss": f"{current_avg_loss:.4f}"})

                    batch_avg_score = sum(scores_batch) / len(scores_batch) if scores_batch else 0.0
                    batch_avg_cost = sum(costs_batch) / len(costs_batch) if costs_batch else 0.0
                    batch_avg_utility = batch_avg_score - args.cost_weight * batch_avg_cost

                    global_step += 1
                    if args.use_wandb and args.log_every_n_batches > 0 and batches_processed % args.log_every_n_batches == 0:
                        wandb.log({
                            "global_step": global_step,
                            "train/batch_loss": loss,
                            "train/batch_avg_score": batch_avg_score,
                            "train/batch_avg_cost": batch_avg_cost,
                            "train/batch_avg_utility": batch_avg_utility,
                        })
                else:
                    logger.warning(f"Optimizer update failed for batch {i // batch_size}.")
            else:
                logger.warning(f"No valid results for batch {i // batch_size}, skipping update.")
            
            pbar.set_description(f"Epoch {epoch_num} Training RL ({processed_items_count}/{total_items})")
                
    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    logger.info(f"Training epoch {epoch_num} complete, average loss: {avg_loss:.4f}")

    if args.use_wandb and batches_processed > 0:
        try:
            wandb.log({"epoch": epoch_num, "train/epoch_avg_loss": avg_loss})
        except Exception as e:
            logger.warning(f"Failed to log training metrics to wandb: {e}")

    return avg_loss, global_step

def initialize_tools(model_dir, temp_dir, device, gpu_ids=None, distributed_tools=False, args=None) -> Dict[str, Any]:
    if args is None:
        class DefaultArgs:
            offload_large_tools = False
            large_tool_threshold = 4.0
        args = DefaultArgs()
    
    saved_proxies = disable_proxy()
    
    tools_dict = {}
    tool_classes = {
        "ChestXRayClassifierTool": ChestXRayClassifierTool,
        "ChestXRaySegmentationTool": ChestXRaySegmentationTool,
        "ChestXRayReportGeneratorTool": ChestXRayReportGeneratorTool,
        "XRayVQATool": XRayVQATool,
        "LlavaMedTool":LlavaMedTool,
        "ImageVisualizerTool": ImageVisualizerTool 
    }
    
    tool_complexity = {
        "ChestXRayReportGeneratorTool": {"complexity": 10, "memory_gb": 8.0, "priority": "high"},
        "LlavaMedTool": {"complexity": 9, "memory_gb": 6.0, "priority": "high"},
        "XRayVQATool": {"complexity": 8, "memory_gb": 4.0, "priority": "medium"},
        "ChestXRaySegmentationTool": {"complexity": 6, "memory_gb": 3.0, "priority": "medium"},
        "ChestXRayClassifierTool": {"complexity": 4, "memory_gb": 2.0, "priority": "low"},
        "ImageVisualizerTool": {"complexity": 1, "memory_gb": 0.1, "priority": "cpu"},
    }
    
    if distributed_tools and gpu_ids and len(gpu_ids) > 1:
        logger.info(f"Distributing tools across multiple GPUs: {gpu_ids}")
        gpu_assignment = {}
        gpu_load = {gpu_id: 0.0 for gpu_id in gpu_ids}
        tool_names = list(tool_classes.keys())
        sorted_tools = sorted(tool_names, 
                            key=lambda x: (tool_complexity.get(x, {"complexity": 5})["complexity"], 
                                         tool_complexity.get(x, {"memory_gb": 2.0})["memory_gb"]), 
                            reverse=True)
        
        for tool_name in sorted_tools:
            tool_info = tool_complexity.get(tool_name, {"complexity": 5, "memory_gb": 2.0, "priority": "medium"})
            if tool_info["priority"] == "cpu" or tool_name == "ImageVisualizerTool":
                gpu_assignment[tool_name] = None
                logger.info(f"  {tool_name}: CPU")
            else:
                best_gpu = min(gpu_ids, key=lambda x: gpu_load[x])
                gpu_assignment[tool_name] = best_gpu
                gpu_load[best_gpu] += tool_info["memory_gb"]
                logger.info(f"  {tool_name}: GPU {best_gpu} (VRAM≈{tool_info['memory_gb']:.1f}GB)")
        
    elif args.offload_large_tools and gpu_ids and len(gpu_ids) > 1:
        logger.info(f"Using hybrid mode: Offloading large tools to other GPUs")
        gpu_assignment = {}
        available_gpus = gpu_ids[1:]
        gpu_index = 0
        
        for tool_name in tool_classes.keys():
            tool_info = tool_complexity.get(tool_name, {"complexity": 5, "memory_gb": 2.0, "priority": "medium"})
            if tool_info["priority"] == "cpu":
                gpu_assignment[tool_name] = None
            elif tool_info["memory_gb"] >= args.large_tool_threshold:
                if available_gpus:
                    assigned_gpu = available_gpus[gpu_index % len(available_gpus)]
                    gpu_assignment[tool_name] = assigned_gpu
                    gpu_index += 1
                    logger.info(f"  {tool_name}: GPU {assigned_gpu} (Large, VRAM≈{tool_info['memory_gb']:.1f}GB)")
                else:
                    gpu_assignment[tool_name] = device
            else:
                gpu_assignment[tool_name] = device
        
    else:
        gpu_assignment = {name: device for name in tool_classes.keys()}
        logger.info(f"All tools will use the same device: {device}")
    
    init_start_time = time.time()
    for name, tool_class in tool_classes.items():
        try:
            tool_device = gpu_assignment.get(name, device)
            if tool_device is not None and isinstance(tool_device, int):
                tool_device = torch.device(f"cuda:{tool_device}")
            elif tool_device is None:
                tool_device = torch.device("cpu")
            
            tool_info = tool_complexity.get(name, {"memory_gb": 2.0})
            if tool_info["memory_gb"] >= 4.0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
            tool_init_start = time.time()
            kwargs = {"device": tool_device} if name != "ImageVisualizerTool" else {}
            
            if name in ["ChestXRayReportGeneratorTool", "XRayVQATool", "LlavaMedTool"]:
                kwargs["cache_dir"] = model_dir
            
            max_retries = 2
            for retry in range(max_retries):
                try:
                    if name != "ImageVisualizerTool":    
                        tools_dict[name] = tool_class(**kwargs)
                    else:
                        tools_dict[name] = tool_class()
                    break
                except torch.cuda.OutOfMemoryError as oom_error:
                    logger.error(f"CUDA OOM while initializing {name}: {oom_error}")
                    if retry < max_retries - 1:
                        torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(2)
                    else:
                        raise
                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"Retry {retry+1} for {name}: {e}")
                        time.sleep(1)
                    else:
                        raise
               
            tool_init_time = time.time() - tool_init_start
            actual_device = tools_dict[name].device if hasattr(tools_dict[name], 'device') else tool_device
            logger.info(f"Tool {name} initialized (Device: {actual_device}, Time: {tool_init_time:.2f}s)")
            
        except Exception as e:
             logger.error(f"Failed to initialize tool {name}: {e}", exc_info=True)
             if name not in ["ChestXRayReportGeneratorTool", "ChestXRayClassifierTool"]:
                 logger.warning(f"Skipping non-critical tool {name}")
                 continue
    
    total_init_time = time.time() - init_start_time
    logger.info(f"Successfully initialized {len(tools_dict)}/{len(tool_classes)} tools (Total: {total_init_time:.2f}s)")
    
    logger.info("Restoring proxy settings...")
    restore_proxy(saved_proxies)
    
    return tools_dict

async def main():
    args = setup_args()
    
    if args.use_wandb:
        try:
            run_name = args.wandb_run_name
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
                tags=["training"]
            )
            logger.info(f"Weights & Biases logging enabled for project '{args.wandb_project}'")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            args.use_wandb = False

    device, gpu_ids, use_multi_gpu = setup_multi_gpu(args)
    optimize_gpu_memory()
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory_total = gpu_props.total_memory / 1024**3
            logger.info(f"GPU {i} ({gpu_props.name}): Total VRAM={gpu_memory_total:.1f}GB")
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        
    logger.info(f"Using device: {device}")
    if use_multi_gpu:
        logger.info(f"Multi-GPU setup: {gpu_ids}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Loading data from {args.data_root}...")
    train_data = load_slake_dataset(args.data_root, args.train_split)
    val_data = load_slake_dataset(args.data_root, args.val_split)
    
    if not train_data:
        logger.error("Failed to load training data. Exiting.")
        return
        
    try:
        logger.info(f"Initializing sentence encoder: {args.encoder_model}")
        sentence_encoder = SentenceEncoder(model_name=args.encoder_model, device=device)
        _dummy_emb = sentence_encoder.encode("test")
        input_dim = _dummy_emb.shape[0]
        logger.info(f"Sentence encoder input dimension: {input_dim}")
    except Exception as e:
        logger.error(f"Failed to initialize sentence encoder: {e}")
        input_dim = 384

    global similarity_encoder
    try:
        logger.info(f"Initializing semantic similarity encoder: {similarity_model_name}")
        similarity_encoder = SentenceTransformer(similarity_model_name, device=device)
    except Exception as e:
        logger.error(f"Failed to load semantic similarity model: {e}")
        similarity_encoder = None

    logger.info("Initializing controller...")
    controller = MultiLayerController(
        input_dim=input_dim,
        hidden_dim=args.controller_hidden_dim,
        num_layers=args.controller_num_layers,
        device=device
    ).to(device)

    controller, actual_device = create_parallel_controller(controller, gpu_ids, use_multi_gpu)
    device = actual_device

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
        tools_dict=tools_dict, controller=controller,
        log_tools=args.log_tools, log_dir=args.log_dir,
        device=device
    )
    if not hasattr(workflow, 'operator_embeddings') or not hasattr(workflow, 'operator_names'):
        logger.error("AgentWorkflow initialization failed.")
        return
    logger.info(f"Workflow initialized, available operators: {workflow.operator_names}")

    logger.info("Initializing optimizer...")
    optimizer = AgentOptimizer(
        controller=controller, operator_embeddings=workflow.operator_embeddings, 
        operator_names=workflow.operator_names, learning_rate=args.learning_rate,
        batch_size=args.batch_size, log_dir=str(output_dir), 
        cost_weight=args.cost_weight, device=device
    )
    
    if use_multi_gpu:
        if isinstance(controller, torch.nn.DataParallel):
            adjusted_concurrent_runs = max(1, args.max_concurrent_runs)
        else:
            adjusted_concurrent_runs = args.max_concurrent_runs * len(gpu_ids)
        args.max_concurrent_runs = adjusted_concurrent_runs
    
    if args.checkpoint_to_load:
        checkpoint_path = Path(args.checkpoint_to_load)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            try:
                optimizer.load_checkpoint(str(checkpoint_path))
                logger.info("Checkpoint loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return
        else:
            logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
            return
    
    if not args.disable_warmup:
        supervised_warmup(
            controller=controller, sentence_encoder=sentence_encoder,
            train_data=train_data, optimizer=optimizer,
            device=device, args=args, warmup_size=args.warmup_size,
            warmup_epochs=args.warmup_epochs
        )
    else:
        logger.info("Warm-up phase skipped as per --disable_warmup flag.")
    
    await contrastive_path_ranking_phase(
        controller=controller, workflow=workflow,
        train_data=train_data, optimizer=optimizer,
        device=device, args=args
    )
    
    best_val_score = -float('inf')
    logger.info(f"Starting training for {args.epochs} epochs...")
    global_step = 0

    for epoch in range(args.epochs):
        epoch_num = epoch + 1
        logger.info(f"Starting training epoch {epoch_num}/{args.epochs}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        try:
            avg_train_loss, global_step = await train_epoch_rl(
                controller, workflow, optimizer, train_data, args.batch_size, device, 
                args.max_concurrent_runs, epoch_num, args=args, global_step=global_step
            )
            logger.info(f"Epoch {epoch_num} training complete, average loss: {avg_train_loss:.4f}")
        except Exception as train_error:
            logger.error(f"Training epoch {epoch_num} failed: {train_error}", exc_info=True)
            continue
        
        if (epoch_num % args.save_every_n_epochs == 0):
            checkpoint_path = optimizer.save_checkpoint(name=f"controller_epoch_{epoch_num}")
            if checkpoint_path:
                logger.info(f"Checkpoint saved: {checkpoint_path}")

    logger.info("All training epochs completed!")

    if args.use_wandb and wandb.run is not None:
        wandb.finish()
        logger.info("Weights & Biases run finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.critical(f"An uncaught error occurred during training: {e}", exc_info=True)
