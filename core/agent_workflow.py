from typing import List, Dict, Any, Optional, Tuple
import torch
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import json
import os
import time
import numpy as np

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage

from .agent_controller import MultiLayerController, SentenceEncoder
from .agent_operators import build_operator_mapping, get_operator_descriptions
from .agent_utils import get_operator_embeddings

logger = logging.getLogger(__name__)

# A custom JSON encoder for special types like torch.Tensor and numpy types.
class CustomJSONEncoder(json.JSONEncoder):
    """A custom JSON encoder for special types like torch.Tensor and numpy types."""
    def default(self, obj):
        # Handle PyTorch Tensors.
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().tolist()
        # Handle NumPy arrays.
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle NumPy scalars - compatible with NumPy 2.0.
        elif isinstance(obj, (np.integer)):  # Replaces specific types like np.int_.
            return int(obj)
        elif isinstance(obj, (np.floating)):  # Replaces np.float_ etc.
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        # Handle Path objects.
        elif isinstance(obj, Path):
            return str(obj)
        # Let the parent class handle other types or raise a TypeError.
        return super(CustomJSONEncoder, self).default(obj)

class AgentWorkflow:
    """Agent workflow execution engine, supports multi-layer operator execution."""
    
    def __init__(
        self, 
        tools_dict: Dict[str, Any], 
        controller: Optional[MultiLayerController] = None,
        log_tools: bool = True,
        log_dir: str = "logs/agent_executions",
        device=None
    ):
        """
        Initializes the Agent workflow execution engine.
        
        Args:
            tools_dict: A dictionary of tools, where keys are tool names and values are tool instances.
            controller: The controller network; a new one will be created if not provided.
            log_tools: Whether to log tool executions.
            log_dir: The log directory.
            device: The computation device.
        """
        self.tools_dict = tools_dict
        self.log_tools = log_tools
        self.log_dir = log_dir
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure the log directory exists.
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize or load the controller.
        if controller is None:
            logger.info("Creating a new multi-layer controller.")
            self.controller = MultiLayerController(device=self.device)
        else:
            logger.info("Using the provided controller.")
            self.controller = controller
            
        # Build the operator mapping.
        self.operator_mapping = build_operator_mapping(tools_dict)
        logger.info(f"Loaded {len(self.operator_mapping)} operators.")
        
        # Pre-compute operator embeddings.
        self.operator_names = list(self.operator_mapping.keys())
        self.operator_descriptions = get_operator_descriptions()
        
        # Only include descriptions for operators present in the operator_mapping.
        filtered_descriptions = []
        for name in self.operator_names:
            if name in self.operator_descriptions:
                filtered_descriptions.append(self.operator_descriptions[name])
            else:
                logger.warning(f"Could not find a description for operator {name}.")
                filtered_descriptions.append(f"Functional description for operator {name}.")
                
        self.operator_embeddings = get_operator_embeddings(filtered_descriptions)
        self.operator_embeddings = self.operator_embeddings.to(self.device)
        
        logger.info(f"Available operators: {', '.join(self.operator_names)}")
        

    async def process_query(self, query: str, image_path: str) -> dict:
        """
        Processes a query and returns the result.
        
        Args:
            query: The user query.
            image_path: The image path.
            
        Returns:
            dict: A dictionary containing the complete execution result.
        """
        logger.info(f"🎯 AgentWorkflow starting to process query: {query[:50]}...")
        logger.info(f"📁 Image path: {image_path}")
        
        if hasattr(self, 'controller') and self.controller:
            device_info = f"DataParallel - Main device: {self.controller.module.device}" if isinstance(self.controller, torch.nn.DataParallel) else f"Single device: {self.controller.device}"
            logger.info(f"🖥️ Controller device status: {device_info}")
        
        try:
            logger.info("🔄 Starting workflow execution...")
            start_time = time.time()
            
            controller_instance = self.controller.module if isinstance(self.controller, torch.nn.DataParallel) else self.controller
            log_probs, selected_ops_layers = controller_instance.forward(
                query, self.operator_embeddings, self.operator_names
            )
            
            if not selected_ops_layers or all(not layer for layer in selected_ops_layers):
                logger.warning("Architecture generated by the controller is empty or invalid, using default architecture.")
                selected_ops_layers = [["ChestXRayReport"], ["EarlyStop"]]
            
            logger.info(f"Architecture selected: {selected_ops_layers}")
            
            temp_data = {str(i): {'operator': None, 'text': None, 'image_path': None} for i in range(len(selected_ops_layers))}
            intermediate_results = []
            layer_results_list = []
            final_result = None
            status = "success"
            error_message = None
            current_query = query
            current_image_path = image_path
            
            for layer_idx, layer_ops in enumerate(selected_ops_layers):
                logger.info(f"Executing layer {layer_idx}: {layer_ops}")
                layer_results = []
                early_stop_encountered = False

                for op_name in layer_ops:
                    if op_name == "EarlyStop":
                        logger.info("Executing EarlyStop operator, terminating workflow.")
                        status = "early_stopped"
                        early_stop_encountered = True
                        break
                        
                    if op_name in self.operator_mapping:
                        operator = self.operator_mapping[op_name]
                        
                        try:
                            logger.info(f"🔄 Starting execution of operator: {op_name} (Layer {layer_idx})")
                            
                            input_params = {'query': current_query}

                            if hasattr(operator, 'invoke') and callable(operator.invoke):
                                logger.debug(f"Invoking operator {op_name} using the invoke method.")
                                
                                if op_name == "VQAnalyze":
                                    vqa_image_paths = [p for p in [image_path] + [temp_data[str(i)].get('image_path') for i in range(layer_idx)] if p]
                                    logger.info(f"VQAnalyze received image paths: {vqa_image_paths}")
                                    input_params['image_path'] = vqa_image_paths
                                else:
                                    input_params['image_path'] = current_image_path

                                result = await operator.invoke(input_params, intermediate_results)

                            else:
                                logger.debug(f"Calling operator {op_name} using the __call__ method.")
                                call_params = {'query': query, 'image_path': image_path, 'context': intermediate_results}
                                input_params.update(call_params)
                                result = await operator(**call_params)
                            
                            if result:
                                result = {"response": result, "status": "success"} if not isinstance(result, dict) else result
                                
                                result_with_metadata = {
                                    "operator": op_name,
                                    "input_params": input_params,
                                    "result": result,
                                    "layer": layer_idx
                                }
                                layer_results.append(result_with_metadata)
                                intermediate_results.append(result_with_metadata)
                                
                                output_image_path = None
                                if op_name == "ChestXRaySegment":
                                    try:
                                        output_image_path = result.get('response', {}).get('visualization_path')
                                    except:
                                        output_image_path = None
                                
                                temp_data[str(layer_idx)] = {
                                    'operator': op_name,
                                    'text': str(result.get('response', '')) if isinstance(result, dict) else str(result),
                                    'image_path': output_image_path
                                }
                                final_result = result
                                
                        except Exception as e:
                            logger.error(f"Operator {op_name} execution failed: {str(e)}", exc_info=True)
                            error_result = {"operator": op_name, "input_params": input_params, "result": {"response": str(e), "status": "error"}, "layer": layer_idx}
                            layer_results.append(error_result)
                            intermediate_results.append(error_result)
                    else:
                        logger.warning(f"Operator not found: {op_name}")
                
                layer_results_list.append(layer_results)
                
                if early_stop_encountered: break
                
                if layer_idx < len(selected_ops_layers) - 1:
                    current_layer_text = temp_data[str(layer_idx)]['text']
                    current_query = f"{current_layer_text} I want to ask:{query}" if current_layer_text else query

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}", exc_info=True)
            status = "error"
            error_message = f"Workflow execution failed: {str(e)}"
            
        duration = time.time() - start_time
        
        if status not in ["error", "early_stopped"]:
            any_failure = any(res.get("result", {}).get("status") == "error" for layer in layer_results_list for res in layer)
            if any_failure: status = "partial_success"
        
        result = {
            "query": query,
            "image_path": image_path,
            "architecture": selected_ops_layers,
            "layer_results": layer_results_list,
            "final_result": final_result,
            "duration": duration,
            "status": status,
            "error": error_message,
            "log_probs": log_probs,
            "temp_data": temp_data
        }
        
        self._save_execution_log(result)
        
        return result
        
    def _save_execution_log(self, result: Dict[str, Any]) -> None:
        """Saves the execution log to a single JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(self.log_dir, f"execution_log_{date_str}.json")
        
        try:
            result["timestamp"] = timestamp
            existing_logs = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        existing_logs = json.load(f)
                    if not isinstance(existing_logs, list):
                        existing_logs = [existing_logs]
                except (json.JSONDecodeError, IOError):
                    logger.warning(f"Could not parse existing log file {log_file}, a new log will be created.")
                    existing_logs = []
            
            existing_logs.append(result)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
            
            logger.info(f"Execution log appended to {log_file}")
        except Exception as e:
            logger.error(f"Failed to save execution log: {str(e)}", exc_info=True)
            
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('model', None)
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)


    async def execute_fixed_path(self, query: str, image_path: str, fixed_path: List[str]) -> dict:
        """
        Executes a predefined, fixed tool path.
        """
        logger.info(f"🦾 Executing fixed path: {fixed_path}")
        start_time = time.time()
        
        # Convert the path to a layered structure
        selected_ops_layers = [[op] for op in fixed_path]
        
        # Most of the following logic is the same as in process_query
        temp_data = {str(i): {'operator': None, 'text': None, 'image_path': None} for i in range(len(selected_ops_layers))}
        intermediate_results = []
        layer_results_list = []
        final_result = None
        status = "success"
        error_message = None
        current_query = query
        current_image_path = image_path

        for layer_idx, layer_ops in enumerate(selected_ops_layers):
            layer_results = []
            for op_name in layer_ops:
                if op_name in self.operator_mapping:
                    operator = self.operator_mapping[op_name]
                    input_params = {'query': current_query, 'image_path': current_image_path}
                    
                    result = await operator.invoke(input_params, intermediate_results)
                    
                    if result:
                        result_with_metadata = {
                            "operator": op_name, "input_params": input_params,
                            "result": result, "layer": layer_idx
                        }
                        layer_results.append(result_with_metadata)
                        intermediate_results.append(result_with_metadata)
                        final_result = result
                else:
                    logger.warning(f"Operator not found in fixed path: {op_name}")
            layer_results_list.append(layer_results)

        duration = time.time() - start_time
        
        # Return a result dictionary with a structure similar to process_query
        return {
            "query": query, "image_path": image_path,
            "architecture": selected_ops_layers,
            "layer_results": layer_results_list,
            "final_result": final_result,
            "duration": duration, "status": status, "error": error_message,
        }
