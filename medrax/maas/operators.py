from typing import List, Dict, Any, Optional, Tuple
import torch
from langchain_core.tools import BaseTool
import os
import json
import logging
import asyncio
import concurrent.futures

logger = logging.getLogger(__name__)

# Base Operator Class
class Operator:
    def __init__(self, tool: BaseTool, name: str):
        self.name = name
        self.tool = tool
        
    async def __call__(self, *args, **kwargs):
        """Executes the tool call."""
        try:
            # Ensure CUDA context consistency in a multi-GPU environment
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                torch.cuda.synchronize(current_device)
            
            # Check if the tool supports async execution and has a callable _arun method
            if hasattr(self.tool, '_arun') and callable(self.tool._arun):
                logger.debug(f"Executing tool {self.name} using async method.")
                result = await self.tool._arun(*args, **kwargs)
            else:
                # Sync execution, wrapped in a thread pool instead of asyncio.to_thread
                logger.debug(f"Executing tool {self.name} using sync method.")
                if not hasattr(self.tool, '_run') or not callable(self.tool._run):
                    raise ValueError(f"Tool {self.name} has neither a callable _arun nor a callable _run method.")
                
                # Use concurrent.futures to avoid blocking issues in some environments
                loop = asyncio.get_event_loop()
                
                # Create a new thread pool executor to avoid issues with shared thread pools
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = loop.run_in_executor(executor, self.tool._run, *args, **kwargs)
                    # Add timeout control
                    try:
                        result = await asyncio.wait_for(future, timeout=120.0)
                    except asyncio.TimeoutError:
                        logger.error(f"Tool {self.name} execution timed out.")
                        return {"response": "Tool execution timed out", "status": "timeout"}
                
            return {"response": result, "status": "success"}
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {str(e)}", exc_info=True)
            return {"response": f"Error: {str(e)}", "status": "error"}
    
    async def invoke(self, params: Dict[str, Any], context: List[Dict] = None) -> Dict[str, Any]:
        """
        Unified operator invocation interface (corrected).
        
        Args:
            params: A dictionary containing all call parameters (e.g., query, image_path).
            context: Context information (results from previous operators).
            
        Returns:
            Dict: The result of the operator execution.
        """
        try:
            # Ensure CUDA context consistency in a multi-GPU environment
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                torch.cuda.synchronize(current_device)
            
            # Directly unpack the parameter dictionary and pass it to the underlying __call__ method.
            # Subclasses (e.g., ChestXRayClassify) will automatically receive the parameters they need
            # and ignore those they don't need (due to **kwargs).
            result = await self.__call__(**params)
            return result
        except Exception as e:
            logger.error(f"Operator {self.name} invocation failed: {str(e)}", exc_info=True)
            return {"response": f"Operator invocation failed: {str(e)}", "status": "error"}

# Specific Medical Operator Implementations
class ChestXRayClassify(Operator):
    """Chest X-ray Classification Operator."""
    def __init__(self, tool: BaseTool, name: str = "ChestXRayClassify"):
        super().__init__(tool, name)
        
    async def __call__(self, image_path, **kwargs):
        try:
            # Determine whether to use sync or async method.
            is_async = hasattr(self.tool, '_arun') and callable(self.tool._arun)
            
            # Get classification results.
            if is_async:
                result = await self.tool._arun(image_path=image_path)
            else:
                # Use concurrent.futures instead of asyncio.to_thread.
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = loop.run_in_executor(executor, self.tool._run, image_path)
                    try:
                        result = await asyncio.wait_for(future, timeout=120.0)
                    except asyncio.TimeoutError:
                        logger.error(f"Classification operation timed out.")
                        return {"response": "Operation timed out", "status": "timeout"}
            
            # Process results, extract the most likely pathologies.
            pathologies = {}
            if isinstance(result, tuple) and len(result) > 0:
                pathologies = result[0]
            
            # Find the top 5 pathologies with the highest probabilities.
            top_pathologies = sorted(pathologies.items(), key=lambda x: x[1], reverse=True)[:5]
            formatted_result = {name: float(prob) for name, prob in top_pathologies}
            
            return {"response": formatted_result, "status": "success", "raw_result": result}
        except Exception as e:
            logger.error(f"Classification operation failed: {str(e)}", exc_info=True)
            return {"response": f"Error: {str(e)}", "status": "error"}

class ChestXRaySegment(Operator):
    """Chest X-ray Segmentation Operator."""
    def __init__(self, tool: BaseTool, name: str = "ChestXRaySegment"):
        super().__init__(tool, name)
        
    async def __call__(self, image_path, organs=None, **kwargs):
        try:
            # Determine whether to use sync or async method.
            is_async = hasattr(self.tool, '_arun') and callable(self.tool._arun)
            
            # Get segmentation results.
            if is_async:
                # Use async method.
                if organs:
                    raw_result = await self.tool._arun(image_path=image_path, organs=organs)
                else:
                    raw_result = await self.tool._arun(image_path=image_path)
            else:
                # Use concurrent.futures instead of asyncio.to_thread.
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    if organs:
                            future = loop.run_in_executor(executor, self.tool._run, image_path, organs)
                    else:
                            future = loop.run_in_executor(executor, self.tool._run, image_path)
                    try:
                        raw_result = await asyncio.wait_for(future, timeout=120.0)
                    except asyncio.TimeoutError:
                        logger.error(f"Segmentation operation timed out.")
                        return {"response": "Operation timed out", "status": "timeout"}
            
            # Ensure raw_result is a tuple.
            if not isinstance(raw_result, tuple):
                logger.warning("Result returned by the segmentation tool is not in tuple format.")
                return {
                    "response": {"error": "Incorrect segmentation result format"},
                    "status": "error"
                }

            # Extract key information from the segmentation result.
            segmentation_info = {}
            
            # Safely extract the visualization path.
            if len(raw_result) > 1 and isinstance(raw_result[1], dict):
                segmentation_info["visualization_path"] = raw_result[1].get("image_path", "")
            
            # Safely extract organ metrics.
            organ_metrics = {}
            if len(raw_result) > 0 and isinstance(raw_result[0], dict) and "metrics" in raw_result[0]:
                metrics_data = raw_result[0]["metrics"]
                if isinstance(metrics_data, dict):
                    for organ, metrics in metrics_data.items():
                        if metrics and isinstance(metrics, dict):
                            organ_metrics[organ] = {
                                "area_cm2": metrics.get("area_cm2", 0),
                                "position": metrics.get("relative_position", {})
                            }
            
            segmentation_info["organ_metrics"] = organ_metrics
            
            return {
                "response": segmentation_info,
                "status": "success",
                "raw_result": raw_result
            }
        except Exception as e:
            logger.error(f"Segmentation operation failed: {str(e)}", exc_info=True)
            # Return a structured error response.
            return {
                "response": {"error": str(e)},
                "status": "error"
            }

class ChestXRayReport(Operator):
    """Chest X-ray Report Generation Operator."""
    def __init__(self, tool: BaseTool, name: str = "ChestXRayReport"):
        super().__init__(tool, name)
        
    async def __call__(self, image_path, **kwargs):
        try:
            is_async = hasattr(self.tool, '_arun') and callable(self.tool._arun)
            
            if is_async:
                result = await self.tool._arun(image_path=image_path)
            else:
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = loop.run_in_executor(executor, self.tool._run, image_path)
                    try:
                        result = await asyncio.wait_for(future, timeout=120.0)
                    except asyncio.TimeoutError:
                        logger.error(f"Report generation timed out.")
                        return {"response": "Operation timed out", "status": "timeout"}
            
            # Extract report text.
            report_text = ""
            if isinstance(result, tuple) and len(result) > 0:
                if isinstance(result[0], dict):
                    findings = result[0].get("findings", "")
                    impression = result[0].get("impression", "")
                    report_text = f"Findings: {findings}\n\nImpression: {impression}"
            
            return {
                "response": report_text, 
                "status": "success",
                "raw_result": result
            }
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}", exc_info=True)
            return {"response": f"Error: {str(e)}", "status": "error"}

class VQAnalyze(Operator):
    """Visual Question Answering (VQA) Analysis Operator."""
    def __init__(self, tool: BaseTool, name: str = "VQAnalyze"):
        super().__init__(tool, name)
        
    async def __call__(self, image_path, query, **kwargs):
        try:
            is_async = hasattr(self.tool, '_arun') and callable(self.tool._arun)
            
            if is_async:
                result = await self.tool._arun(image_paths=image_path, prompt=query)
            else:
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = loop.run_in_executor(executor, self.tool._run, image_path, query)
                    try:
                        result = await asyncio.wait_for(future, timeout=120.0)
                    except asyncio.TimeoutError:
                        logger.error(f"VQA execution timed out.")
                        return {"response": "Operation timed out", "status": "timeout"}
            
            # Extract the answer.
            answer = ""
            if isinstance(result, tuple) and len(result) > 0:
                if isinstance(result[0], dict):
                    answer = result[0].get("response", "")
                elif isinstance(result[0], str):
                    answer = result[0]
            
            return {
                "response": answer,
                "status": "success",
                "raw_result": result
            }
        except Exception as e:
            logger.error(f"VQA analysis failed: {str(e)}", exc_info=True)
            return {"response": f"Error: {str(e)}", "status": "error"}

class GroundFindings(Operator):
    """Finding Grounding Operator."""
    def __init__(self, tool: BaseTool, name: str = "GroundFindings"):
        super().__init__(tool, name)
        
    async def __call__(self, image_path, query=None, text=None, **kwargs):
        try:
            # Ensure a text query parameter is present.
            text_query = text or query
            if not text_query:
                return {
                    "response": {"error": "Missing text query parameter"},
                    "status": "error"
                }
                
            is_async = hasattr(self.tool, '_arun') and callable(self.tool._arun)
            
            if is_async:
                result = await self.tool._arun(image_path=image_path, text=text_query)
            else:
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = loop.run_in_executor(executor, self.tool._run, image_path, text_query)
                    try:
                        result = await asyncio.wait_for(future, timeout=120.0)
                    except asyncio.TimeoutError:
                        logger.error(f"Finding grounding timed out.")
                        return {"response": "Operation timed out", "status": "timeout"}
            
            # Extract key information.
            grounding_info = {}
            if isinstance(result, tuple) and len(result) > 0:
                if isinstance(result[0], dict):
                    grounding_info = {
                        "visualization_path": result[0].get("visualization_path", ""),
                        "bounding_boxes": result[0].get("bounding_boxes", []),
                        "confidence": result[0].get("confidence_scores", [])
                    }
            
            return {
                "response": grounding_info,
                "status": "success", 
                "raw_result": result
            }
        except Exception as e:
            logger.error(f"Finding grounding operation failed: {str(e)}", exc_info=True)
            return {"response": f"Error: {str(e)}", "status": "error"}

class LlaVAMed(Operator):
    """LLaVA-Med Visual Question Answering Operator."""
    def __init__(self, tool: BaseTool, name: str = "LlaVAMed"):
        super().__init__(tool, name)
        
    async def __call__(self, image_path, query, **kwargs):
        """Executes the LLaVA-Med tool call."""
        try:
            is_async = hasattr(self.tool, '_arun') and callable(self.tool._arun)
            
            if is_async:
                # Assume the LLaVA-Med tool accepts image_path and question parameters.
                result = await self.tool._arun(image_path=image_path, question=query)
            else:
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = loop.run_in_executor(executor, self.tool._run, image_path, query)
                    try:
                        result = await asyncio.wait_for(future, timeout=120.0)
                    except asyncio.TimeoutError:
                        logger.error(f"LLaVA-Med execution timed out.")
                        return {"response": "Operation timed out", "status": "timeout"}
            
            # Extract the answer - adapt to different possible return formats.
            answer = ""
            if isinstance(result, tuple) and len(result) > 0:
                if isinstance(result[0], dict):
                    # Try to extract 'response' from the dictionary.
                    answer = result[0].get("response", "")
                elif isinstance(result[0], str):
                    # Use the string result directly.
                    answer = result[0]
                else:
                     # Handle other unknown formats, log a warning.
                    logger.warning(f"LLaVA-Med tool returned unexpected format: {type(result[0])}")
                    answer = str(result[0]) # Attempt to convert to string.
            elif isinstance(result, str):
                 # If a string is returned directly.
                 answer = result
            
            return {
                "response": answer,
                "status": "success",
                "raw_result": result
            }
        except Exception as e:
            logger.error(f"LLaVA-Med operation failed: {str(e)}", exc_info=True)
            return {"response": f"Error: {str(e)}", "status": "error"}

class EarlyStop(Operator):
    """Early Stop Operator."""
    def __init__(self, name: str = "EarlyStop"):
        super().__init__(None, name)
        
    async def __call__(self, **kwargs):
        return {"response": "Process completed", "status": "success"}
        
    async def invoke(self, params: Dict[str, Any], context: List[Dict] = None) -> Dict[str, Any]:
        """The invoke method for the EarlyStop operator."""
        return {"response": "Workflow terminated early.", "status": "success"}

# Operator Mapping and Name List
def build_operator_mapping(tools_dict):
    """Builds the mapping from tools to operators."""
    operator_mapping = {}
    
    if 'ChestXRayClassifierTool' in tools_dict:
        operator_mapping['ChestXRayClassify'] = ChestXRayClassify(tools_dict['ChestXRayClassifierTool'])
        
    if 'ChestXRaySegmentationTool' in tools_dict:
        operator_mapping['ChestXRaySegment'] = ChestXRaySegment(tools_dict['ChestXRaySegmentationTool'])
        
    if 'ChestXRayReportGeneratorTool' in tools_dict:
        operator_mapping['ChestXRayReport'] = ChestXRayReport(tools_dict['ChestXRayReportGeneratorTool'])
        
    if 'XRayVQATool' in tools_dict:
        operator_mapping['VQAnalyze'] = VQAnalyze(tools_dict['XRayVQATool'])
        
    if 'XRayPhraseGroundingTool' in tools_dict:
        operator_mapping['GroundFindings'] = GroundFindings(tools_dict['XRayPhraseGroundingTool'])
    
    # Add the LlaVAMed operator.
    if 'LlavaMedTool' in tools_dict:
        operator_mapping['LlaVAMed'] = LlaVAMed(tools_dict['LlavaMedTool'])
    
    # Add the EarlyStop operator.
    # operator_mapping['EarlyStop'] = EarlyStop()
    
    return operator_mapping

def get_operator_descriptions():
    """Get operator descriptions for embedding generation (English version)."""
    return {
        'ChestXRayClassify': "Analyze chest X-rays and identify up to 18 possible pathologies, such as atelectasis, cardiomegaly, etc. Returns the probability of each disease.",
        'ChestXRaySegment': "Segment chest X-rays into different anatomical structures, such as left and right lungs, heart, aorta, etc., and provide the area and location information of each organ.",
        'ChestXRayReport': "Generate a detailed medical report based on the chest X-ray, including findings and impression sections, similar to a radiologist's report.",
        'VQAnalyze': "Answer specific medical questions about chest X-rays, combining visual understanding and medical knowledge. Can answer questions about lesions, structures, or diagnoses.",
        'GroundFindings': "Locate specific medical findings or abnormal regions on the chest X-ray, mapping text-described lesions to specific locations in the image.",
        'EarlyStop': "Complete the current analysis process and do not execute subsequent operators; used when there is already enough information to answer the query.",
        'LlaVAMed': "Use the LLaVA-Med model for general medical image visual question answering, capable of understanding and answering a wide range of questions about the input image."
    }

import tiktoken

try:
    tokenizer_for_cost = tiktoken.get_encoding("cl100k_base")
except Exception:
    # Fallback if loading fails
    tokenizer_for_cost = None
    print("Warning: tiktoken encoder failed to load. Token-based cost will be inaccurate.")

def get_operator_cost_models() -> dict:
    """
    Returns a dictionary containing the cost models for all operators.
    Uses the conversion rate: FLOPs = 2 * model parameters * tokens.

    'type': 'fixed' indicates a fixed computation cost (CU).
    'type': 'tokens' indicates a dynamic cost based on tokens (CU per 1k tokens).
    'type': 'generative_diffusion' indicates a step-based generation cost (CU per step).
    'type': 'hybrid' indicates a hybrid cost.
    """
    # Exchange rate definition:
    # 1 TFLOP = 1 CU
    # 1k tokens on a 7B LLM (baseline) @ 2PT = 2 * 7B * 1k = 14 TFLOPs => 14 CU
    
    cost_models = {
        # --- Fixed-cost tools (FLOPs-based) ---
        'ChestXRayClassify': {
            'type': 'fixed',
            'cost_cu': 0.0076,  # DenseNet-121 ~7.6 GFLOPs.
            'description': 'Based on DenseNet-121 FLOPs'
        },
        'ChestXRaySegment': {
            'type': 'fixed',
            'cost_cu': 0.060,   # PSPNet ~60 GFLOPs.
            'description': 'Based on PSPNet FLOPs'
        },
        'DicomProcessor': { 
            'type': 'fixed',
            'cost_cu': 0.0001, # Very low CPU computation cost
            'description': 'Low CPU/IO cost'
        },

        # --- Dynamic-cost tools (Token-based) ---
        # Based on the 2*P*T formula
        'LlaVAMed': {
            'type': 'tokens',
            'cu_per_1k_tokens': 14.0, # 7B Mistral-based model (baseline).
            'description': 'Based on 2 * 7B model parameters * tokens'
        },
        'VQAnalyze': { # Corresponds to XRayVQATool
            'type': 'tokens',
            'cu_per_1k_tokens': 6.0, # CheXagent 3B model (14 * 3/7)
            'description': 'Based on 2 * 3B model parameters * tokens'
        },
        'GroundFindings': { # Corresponds to XRayPhraseGroundingTool
            'type': 'tokens',
            'cu_per_1k_tokens': 6.0, # MAIRA-2, assumed similar to CheXagent 3B
            'description': 'Based on 2 * ~3B model parameters * tokens'
        },

        # --- Hybrid-cost tools ---
        'ChestXRayReport': {
            'type': 'hybrid',
            'fixed_cost_cu': 0.017, # ViT Encoder part cost is unchanged
            'cu_per_1k_tokens': 2.0, # BERT Decoder part
            'description': 'ViT FLOPs (fixed) + (2 * 1B BERT params * tokens)'
        },
        
        # --- Generative tools (not affected by the formula) ---
        'ChestXRayGenerator': {
            'type': 'generative_diffusion',
            'cu_per_step': 1.5, # Cost per inference step
            'description': 'Based on Stable Diffusion UNet FLOPs per step'
        },
        
        # --- Special operators ---
        'EarlyStop': {
            'type': 'fixed',
            'cost_cu': 0.0, # No computational cost
            'description': 'No computational cost'
        }
    }
    return cost_models

def calculate_dynamic_cost(operator_name: str, cost_models: dict, tool_input: dict, tool_output: dict) -> float:
    """
    Calculates the actual cost (in CU) based on the operator name and runtime parameters.

    Args:
        operator_name: The name of the operator.
        cost_models: The dictionary of cost models from get_operator_cost_models().
        tool_input: The input dictionary for the tool call.
        tool_output: The output dictionary from the tool execution.
    """
    model = cost_models.get(operator_name)
    if not model:
        return 0.1 # default cost

    cost_type = model['type']

    if cost_type == 'fixed':
        return model['cost_cu']

    elif cost_type == 'tokens' or cost_type == 'hybrid':
        # Handle token calculation uniformly.
        input_text = tool_input.get('query', '') or tool_input.get('prompt', '') or tool_input.get('phrase', '')
        
        response_data = tool_output.get('response', '')
        raw_result_data = tool_output.get('raw_result', '')

        # Prefer fetching the more complete text output from raw_result.
        if raw_result_data and isinstance(raw_result_data, (list, tuple)) and len(raw_result_data) > 0 and isinstance(raw_result_data[0], str):
            output_text = raw_result_data[0]
        else: # otherwise use response.
            output_text = str(response_data) if isinstance(response_data, dict) else response_data

        if tokenizer_for_cost:
            input_tok_count = len(tokenizer_for_cost.encode(input_text))
            output_tok_count = len(tokenizer_for_cost.encode(output_text or ""))
            total_tokens = input_tok_count + output_tok_count
        else: # Fallback
            total_tokens = (len(input_text) + len(output_text or "")) / 4.0

        dynamic_cost = (total_tokens / 1000) * model['cu_per_1k_tokens']
        
        if cost_type == 'hybrid':
            return model['fixed_cost_cu'] + dynamic_cost
        return dynamic_cost

    elif cost_type == 'generative_diffusion':
        steps = tool_input.get('num_inference_steps', 50)
        return steps * model['cu_per_step']

    return 0.1 # default cost