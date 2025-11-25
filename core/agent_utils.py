import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Union
import numpy as np
import re # For VQA answer cleaning
from core.agent_operators import get_operator_cost_models, calculate_dynamic_cost

def get_sentence_embedding(sentence: str) -> torch.Tensor:
    """
    Gets the embedding representation of a sentence.
    
    Args:
        sentence: The input sentence.
        
    Returns:
        torch.Tensor: The sentence embedding vector.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentence)
    return torch.tensor(embeddings)

def get_operator_embeddings(descriptions: List[str]) -> torch.Tensor:
    """
    Gets the embedding representations for a set of operator descriptions.
    
    Args:
        descriptions: A list of operator descriptions.
        
    Returns:
        torch.Tensor: An operator embedding matrix with shape [num_operators, embedding_dim].
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(descriptions)
    return torch.tensor(embeddings)

def _simple_vqa_score(generated_answer: str, ground_truth: str) -> float:
    """Simple VQA scoring, checks for exact match (ignores case and punctuation)."""
    if not generated_answer or not ground_truth:
        return 0.0
    
    # Clean the answer: convert to lowercase, remove punctuation, remove extra whitespace.
    clean_gen = re.sub(r'[^\w\s]', '', generated_answer).lower().strip()
    clean_gt = re.sub(r'[^\w\s]', '', ground_truth).lower().strip()
    
    # Split possible list-style answers (e.g., "Liver, Heart, Spleen").
    gt_parts = set(p.strip() for p in clean_gt.split(',') if p.strip())
    gen_parts = set(p.strip() for p in clean_gen.split(',') if p.strip())
    
    # If GT has only one answer, check for exact match.
    if len(gt_parts) == 1:
        return 1.0 if clean_gen == clean_gt else 0.0
    
    # If GT has multiple answers, check if the generated one is among them.
    if len(gen_parts) == 1:
        return 1.0 if clean_gen in gt_parts else 0.0
    
    # If both generated and GT have multiple answers, calculate the overlap ratio (Jaccard similarity).
    if len(gt_parts) > 1 and len(gen_parts) > 1:
        intersection = len(gen_parts.intersection(gt_parts))
        union = len(gen_parts.union(gt_parts))
        return float(intersection) / union if union > 0 else 0.0
    
    return 0.0

def calculate_score(result: Dict[str, Any], ground_truth: Any = None) -> float:
    """
    Calculates the score of an execution result (adapted for VQA).
    
    Args:
        result: The workflow execution result dictionary.
        ground_truth: The ground truth answer string (from the Slake dataset).
        
    Returns:
        float: A score between 0 and 1.
    """
    # Handle error cases.
    if not result or result.get("status") == "error":
        return 0.0
    
    # Base score: 0.1 points for successful execution (non-error status).
    base_score = 0.1 if result.get("status") != "error" else 0.0
    
    # Core VQA task score: compare the final answer.
    vqa_score = 0.0
    final_solution = result.get("final_solution")
    
    if final_solution and isinstance(ground_truth, str):
        # Try to extract the answer from final_solution.
        # Assume final_solution is a string, or a dictionary from which the answer can be parsed.
        generated_answer = ""
        if isinstance(final_solution, str):
            generated_answer = final_solution
        elif isinstance(final_solution, dict):
            # Try to find common answer fields.
            if "answer" in final_solution:
                generated_answer = final_solution["answer"]
            elif "response" in final_solution:
                generated_answer = final_solution["response"]
            # More lookup logic can be added.
       
        if isinstance(generated_answer, str):
            vqa_score = _simple_vqa_score(generated_answer, ground_truth)
        else:
            # If the generated answer is not a string, it cannot be compared.
            vqa_score = 0.0
    
    # Weight the VQA score (e.g., by 0.8).
    weighted_vqa_score = vqa_score * 0.8
    
    # Architecture complexity score: fewer layers are better (slight penalty for more layers).
    complexity_score = 0.0
    architecture = result.get("architecture", [])
    num_layers = len(architecture)
    if num_layers > 0:
        # Example: 1 layer=0.1, 2 layers=0.05, 3 layers=0.02, 4+ layers=0.0.
        complexity_score = max(0.0, 0.1 - 0.03 * (num_layers - 1))
    
    # Final score = base score + VQA score + complexity score.
    total_score = base_score + weighted_vqa_score + complexity_score
    
    # Ensure the score is within the 0-1 range.
    return min(max(total_score, 0.0), 1.0)

COST_MODELS = get_operator_cost_models()

def calculate_cost(workflow_result: dict) -> float:
    """
    Calculates the total cost of the entire workflow.

    Args:
        workflow_result: The complete result dictionary returned after workflow execution.

    Returns:
        The total cost (in CU).
    """
    total_cost = 0.0
    layer_results = workflow_result.get("layer_results", [])
    
    for layer in layer_results:
        for step_result in layer:
            operator_name = step_result.get("operator")
            
            # Get inputs and outputs
            tool_output = step_result.get("result", {})
            tool_input = step_result.get("input_params", {}) 

            if operator_name:
                try:
                    op_cost = calculate_dynamic_cost(
                        operator_name,
                        COST_MODELS,
                        tool_input=tool_input,
                        tool_output=tool_output
                    )
                    total_cost += op_cost
                except Exception as e:
                    print(f"Warning: Failed to calculate cost for {operator_name}. Error: {e}")
                    total_cost += 0.1 # Add a default cost even if calculation fails.
    
    return total_cost

def get_text_similarity(text1: str, text2: str) -> float:
    """
    Calculates the similarity between two texts.
    
    Args:
        text1: The first text.
        text2: The second text.
        
    Returns:
        float: A similarity score (0-1).
    """
    if not text1 or not text2:
        return 0.0
        
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Get embeddings
    embedding1 = model.encode(text1, convert_to_numpy=True)
    embedding2 = model.encode(text2, convert_to_numpy=True)
    
    # Calculate cosine similarity
    # Avoid division by zero
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    return float(np.clip(similarity, 0.0, 1.0)) # Ensure the result is in the 0-1 range

def sample_operators(probs: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
    """
    Samples operators based on a probability distribution and a cumulative threshold.

    Args:
        probs (torch.Tensor): The probability distribution Tensor for the operators.
        threshold (float, optional): The cumulative probability threshold. Defaults to 0.25.

    Returns:
        torch.Tensor: A Tensor of indices for the selected operators.
    """
    device = probs.device
    # Ensure operations are on the correct device
    probs = probs.to(device).detach() # Ensure gradients are not computed
    
    num_ops = probs.size(0)
    if num_ops == 0:
        # If there are no operator probabilities, return an empty Tensor
        return torch.tensor([], dtype=torch.long, device=device)

    selected = torch.tensor([], dtype=torch.long, device=device)
    cumulative = 0.0
    # Create a Tensor containing all operator indices
    remaining = torch.arange(num_ops, device=device)
    
    # Check if the sum of probabilities is zero or very small
    prob_sum = probs.sum().item()
    if prob_sum < 1e-9:
        # If the sum of probabilities is close to zero, sampling may not be possible; select the one with the highest probability.
        if num_ops > 0:
            return probs.argmax().unsqueeze(0)
        else:
            return torch.tensor([], dtype=torch.long, device=device)
    
    # Normalize probabilities just in case
    probs = probs / prob_sum
    
    # Continue sampling while the cumulative probability is less than the threshold and there are remaining operators.
    # Add an iteration limit to prevent infinite loops.
    max_iterations = num_ops * 2
    iterations = 0
    while cumulative < threshold and remaining.numel() > 0 and iterations < max_iterations:
        iterations += 1
        remaining_probs = probs[remaining]
        
        # Check the sum of remaining probabilities
        remaining_prob_sum = remaining_probs.sum().item()
        if remaining_prob_sum < 1e-9:
            break # No valid probabilities available for sampling
        
        # Normalize remaining probabilities
        remaining_probs = remaining_probs / remaining_prob_sum
        
        # Sample one operator from the remaining ones using multinomial sampling based on their probabilities.
        try:
            sampled_indices_in_remaining = torch.multinomial(remaining_probs, num_samples=1)
        except RuntimeError as e:
            # Handle possible errors, e.g., if all probabilities are zero or NaN.
            print(f"Sampling error: {e}, remaining probabilities: {remaining_probs}")
            # Try to select the one with the highest remaining probability.
            if remaining.numel() > 0:
                max_prob_idx_in_remaining = remaining_probs.argmax()
                sampled_indices_in_remaining = max_prob_idx_in_remaining.unsqueeze(0)
            else:
                break # No remaining items to choose from
        
        # Get the true index of the sampled operator in the original probs tensor.
        idx_tensor = remaining[sampled_indices_in_remaining].squeeze()
        # Ensure idx is a scalar Long Tensor.
        if idx_tensor.dim() > 0:
            idx = idx_tensor[0] # If multinomial returns multiple, take the first one.
        else:
            idx = idx_tensor
        
        idx = idx.long() # Ensure it is of Long type.

        # Check if this operator has already been selected.
        if not torch.any(selected == idx):
            selected = torch.cat([selected, idx.unsqueeze(0)]) # Add the selected index to the 'selected' list.
            # Accumulate the original probability (from the unnormalized probs).
            cumulative += probs[idx].item()
        
        # Remove the selected operator index from the 'remaining' list.
        # Create a boolean mask to mark the unselected elements.
        mask = (remaining != idx)
        remaining = remaining[mask] # Update the 'remaining' list.
    
    # If 'selected' is still empty after the loop (e.g., due to a very small threshold or concentrated probabilities), 
    # select at least the operator with the highest probability.
    if selected.numel() == 0 and num_ops > 0:
        selected = probs.argmax().unsqueeze(0)
    
    return selected.to(device)
