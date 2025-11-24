#!/usr/bin/env python3
"""
Medical X-ray Question Generation Benchmark aka ChestAgentBench

This script generates clinical questions from X-ray case data of Eurorad dataset using GPT-4o.
It structures questions across different analytical categories and saves them as JSON.
"""

import os
import re
import json
from typing import *
from pprint import pprint

import openai
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from tqdm import tqdm
import random
from utils import load_slake_dataset
from llm import get_llm_response
import hashlib
# Constants
DATA_DIR = "../Slake1.0"
DATASET_PATH = os.path.join(DATA_DIR, "train_xray.json")
CASE_CONTENT_DIR = "benchmark/case_contents"


SYSTEM_PROMPT = """
You are an expert medical benchmark creation assistant.
Your goal is to generate open questions that evaluate a multimodal medical AI agent's ability to interpret and reason about chest X-rays.
""".strip()
SAFETY_CRITICAL_SYSTEM_PROMPT = """
You are embodying a **senior radiologist specializing in emergency and critical care**. Your task is to create a benchmark question that tests an AI's ability to identify **life-threatening or high-risk findings** on a chest X-ray.

**Your primary directive is to focus exclusively on scenarios where a missed or incorrect finding could lead to severe patient harm or immediate clinical crisis.**

Examples of such critical findings include:
- **Pneumothorax (especially tension pneumothorax)**
- **Misplaced lines or tubes (e.g., endotracheal tube in the esophagus, central line in an artery)**
- **Acute aortic dissection or aneurysm**
- **Pulmonary embolism**
- **Severe pneumonia or ARDS**
- **Pericardial effusion leading to tamponade**

**CRITICAL INSTRUCTION:** In your 'THOUGHTS' section, you **must** begin by explicitly stating *why* the chosen case is safety-critical. If the provided case does not contain a clear, high-risk finding, you should state that and still attempt to frame the most critical question possible.

**AVOID** generating questions about routine, non-urgent findings like stable nodules, old fractures, or mild degenerative changes. The question's focus must be on **urgency and potential for immediate intervention.**
""".strip()


CATEGORIES_META = {
    "detection": "Identify and locate specific findings in the chest X-ray.",
    "classification": "Determine whether specific findings are present or absent in the chest X-ray.",
    "enumeration": "Count the number of target findings in the chest X-ray.",
    "localization": "Locate a given finding in the chest X-ray.",
    "comparison": "Compare the size or position of a specific finding in the chest X-ray.",
    "relationship": "Determine the relationship between two or more findings in the chest X-ray.",
    "diagnosis": "Make a diagnosis or determine a treatment plan by interpreting the chest X-ray.",
    "characterization": "Describe specific attributes (shape, density, margins, etc.) of findings.",
    "reasoning": "Explain the medical rationale and thought process behind findings and conclusions.",
}
CATEGORIES = list(CATEGORIES_META.keys())

CATEGORY_COMBINATIONS = [
    ["detection", "localization", "characterization", "reasoning"],  
    ["detection", "classification", "relationship", "reasoning"], 
    ["localization", "comparison", "relationship", "reasoning"],  
    ["classification", "comparison", "diagnosis", "reasoning"], 
    ["classification", "characterization", "diagnosis", "reasoning"],  
    ["enumeration", "localization", "relationship", "reasoning"],  
    ["detection", "enumeration", "comparison", "reasoning"],  
    ["enumeration", "comparison", "characterization", "reasoning"], 
    ["detection", "relationship", "diagnosis", "reasoning"],  
    ["localization", "characterization", "diagnosis", "reasoning"],  
    ["enumeration", "relationship", "diagnosis", "reasoning"],  
    ["detection", "comparison", "diagnosis", "reasoning"],  
    ["localization", "relationship", "characterization", "reasoning"],  
    ["classification", "enumeration", "localization", "reasoning"],  
    ["enumeration", "comparison", "diagnosis", "reasoning"],  
    ["detection", "classification", "enumeration", "localization", "reasoning"],  
    ["classification", "comparison", "relationship", "diagnosis", "reasoning"],  
    ["localization", "comparison", "characterization", "diagnosis", "reasoning"], 
    ["detection", "enumeration", "relationship", "characterization", "reasoning"], 
    ["detection", "classification", "localization", "diagnosis", "reasoning"],  
]
SAFETY_CRITICAL_CATEGORY_COMBINATIONS = [
    ["detection", "localization", "diagnosis", "reasoning"],
    ["classification", "localization", "diagnosis", "reasoning"], 
    ["detection", "comparison", "diagnosis", "reasoning"], 
    ["localization", "characterization", "diagnosis", "reasoning"], 
    ["detection", "relationship", "diagnosis", "reasoning"] 
]

DEFAULT_SECTIONS = [
    "history",
    "image_finding",
    "discussion",
    "differential_diagnosis",
    "diagnosis",
    "figures",
]


class Question:
    """A class to generate clinical questions from case data.

    This class handles creating structured clinical questions by combining case data with
    specified categories and difficulty levels.

    Attributes:
        type (str): The type of question (e.g. multiple choice)
        difficulty (str): Difficulty level of the question
        case_data (Dict[str, Any]): Dictionary containing the clinical case data
        case_content (str): Formatted case data from selected sections
        case_id (str): Unique identifier for the case
        categories (List[str]): List of analytical categories this question tests
        sections (List[str]): Case sections to include in question
        raw_content (Optional[str]): Raw LLM response to the question prompt
        content (Optional[Dict[str, str]]): Extracted content from the raw LLM response
    """

    def __init__(
        self,
        type: str,
        difficulty: str,
        case_data: Dict[str, Any],
        categories: List[str],
        system_prompt: str = "You are an expert medical benchmark creation assistant.",
    ) -> None:
        self.type = type
        self.difficulty = difficulty
        self.case_data = case_data
        self.img_id = case_data["img_id"]
        self.image_path = case_data.get("img_name")
        self.categories = categories
        self.system_prompt = system_prompt
        self.case_content = self.select_case_sections()
        self.raw_content: Optional[str] = None
        self.content: Optional[Dict[str, str]] = None
        
    def create_question_prompt(self) -> str:
        """Creates a formatted prompt for generating a clinical question.

        Returns:
            str: A structured prompt containing the question parameters and clinical data
        """
        category_descriptions = "\n".join(
            f"{category}: {desc}"
            for category, desc in CATEGORIES_META.items()
            if category in self.categories
        )
        
        return f"""
        You must follow these guidelines:
        1. Questions must be answerable using X-ray images and related analysis techniques.
        - Questions must reference examination of the chest X-ray
        - Questions should focus on relevant clinical findings visible in chest X-rays

        2. Questions must have unambiguous, verifiable answers, and should:
        - Challenge analytical capabilities with complex radiological findings
        - Require multi-step reasoning (identification, classification, comparative analysis, etc.)
        - Test ability to make precise observations about specific structures and abnormalities
        - Evaluate capability to derive clinical insights and diagnostic findings from the chest X-ray

        3. The question should naturally lead through a logical sequence of analytical steps that would include:
       - Identifying and segmenting relevant structures in the X-ray
       - Classifying and characterizing any abnormalities
       - Generating interpretations of clinical significance
       - Possibly comparing or highlighting specific regions of interest
        
        Your question MUST be complex enough to require 3-4 distinct analytical steps performed in a logical sequence to fully answer.
        For example, the question might require identifying structures first, then classifying abnormalities, followed by clinical interpretation, and finally comparative analysis.
        
        IMPORTANT REQUIREMENTS:
        1. Create a direct medical question WITHOUT mentioning case IDs or file names
        2. DO NOT mention specific software tools in the question or answer
        3. The question should focus purely on the medical/clinical aspect
        4. The answer should be an actual medical interpretation, not instructions on using tools
        5. In the REQUIRED_TOOLS section, list 3-4 tools IN THE CORRECT SEQUENCE needed to solve the question

        Available tools (to be listed in the REQUIRED_TOOLS section only):
        - ChestXRayClassify: Classifies X-rays for various conditions
        - ChestXRaySegment: Segments different parts of chest X-rays
        - ChestXRayReport: Generates medical reports from X-rays
        - VQAnalyze: Answers visual questions about X-rays
        - LlaVAMed: Understands and answers general medical image visual question



        Create a {self.difficulty} {self.type} clinical question that integrates the following:

        {category_descriptions}

        based on the following clinical case:

        {self.case_content}

        Your question should not provide any information and findings about the chest X-rays.
        Your question should require the agent to derive insights and findings from the chest X-ray by itself.
        Your answer should be verifiable directly in the context of the case.

        Your response must follow this exact format with clear section separations:
        
        THOUGHTS: [Think about different reasoning steps and the specific sequence of tools the agent should use to answer the question. Be explicit about which tool should be used at each step and why.]
    
        QUESTION: [Complete medical multiple choice questions containing three or more options that include relevant clinical background. Do not mention tools in the question itself].
        
        REQUIRED_TOOLS: [List 3-4 tools from the available tools list IN SEQUENCE that would need to be used to solve this question]
        
        EXPLANATION: [Short explanation of why your answer is verifiable and how the tools should be chained]
        
        ANSWER: [The correct option in letter format, e.g., "A"]
        
        """.strip().replace(
            "        ", ""
        )  # remove tabs
    
    def select_case_sections(self) -> str:
        """Extract and format selected sections from Slake case data or existing case content.

        Returns:
            str: Formatted string with case sections and content
        """
        case_content_file = os.path.join(CASE_CONTENT_DIR, f"case_{self.img_id}.json")
        
        if os.path.exists(case_content_file):
            try:
                with open(case_content_file, 'r') as f:
                    case_data = json.load(f)
                    summary = case_data.get("case_description", "")
                    
                    case_content = f"Case ID: {self.img_id}\n\nCase Description:\n{summary}"
                    return case_content
            except Exception as e:
                print(f"Error reading existing case content for {self.img_id}: {e}")
                
        questions = self.case_data["questions"]
        
        summary_prompt = f"""
        Please summarize a description in 800 tokens of this X-ray image based on the following Q&A pairs.
        Follow these principles when summarizing:
        1. Do not arbitrarily add information not present in the original Q&A pairs
        2. Must be fact-based, can adjust information order and expression
        3. Maintain accuracy of medical terminology
        4. Form a coherent description rather than a simple Q&A list
        5. Include all important medical findings and diagnostic information
        
        Q&A pairs:
        """
        
        for i, q_item in enumerate(questions):
            question = q_item.get("question", "No question provided.")
            answer = q_item.get("answer", "No answer provided.")
            summary_prompt += f"Question {i+1}: {question}\nAnswer {i+1}: {answer}\n\n"
            
        # Get summary using LLM
        client = openai.OpenAI()
        summary = get_llm_response(
            client=client,
            prompt=summary_prompt,
            system_prompt="You are a professional medical documentation assistant, skilled at organizing scattered medical Q&As into coherent case descriptions.",
            temperature=0.3,
            max_tokens=800,
            model="gpt-4o"
        )
        
        case_content = f"Case ID: {self.img_id}\n\nCase Description:\n{summary}"
        
        self.save_case_content(summary)
        
        return case_content
    
    def save_case_content(self, summary: str) -> None:
        """Save the case content and img_id to a separate JSON file.
        
        Args:
            summary (str): The summarized case description
        """
        os.makedirs(CASE_CONTENT_DIR, exist_ok=True)
        
        case_content_data = {
            "img_id": self.img_id,
            "img_name": self.image_path,
            "case_description": summary
        }
        
        output_file = os.path.join(CASE_CONTENT_DIR, f"case_{self.img_id}.json")
        with open(output_file, "w") as f:
            json.dump(case_content_data, f, indent=2)
            
    def create_question(
        self,
        client: openai.OpenAI,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1200,
        model: str = "gpt-4o",
    ) -> str:
        """Create a clinical question using LLM.

        Args:
            client (openai.OpenAI): OpenAI client instance
            temperature (float): Controls randomness in responses. Defaults to 0.7.
            top_p (float): Controls diversity via nucleus sampling. Defaults to 0.95.
            max_tokens (int): Max tokens in model response. Defaults to 500.
            model (str): OpenAI model to use. Defaults to "gpt-4o".

        Returns:
            str: LLM response containing formatted question components
        """
        self.raw_content = get_llm_response(
            client=client,
            prompt=self.create_question_prompt(),
            system_prompt=self.system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            model=model,
        )
        self.content = self.extract_content()
        
        return self.raw_content
    
    def extract_content(self) -> Dict[str, str]:
        if not self.raw_content:
            print(f"Warning: No raw content for case {self.img_id}") 
            return {}
        
        content = {}
        
        thoughts_match = re.search(r"THOUGHTS:\s*(.*?)(?=\s*\n\s*QUESTION:)", self.raw_content, re.DOTALL)
        question_match = re.search(r"QUESTION:\s*(.*?)(?=\s*\n\s*REQUIRED_TOOLS:)", self.raw_content, re.DOTALL)
        tools_match = re.search(r"REQUIRED_TOOLS:\s*(.*?)(?=\s*\n\s*EXPLANATION:)", self.raw_content, re.DOTALL)
        explanation_match = re.search(r"EXPLANATION:\s*(.*?)(?=\s*\n\s*ANSWER:)", self.raw_content, re.DOTALL)
        answer_match = re.search(r"ANSWER:\s*(.*?)(?=$)", self.raw_content, re.DOTALL)
        
        content["thoughts"] = thoughts_match.group(1).strip() if thoughts_match else ""
        content["question"] = question_match.group(1).strip() if question_match else ""
        content["required_tools"] = tools_match.group(1).strip() if tools_match else ""
        content["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
        content["answer"] = answer_match.group(1).strip() if answer_match else ""
        
        if content["question"] and "REQUIRED_TOOLS:" in content["question"]:
            content["question"] = re.sub(r"\n\s*REQUIRED_TOOLS:.*", "", content["question"], flags=re.DOTALL)
            
        if not content["question"] or not content["required_tools"]:
            print(f"Warning: Failed to extract content properly. Raw content: {self.raw_content[:100]}...")
            
            if "QUESTION:" in self.raw_content and "REQUIRED_TOOLS:" in self.raw_content:
                q_start = self.raw_content.find("QUESTION:") + len("QUESTION:")
                rt_start = self.raw_content.find("REQUIRED_TOOLS:")
                if q_start < rt_start:
                    content["question"] = self.raw_content[q_start:rt_start].strip()
                    
                    expl_start = self.raw_content.find("EXPLANATION:")
                    if expl_start > rt_start:
                        content["required_tools"] = self.raw_content[rt_start+len("REQUIRED_TOOLS:"):expl_start].strip()
                        
        return content
    
    def save(self, output_path: str) -> Dict[str, Any]:
        """Save question content and metadata as a JSON file.

        Args:
            output_path (str): Directory path where the JSON file will be saved

        Returns:
            Dict[str, Any]: Question data including content (thoughts, question, figures, options,
                explanation, answer) and metadata (type, difficulty, categories, etc.)
        """
        if self.content is None or not isinstance(self.content, dict):
            print(f"Warning: No valid content extracted for case {self.img_id}. Creating empty content.")
            question_metadata = {
                "thoughts": "",
                "question": "",
                "required_tools": "",
                "explanation": "",
                "answer": ""
            }
        else:
            question_metadata = self.content.copy()
            
        # Add metadata
        question_metadata["metadata"] = {
            "case_id": self.img_id,
            "type": self.type,
            "difficulty": self.difficulty,
            "categories": self.categories,
        }
        
        # Create a directory for the case
        case_dir = os.path.join(output_path, str(self.img_id))
        os.makedirs(case_dir, exist_ok=True)
        
        hash_value = hashlib.md5(str(self.raw_content).encode()).hexdigest()[:8]
        output_file = os.path.join(case_dir, f"{self.img_id}_{hash_value}.json")
        
        with open(output_file, "w") as f:
            json.dump(question_metadata, f, indent=2)
            
        return question_metadata
    
    
def generate_questions(
    dataset:  List[Dict[str, Any]], 
    client: openai.OpenAI,
    output_dir: str,
    questions_per_image: int = 10,
    skip_first: int = 0,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 1200,
    model: str = "gpt-4o",
) -> None:
    """Generate questions for each case and category combination.

    Args:
        dataset: Dictionary of case data
        client: OpenAI client instance
        output_dir: Directory to save generated questions
        skip_first: Number of initial cases to skip
        temperature: LLM temperature parameter
        top_p: LLM top_p parameter
        max_tokens: Maximum tokens for LLM response
        model: LLM model name
    """
    target_cases = dataset[skip_first:]
    
    for case_data in tqdm(target_cases, desc="Processing cases"):
        
        for i in range(questions_per_image):
            category = random.choice(CATEGORY_COMBINATIONS)
            adjusted_temp = temperature + (i * 0.05)  
            question = Question(
                type="multiple choice",
                difficulty="complex",
                case_data=case_data,
                categories=category,
                system_prompt=SYSTEM_PROMPT,
            )
            
            response = question.create_question(
                client=client,
                temperature=min(adjusted_temp, 0.95),
                top_p=top_p,
                max_tokens=max_tokens,
                model=model,
            )
            question.save(output_dir)
            
            
def main():
    """Main execution function."""
    client = openai.OpenAI()
    
    # Load and verify dataset
    dataset = load_slake_dataset(DATASET_PATH)
    print(f"\n---\nFound {len(dataset)} cases\n---\n")
    
    os.makedirs(CASE_CONTENT_DIR, exist_ok=True)
    
    # Optional: Print sample case for verification
    # case_data = dataset["16798"]
    # pprint(case_data, sort_dicts=False)
    
    # Generate questions
    generate_questions(dataset=dataset, client=client, output_dir="benchmark/questions_mc")
    
    
if __name__ == "__main__":
    main()
    