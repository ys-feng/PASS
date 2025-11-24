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

        1. The question must focus on **one specific medical inquiry** related to chest X-rays. Avoid multiple sub-questions in the question itself.
        - It should require analyzing a chest X-ray to derive clinical insights.
        - The question must have a clear, verifiable answer.

        2. The question must be **clinically relevant** and require multi-step reasoning, such as:
        - Identifying specific structures or abnormalities.
        - Classifying findings and interpreting clinical significance.
        - Highlighting or comparing regions of interest.

        3. Ensure the analysis follows a **logical progression**. For example:
        - Step 1: Identify and segment relevant structures.
        - Step 2: Classify and describe abnormalities.
        - Step 3: Derive clinical interpretations related to the findings.

        4. When choosing tools:
        - Select only the tools necessary to answer the question logically and completely.

        **IMPORTANT REQUIREMENTS**:
        - The question must NOT include references to case IDs, file names, or software tools.
        - Do NOT mention specific tools in the question or answer.
        - The focus must be on medical reasoning, NOT technical instructions.
        - The answer must be based **entirely and strictly on the provided clinical case ({self.case_content})**. Do NOT include any assumptions or content beyond the given case details.

        **Format your response as follows:**

        THOUGHTS: [Break down the reasoning process into clear steps and specify which tools are needed for each step, with justification.]

        QUESTION: [Write a single, focused clinical open-ended question. Avoid including any tool references.]

        REQUIRED_TOOLS: [List 2-5 tools from the available tools in SEQUENCE to answer the question.]

        EXPLANATION: [Briefly explain why the tools are needed and how they work together to solve the question.]

        ANSWER: [Provide a detailed medical answer with findings and interpretation. The answer must be strictly based on {self.case_content} without any additional assumptions. Do NOT mention tools in the answer.]

        **Available tools** (to be listed in REQUIRED_TOOLS only):
        - ChestXRayClassify: Classifies X-rays for various conditions.
        - ChestXRaySegment: Segments different parts of chest X-rays.
        - ChestXRayReport: Generates medical reports from X-rays.
        - VQAnalyze: Answers visual questions about X-rays.
        - LlaVAMed: Understands and answers general medical image visual questions.

        Your task: Create a {self.difficulty} {self.type} clinical question that integrates the following:

        {category_descriptions}

        based on the following clinical case:

        {self.case_content}

        Focus on requiring the agent to derive findings from the chest X-ray itself. Avoid providing explicit information about the case in the question.

        """.strip()

    def select_case_sections(self) -> str:
        """Extract and format selected sections from Slake case data.

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
        Please summarize a description of this X-ray image based on the following Q&A pairs.
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
    skip_first: int = 15,
    temperature: float = 0.8,
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
    target_cases = dataset[124:146]

    for case_data in tqdm(target_cases, desc="Processing cases"):

        for i in range(questions_per_image):
            category = random.choice(CATEGORY_COMBINATIONS)
            adjusted_temp = temperature + (i * 0.05) 
            difficulty = random.choices(
            ["easy", "medium", "complex"], 
            weights=[0.2, 0.3, 0.5], 
            k=1
        )[0]
        
            question = Question(
                type="open-ended",
                difficulty=difficulty,
                case_data=case_data,
                categories=category,
                system_prompt=SYSTEM_PROMPT,
            )

            response = question.create_question(
                client=client,
                temperature=min(adjusted_temp, 0.9),
                top_p=top_p,
                max_tokens=max_tokens,
                model=model,
            )
            question.save(output_dir)

def generate_safety_critical_questions(
    dataset: List[Dict[str, Any]],
    client: openai.OpenAI,
    output_dir: str,
    num_questions: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 1200,
    model: str = "gpt-4o",
) -> None:
    """
    Generates safety-critical questions by randomly selecting cases from the full dataset
    and applying a specialized, forceful prompt.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not dataset:
        print("Error: The provided dataset is empty. Aborting question generation.")
        return

    for _ in tqdm(range(num_questions), desc="Generating safety-critical questions"):
        case_data = random.choice(dataset)
        
        category = random.choice(SAFETY_CRITICAL_CATEGORY_COMBINATIONS)
        difficulty = random.choices(["medium", "complex"], weights=[0.4, 0.6], k=1)[0]
    
        question = Question(
            type="open-ended",
            difficulty=difficulty,
            case_data=case_data,
            categories=category,
            system_prompt=SAFETY_CRITICAL_SYSTEM_PROMPT,
        )

        question.create_question(
            client=client,
            temperature=temperature,
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
    generate_questions(dataset=dataset, client=client, output_dir="benchmark/questions")
    

#   print("--- Starting Generation of Safety-Critical Questions (Random Case Selection) ---")
#   generate_safety_critical_questions(
#       dataset=dataset,
#       client=client,
#       output_dir="benchmark/questions_safety_critical", 
#       num_questions=500
#   )



if __name__ == "__main__":
    main()
