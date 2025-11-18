import os
import json
from typing import Dict, List
from collections import defaultdict

def load_slake_dataset(dataset_path: str) -> List[Dict[str, any]]:
    """
    [
      {
        "img_id": ...,
        "img_name": "...",
        "questions": [
            { "question": ..., "answer": ..., ...},
            ...
        ]
      },
      ...
    ]
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_dict = defaultdict(list)
    img_name_map = {}

    for item in data:
        img_id = item["img_id"]
        img_name = f"../data/Slake1.0/imgs/{item['img_name']}"
        img_name_map[img_id] = img_name
        qinfo = {k: v for k, v in item.items() if k not in ("img_id", "img_name")}
        img_dict[img_id].append(qinfo)

    dataset = []
    for img_id, questions in img_dict.items():
        dataset.append({
            "img_id": img_id,
            "img_name": img_name_map[img_id],
            "questions": questions
        })

    return dataset
def load_eurorad_dataset(
    dataset_path: str,
    section: str = "any",
    as_dict: bool = False,
    filter_by_caption: List[str] = [
        "xray",
        "x-ray",
        "x ray",
        "ray",
        "xr",
        "radiograph",
        "radiogram",
        "plain film",
    ],
) -> List[Dict] | Dict[str, Dict]:
    """
    Load a dataset from a JSON file.

    Args:
        dataset_path (str): Path to the JSON dataset file.
        section (str, optional): Section of the dataset to load. Defaults to "any".
        as_dict (bool, optional): Whether to return data as dict. Defaults to False.
        filter_by_caption (List[str], optional): List of strings to filter cases by caption content. Defaults to [].

    Returns:
        List[Dict] | Dict[str, Dict]: The loaded dataset as a list of dictionaries or dict if as_dict=True.

    Raises:
        FileNotFoundError: If dataset_path does not exist
        json.JSONDecodeError: If file is not valid JSON
    """

    with open(dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if filter_by_caption:
        filtered_data = {}
        for case_id, case in data.items():
            if any(
                any(x in subfig["caption"].lower() for x in filter_by_caption)
                for figure in case["figures"]
                for subfig in figure["subfigures"]
            ) or any(x in case["image_finding"].lower() for x in filter_by_caption):
                filtered_data[case_id] = case
        data = filtered_data

    if section != "any":
        section = section.strip().lower()
        if not as_dict:
            data = [
                item for item in data.values() if item.get("section", "").strip().lower() == section
            ]
        else:
            data = {
                k: v for k, v in data.items() if v.get("section", "").strip().lower() == section
            }

    elif not as_dict:
        data = list(data.values())

    return data


def save_dataset(dataset: Dict | List[Dict], dataset_path: str):
    """
    Save a dataset to a JSON file.

    Args:
        dataset (Dict | List[Dict]): The dataset to save as a dictionary or list of dictionaries.
        dataset_path (str): Path where the JSON dataset file will be saved.
    """
    with open(dataset_path, "w", encoding="utf-8") as file:
        json.dump(dataset, file)
