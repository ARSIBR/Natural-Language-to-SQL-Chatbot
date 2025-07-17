import json
from typing import List, Dict, Any

def load_few_shot_examples(file_path="few_shot_examples.json") -> List[Dict[str, Any]]:
    """Loads few-shot examples from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            examples = json.load(f)
        return examples
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading few-shot examples from {file_path}: {e}")
        return []
