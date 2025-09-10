"""
Data Preparation for GPT Fine-tuning
Handles various data formats and preprocessing steps.
"""

import json
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import re
from typing import List, Dict, Any, Optional

class DataPreprocessor:
    """Handles data preprocessing for GPT fine-tuning."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def load_data_from_jsonl(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def load_data_from_csv(self, file_path: str, text_column: str) -> List[Dict]:
        """Load data from CSV file."""
        df = pd.read_csv(file_path)
        return [{"text": row[text_column]} for _, row in df.iterrows()]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with training
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def format_for_instruction_tuning(self, data: List[Dict]) -> List[Dict]:
        """
        Format data for instruction tuning.
        Expected input format: [{"instruction": "...", "input": "...", "output": "..."}]
        """
        formatted_data = []
        
        for item in data:
            # Create instruction-following format
            if "input" in item and item["input"]:
                text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            else:
                text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            
            formatted_data.append({"text": self.clean_text(text)})
        
        return formatted_data
    
    def format_for_conversation(self, data: List[Dict]) -> List[Dict]:
        """
        Format data for conversational fine-tuning.
        Expected input: [{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]
        """
        formatted_data = []
        
        for item in data:
            conversation = ""
            for message in item["messages"]:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    conversation += f"Human: {content}\n\n"
                elif role == "assistant":
                    conversation += f"Assistant: {content}\n\n"
            
            formatted_data.append({"text": self.clean_text(conversation)})
        
        return formatted_data
    
    def tokenize_function(self, examples):
        """Tokenize the text for training."""
        # Tokenize the text
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=512,  # Adjust based on your model's context length
            return_overflowing_tokens=False,
        )
        
        # For language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(self, 
                       data: List[Dict], 
                       test_size: float = 0.1,
                       val_size: float = 0.1) -> DatasetDict:
        """Prepare the dataset for training."""
        
        # Split the data
        if test_size + val_size >= 1.0:
            raise ValueError("test_size + val_size must be less than 1.0")
        
        # First split: train + val vs test
        train_val_data, test_data = train_test_split(
            data, test_size=test_size, random_state=42
        )
        
        # Second split: train vs val
        if val_size > 0:
            adjusted_val_size = val_size / (1 - test_size)
            train_data, val_data = train_test_split(
                train_val_data, test_size=adjusted_val_size, random_state=42
            )
        else:
            train_data = train_val_data
            val_data = []
        
        # Create datasets
        dataset_dict = {}
        dataset_dict["train"] = Dataset.from_list(train_data)
        
        if val_data:
            dataset_dict["validation"] = Dataset.from_list(val_data)
        
        if test_data:
            dataset_dict["test"] = Dataset.from_list(test_data)
        
        # Tokenize all datasets
        tokenized_datasets = DatasetDict()
        for split, dataset in dataset_dict.items():
            tokenized_datasets[split] = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
            )
        
        return tokenized_datasets

# Example usage functions
def create_sample_instruction_data():
    """Create sample instruction-following data."""
    sample_data = [
        {
            "instruction": "Write a short poem about technology",
            "input": "",
            "output": "Silicon dreams and digital streams,\nCode flows like electronic beams.\nIn circuits bright, we find our way,\nThrough the digital night and day."
        },
        {
            "instruction": "Explain what machine learning is",
            "input": "in simple terms",
            "output": "Machine learning is a way for computers to learn patterns from data without being explicitly programmed for every scenario. It's like teaching a computer to recognize cats by showing it thousands of cat pictures, rather than programming every possible way a cat could look."
        },
        {
            "instruction": "Translate the following to French",
            "input": "Hello, how are you?",
            "output": "Bonjour, comment allez-vous ?"
        }
    ]
    
    return sample_data

def create_sample_conversation_data():
    """Create sample conversational data."""
    sample_data = [
        {
            "messages": [
                {"role": "user", "content": "What's the weather like today?"},
                {"role": "assistant", "content": "I don't have access to real-time weather data, but I'd be happy to help you find weather information! You could check a weather app, website like weather.com, or ask a voice assistant for current conditions in your area."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Can you help me write a resume?"},
                {"role": "assistant", "content": "I'd be happy to help you write a resume! To get started, I'll need some information about you. Could you tell me about your work experience, education, and the type of job you're applying for?"},
                {"role": "user", "content": "I'm applying for a software engineering position. I have 3 years of experience with Python and JavaScript."},
                {"role": "assistant", "content": "Great! Here's a basic structure for your software engineering resume:\n\n**Contact Information**\n- Name, phone, email, LinkedIn, GitHub\n\n**Professional Summary**\n- 3+ years of software development experience\n- Proficient in Python and JavaScript\n- [Add specific achievements or specializations]\n\n**Technical Skills**\n- Languages: Python, JavaScript\n- [Add frameworks, databases, tools you know]\n\n**Work Experience**\n- [List your positions with bullet points of achievements]\n\nWould you like me to help you fill in any specific sections?"}
            ]
        }
    ]
    
    return sample_data

if __name__ == "__main__":
    # Example of how to use the DataPreprocessor
    from transformers import AutoTokenizer
    
    # Load a tokenizer (you'd use your chosen model's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(tokenizer)
    
    # Create sample data
    instruction_data = create_sample_instruction_data()
    conversation_data = create_sample_conversation_data()
    
    # Format data
    formatted_instruction = preprocessor.format_for_instruction_tuning(instruction_data)
    formatted_conversation = preprocessor.format_for_conversation(conversation_data)
    
    # Prepare datasets
    instruction_dataset = preprocessor.prepare_dataset(formatted_instruction)
    conversation_dataset = preprocessor.prepare_dataset(formatted_conversation)
    
    print("Sample formatted instruction data:")
    print(formatted_instruction[0]["text"][:200] + "...")
    print("\nDataset splits:", instruction_dataset.keys())
    print("Training samples:", len(instruction_dataset["train"]))
