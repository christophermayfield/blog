"""
Model Configuration for GPT Fine-tuning
Handles model loading and configuration for various open-source GPT models.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import bitsandbytes as bnb
from typing import Tuple, Optional, Dict, Any

class ModelConfigurator:
    """Handles model loading and configuration for fine-tuning."""
    
    # Popular open-source models and their configurations
    SUPPORTED_MODELS = {
        # GPT-2 variants
        "gpt2": {
            "model_name": "gpt2",
            "context_length": 1024,
            "size": "124M parameters"
        },
        "gpt2-medium": {
            "model_name": "gpt2-medium", 
            "context_length": 1024,
            "size": "355M parameters"
        },
        "gpt2-large": {
            "model_name": "gpt2-large",
            "context_length": 1024, 
            "size": "774M parameters"
        },
        "gpt2-xl": {
            "model_name": "gpt2-xl",
            "context_length": 1024,
            "size": "1.5B parameters"
        },
        
        # EleutherAI models
        "gpt-neo-125m": {
            "model_name": "EleutherAI/gpt-neo-125M",
            "context_length": 2048,
            "size": "125M parameters"
        },
        "gpt-neo-1.3b": {
            "model_name": "EleutherAI/gpt-neo-1.3B",
            "context_length": 2048,
            "size": "1.3B parameters"
        },
        "gpt-neo-2.7b": {
            "model_name": "EleutherAI/gpt-neo-2.7B", 
            "context_length": 2048,
            "size": "2.7B parameters"
        },
        "gpt-j-6b": {
            "model_name": "EleutherAI/gpt-j-6B",
            "context_length": 2048,
            "size": "6B parameters"
        },
        
        # Meta Llama models (require approval)
        "llama-7b": {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "context_length": 4096,
            "size": "7B parameters"
        },
        "llama-13b": {
            "model_name": "meta-llama/Llama-2-13b-hf",
            "context_length": 4096,
            "size": "13B parameters"
        },
        
        # Microsoft DialoGPT (good for conversations)
        "dialogpt-small": {
            "model_name": "microsoft/DialoGPT-small",
            "context_length": 1024,
            "size": "117M parameters"
        },
        "dialogpt-medium": {
            "model_name": "microsoft/DialoGPT-medium", 
            "context_length": 1024,
            "size": "345M parameters"
        },
        "dialogpt-large": {
            "model_name": "microsoft/DialoGPT-large",
            "context_length": 1024,
            "size": "762M parameters"
        }
    }
    
    def __init__(self, model_key: str, use_quantization: bool = False):
        """
        Initialize model configurator.
        
        Args:
            model_key: Key from SUPPORTED_MODELS
            use_quantization: Whether to use 4-bit quantization to save memory
        """
        if model_key not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_key}. Choose from: {list(self.SUPPORTED_MODELS.keys())}")
            
        self.model_key = model_key
        self.model_config = self.SUPPORTED_MODELS[model_key]
        self.model_name = self.model_config["model_name"]
        self.use_quantization = use_quantization
        
    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load the model and tokenizer."""
        print(f"Loading {self.model_config['size']} model: {self.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Configure quantization if requested
        if self.use_quantization:
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            }
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
        return model, tokenizer
    
    def setup_lora_fine_tuning(self, 
                              model: AutoModelForCausalLM,
                              r: int = 16,
                              lora_alpha: int = 32,
                              lora_dropout: float = 0.1,
                              target_modules: Optional[list] = None) -> AutoModelForCausalLM:
        """
        Set up LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
        
        Args:
            model: The base model
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Modules to apply LoRA to (auto-detected if None)
        """
        
        # Prepare model for k-bit training if using quantization
        if self.use_quantization:
            model = prepare_model_for_kbit_training(model)
            
        # Auto-detect target modules if not specified
        if target_modules is None:
            if "llama" in self.model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            elif "gpt" in self.model_name.lower():
                target_modules = ["c_attn", "c_proj"]
            else:
                # Generic approach - target attention layers
                target_modules = ["attention"]
                
        # Configure LoRA
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def get_training_arguments(self,
                             output_dir: str = "./results",
                             num_train_epochs: int = 3,
                             per_device_train_batch_size: int = 4,
                             per_device_eval_batch_size: int = 4,
                             gradient_accumulation_steps: int = 1,
                             learning_rate: float = 5e-5,
                             weight_decay: float = 0.01,
                             logging_steps: int = 10,
                             save_steps: int = 500,
                             eval_steps: int = 500,
                             warmup_steps: int = 100,
                             max_grad_norm: float = 1.0,
                             **kwargs) -> TrainingArguments:
        """Get training arguments optimized for the model size."""
        
        # Adjust batch size based on model size and available memory
        model_size = self.model_config["size"]
        if "6B" in model_size or "7B" in model_size or "13B" in model_size:
            # Larger models need smaller batch sizes
            per_device_train_batch_size = min(per_device_train_batch_size, 2)
            per_device_eval_batch_size = min(per_device_eval_batch_size, 2)
            gradient_accumulation_steps = max(gradient_accumulation_steps, 4)
            
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
            report_to=["wandb"] if "wandb" in kwargs else [],
            **kwargs
        )
    
    def estimate_memory_requirements(self) -> Dict[str, str]:
        """Estimate memory requirements for training."""
        model_size = self.model_config["size"]
        context_length = self.model_config["context_length"]
        
        # Rough estimates (these can vary significantly)
        memory_estimates = {
            "124M": {"training": "4-6GB", "inference": "1-2GB"},
            "355M": {"training": "8-12GB", "inference": "2-3GB"}, 
            "774M": {"training": "16-24GB", "inference": "3-4GB"},
            "1.5B": {"training": "24-32GB", "inference": "4-6GB"},
            "2.7B": {"training": "40-48GB", "inference": "6-8GB"},
            "6B": {"training": "80-96GB", "inference": "12-16GB"},
            "7B": {"training": "96-112GB", "inference": "14-18GB"},
            "13B": {"training": "160-192GB", "inference": "26-32GB"}
        }
        
        # Extract size number
        for size_key in memory_estimates:
            if size_key in model_size:
                base_estimates = memory_estimates[size_key]
                break
        else:
            base_estimates = {"training": "Unknown", "inference": "Unknown"}
            
        result = {
            "model_size": model_size,
            "context_length": f"{context_length} tokens",
            "training_memory": base_estimates["training"],
            "inference_memory": base_estimates["inference"],
            "with_quantization": f"~50% of training memory ({base_estimates['training']})" if self.use_quantization else "N/A",
            "with_lora": f"~30% of training memory ({base_estimates['training']})"
        }
        
        return result

def print_model_info():
    """Print information about all supported models."""
    print("=== Supported Open-Source GPT Models ===\\n")
    
    configurator = ModelConfigurator("gpt2")  # Dummy instance
    
    for key, config in configurator.SUPPORTED_MODELS.items():
        print(f"Key: {key}")
        print(f"  Model: {config['model_name']}")
        print(f"  Size: {config['size']}")
        print(f"  Context: {config['context_length']} tokens")
        print()

if __name__ == "__main__":
    print_model_info()
    
    # Example usage
    print("=== Example: GPT-2 Medium Configuration ===")
    config = ModelConfigurator("gpt2-medium", use_quantization=False)
    memory_info = config.estimate_memory_requirements()
    
    for key, value in memory_info.items():
        print(f"{key}: {value}")
