"""
Complete GPT Fine-tuning Implementation
Main script that ties together all components for fine-tuning open-source GPT models.
"""

import os
import torch
import wandb
from transformers import (
    Trainer, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from data_preparation import DataPreprocessor, create_sample_instruction_data
from model_config import ModelConfigurator
import json
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTFineTuner:
    """Main class for fine-tuning GPT models."""
    
    def __init__(self, 
                 model_key: str,
                 output_dir: str = "./fine_tuned_model",
                 use_quantization: bool = False,
                 use_lora: bool = True,
                 use_wandb: bool = False):
        """
        Initialize the fine-tuner.
        
        Args:
            model_key: Model key from ModelConfigurator.SUPPORTED_MODELS
            output_dir: Directory to save the fine-tuned model
            use_quantization: Whether to use 4-bit quantization
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            use_wandb: Whether to log to Weights & Biases
        """
        self.model_key = model_key
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.use_wandb = use_wandb
        
        # Initialize model configurator
        self.model_configurator = ModelConfigurator(model_key, use_quantization)
        
        # Will be set during training
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def setup_wandb(self, project_name: str = "gpt-finetuning", run_name: Optional[str] = None):
        """Setup Weights & Biases for experiment tracking."""
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=run_name or f"{self.model_key}_finetune",
                config={
                    "model": self.model_key,
                    "use_lora": self.use_lora,
                    "use_quantization": self.model_configurator.use_quantization
                }
            )
            
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.model_key}")
        self.model, self.tokenizer = self.model_configurator.load_model_and_tokenizer()
        
        # Apply LoRA if requested
        if self.use_lora:
            logger.info("Applying LoRA for parameter-efficient fine-tuning")
            self.model = self.model_configurator.setup_lora_fine_tuning(self.model)
            
    def prepare_data(self, data_source, data_type: str = "instruction"):
        """
        Prepare training data.
        
        Args:
            data_source: Can be a file path, list of dicts, or None for sample data
            data_type: "instruction" or "conversation"
        """
        logger.info("Preparing training data...")
        
        # Initialize data preprocessor
        preprocessor = DataPreprocessor(self.tokenizer)
        
        # Load data
        if data_source is None:
            logger.info("Using sample instruction data for demonstration")
            raw_data = create_sample_instruction_data()
        elif isinstance(data_source, str):
            if data_source.endswith('.jsonl'):
                raw_data = preprocessor.load_data_from_jsonl(data_source)
            elif data_source.endswith('.csv'):
                raw_data = preprocessor.load_data_from_csv(data_source, 'text')
            else:
                raise ValueError("Unsupported file format. Use .jsonl or .csv")
        elif isinstance(data_source, list):
            raw_data = data_source
        else:
            raise ValueError("data_source must be a file path, list of dicts, or None")
            
        # Format data based on type
        if data_type == "instruction":
            formatted_data = preprocessor.format_for_instruction_tuning(raw_data)
        elif data_type == "conversation":
            formatted_data = preprocessor.format_for_conversation(raw_data)
        else:
            # Assume data is already in the right format
            formatted_data = raw_data
            
        # Prepare dataset splits
        self.tokenized_datasets = preprocessor.prepare_dataset(formatted_data)
        
        logger.info(f"Training samples: {len(self.tokenized_datasets['train'])}")
        if 'validation' in self.tokenized_datasets:
            logger.info(f"Validation samples: {len(self.tokenized_datasets['validation'])}")
            
    def train(self, 
              num_epochs: int = 3,
              learning_rate: float = 5e-5,
              batch_size: int = 4,
              gradient_accumulation_steps: int = 1,
              warmup_steps: int = 100,
              save_steps: int = 500,
              eval_steps: int = 500,
              max_grad_norm: float = 1.0,
              early_stopping_patience: int = 3):
        """Train the model."""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first. Call load_model_and_tokenizer()")
            
        if not hasattr(self, 'tokenized_datasets'):
            raise ValueError("Data must be prepared first. Call prepare_data()")
            
        logger.info("Starting training...")
        
        # Get training arguments
        training_args = self.model_configurator.get_training_arguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            max_grad_norm=max_grad_norm,
            logging_dir=f"{self.output_dir}/logs",
            run_name=f"{self.model_key}_finetune" if self.use_wandb else None
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked LM
            pad_to_multiple_of=8 if training_args.fp16 else None
        )
        
        # Callbacks
        callbacks = []
        if 'validation' in self.tokenized_datasets and early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
            
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets['train'],
            eval_dataset=self.tokenized_datasets.get('validation', None),
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save the model
        self.trainer.save_model()
        
        # Save training metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        logger.info("Training completed!")
        return train_result
        
    def evaluate(self):
        """Evaluate the model on validation set."""
        if self.trainer is None:
            raise ValueError("Model must be trained first. Call train()")
            
        if 'validation' not in self.tokenized_datasets:
            logger.warning("No validation set available for evaluation")
            return None
            
        logger.info("Evaluating model...")
        eval_result = self.trainer.evaluate()
        
        self.trainer.log_metrics("eval", eval_result)
        self.trainer.save_metrics("eval", eval_result)
        
        return eval_result
        
    def generate_sample_outputs(self, prompts: list, max_length: int = 100, num_return_sequences: int = 1):
        """Generate sample outputs to test the fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
            
        logger.info("Generating sample outputs...")
        self.model.eval()
        
        results = []
        with torch.no_grad():
            for prompt in prompts:
                # Encode the prompt
                inputs = self.tokenizer.encode(prompt, return_tensors='pt')
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    
                # Generate response
                with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=max_length,
                        num_return_sequences=num_return_sequences,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode the generated text
                generated_texts = []
                for output in outputs:
                    generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    # Remove the prompt from the generated text
                    if generated_text.startswith(prompt):
                        generated_text = generated_text[len(prompt):].strip()
                    generated_texts.append(generated_text)
                    
                results.append({
                    "prompt": prompt,
                    "generated": generated_texts
                })
                
        return results
        
    def save_model_info(self):
        """Save model configuration and training info."""
        info = {
            "model_key": self.model_key,
            "base_model": self.model_configurator.model_name,
            "model_size": self.model_configurator.model_config["size"],
            "context_length": self.model_configurator.model_config["context_length"],
            "use_lora": self.use_lora,
            "use_quantization": self.model_configurator.use_quantization,
            "training_completed": True
        }
        
        with open(f"{self.output_dir}/model_info.json", "w") as f:
            json.dump(info, f, indent=2)
            
    def cleanup(self):
        """Cleanup resources."""
        if self.use_wandb:
            wandb.finish()

def main():
    """Example usage of the GPTFineTuner."""
    
    # Configuration
    MODEL_KEY = "gpt2"  # Start with a small model for testing
    OUTPUT_DIR = "./fine_tuned_gpt2"
    USE_LORA = True
    USE_QUANTIZATION = False
    USE_WANDB = False  # Set to True if you want to track experiments
    
    # Initialize fine-tuner
    fine_tuner = GPTFineTuner(
        model_key=MODEL_KEY,
        output_dir=OUTPUT_DIR,
        use_quantization=USE_QUANTIZATION,
        use_lora=USE_LORA,
        use_wandb=USE_WANDB
    )
    
    try:
        # Setup experiment tracking
        if USE_WANDB:
            fine_tuner.setup_wandb()
            
        # Load model and tokenizer
        fine_tuner.load_model_and_tokenizer()
        
        # Prepare data (using sample data for demonstration)
        fine_tuner.prepare_data(None, data_type="instruction")
        
        # Train the model
        train_result = fine_tuner.train(
            num_epochs=2,  # Small number for demonstration
            learning_rate=5e-5,
            batch_size=2,  # Small batch size for memory efficiency
            save_steps=100,
            eval_steps=100
        )
        
        # Evaluate the model
        eval_result = fine_tuner.evaluate()
        if eval_result:
            print(f"Evaluation perplexity: {torch.exp(torch.tensor(eval_result['eval_loss'])):.2f}")
            
        # Generate sample outputs
        test_prompts = [
            "### Instruction:\\nWrite a haiku about programming\\n\\n### Response:\\n",
            "### Instruction:\\nExplain recursion in simple terms\\n\\n### Response:\\n"
        ]
        
        sample_outputs = fine_tuner.generate_sample_outputs(test_prompts, max_length=150)
        
        print("\\n=== Sample Outputs ===")
        for result in sample_outputs:
            print(f"Prompt: {result['prompt']}")
            print(f"Generated: {result['generated'][0]}")
            print("-" * 50)
            
        # Save model info
        fine_tuner.save_model_info()
        
        print(f"\\nFine-tuning completed! Model saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        fine_tuner.cleanup()

if __name__ == "__main__":
    main()
