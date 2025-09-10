#!/usr/bin/env python3
"""
Quick Start Example for GPT Fine-tuning
This script demonstrates the simplest way to fine-tune a GPT model.
"""

import sys
import os

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpt_fine_tuner import GPTFineTuner

def main():
    """Simple fine-tuning example."""
    
    print("=== GPT Fine-tuning Quick Start ===\n")
    
    # Step 1: Choose your model
    # Start with a small model for testing
    MODEL_OPTIONS = {
        "1": ("gpt2", "GPT-2 Small (124M) - Good for testing"),
        "2": ("gpt2-medium", "GPT-2 Medium (355M) - Better quality"),
        "3": ("gpt-neo-125m", "GPT-Neo 125M - Similar to GPT-2 Small"),
        "4": ("dialogpt-medium", "DialoGPT Medium (345M) - Good for conversations")
    }
    
    print("Choose a model:")
    for key, (model_key, description) in MODEL_OPTIONS.items():
        print(f"{key}. {description}")
    
    choice = input("\nEnter your choice (1-4) [default: 1]: ").strip() or "1"
    
    if choice not in MODEL_OPTIONS:
        print("Invalid choice, using GPT-2 Small")
        choice = "1"
        
    model_key, model_description = MODEL_OPTIONS[choice]
    print(f"\nSelected: {model_description}")
    
    # Step 2: Configure training
    print("\n=== Training Configuration ===")
    
    # Memory-efficient settings for most users
    use_lora = True
    use_quantization = input("Use 4-bit quantization to save memory? (y/N): ").lower().startswith('y')
    
    output_dir = input("Output directory [./my_fine_tuned_model]: ").strip() or "./my_fine_tuned_model"
    
    # Step 3: Initialize fine-tuner
    print(f"\n=== Initializing Fine-tuner ===")
    
    fine_tuner = GPTFineTuner(
        model_key=model_key,
        output_dir=output_dir,
        use_quantization=use_quantization,
        use_lora=use_lora,
        use_wandb=False  # Disable for simplicity
    )
    
    # Step 4: Load model
    print("Loading model and tokenizer...")
    fine_tuner.load_model_and_tokenizer()
    
    # Step 5: Create sample training data
    print("Preparing sample training data...")
    
    # You can replace this with your own data
    sample_data = [
        {
            "instruction": "Write a haiku about programming",
            "input": "",
            "output": "Code flows like water,\nBugs hide in silent shadows,\nDebug brings the light."
        },
        {
            "instruction": "Explain what recursion is",
            "input": "in simple terms",
            "output": "Recursion is when a function calls itself to solve a problem by breaking it into smaller, similar problems. It's like looking into two mirrors facing each other - you see reflections of reflections until they become too small to see."
        },
        {
            "instruction": "Write a short story",
            "input": "about a robot learning to paint",
            "output": "Unit-7 had calculated a million color combinations, but had never truly seen beauty. When it first held a brush, something unexpected happened. The paint didn't follow algorithms - it flowed with emotion the robot didn't know it possessed. Each stroke taught Unit-7 that art wasn't about perfection, but about expression."
        },
        {
            "instruction": "Translate to Spanish",
            "input": "Hello, how are you today?",
            "output": "Hola, ¿cómo estás hoy?"
        },
        {
            "instruction": "Write a function to reverse a string",
            "input": "",
            "output": "def reverse_string(s):\n    return s[::-1]\n\n# Alternative approach:\ndef reverse_string_loop(s):\n    result = ''\n    for char in s:\n        result = char + result\n    return result"
        }
    ]
    
    # Prepare the data
    fine_tuner.prepare_data(sample_data, data_type="instruction")
    
    # Step 6: Train the model
    print("\n=== Starting Training ===")
    print("This may take several minutes to hours depending on your hardware...")
    
    # Conservative settings for most hardware
    train_result = fine_tuner.train(
        num_epochs=2,           # Small number for demonstration
        learning_rate=5e-5,     # Conservative learning rate
        batch_size=1,           # Small batch size for memory efficiency
        gradient_accumulation_steps=4,  # Effective batch size = 4
        save_steps=50,
        eval_steps=50,
        warmup_steps=10
    )
    
    print(f"Training completed! Final loss: {train_result.training_loss:.4f}")
    
    # Step 7: Evaluate
    eval_result = fine_tuner.evaluate()
    if eval_result:
        perplexity = 2 ** eval_result['eval_loss']  # Convert loss to perplexity
        print(f"Evaluation perplexity: {perplexity:.2f}")
    
    # Step 8: Test the model
    print("\n=== Testing Fine-tuned Model ===")
    
    test_prompts = [
        "### Instruction:\nWrite a haiku about the ocean\n\n### Response:\n",
        "### Instruction:\nExplain photosynthesis\n### Input:\nin simple terms\n\n### Response:\n",
        "### Instruction:\nWrite a short poem about friendship\n\n### Response:\n"
    ]
    
    sample_outputs = fine_tuner.generate_sample_outputs(
        test_prompts, 
        max_length=150,
        num_return_sequences=1
    )
    
    print("\n" + "="*60)
    print("SAMPLE OUTPUTS FROM FINE-TUNED MODEL")
    print("="*60)
    
    for i, result in enumerate(sample_outputs, 1):
        print(f"\n--- Example {i} ---")
        print(f"Prompt: {result['prompt'].strip()}")
        print(f"Generated: {result['generated'][0]}")
        print("-" * 40)
    
    # Step 9: Save model info
    fine_tuner.save_model_info()
    
    print(f"\n=== Success! ===")
    print(f"Fine-tuned model saved to: {output_dir}")
    print(f"You can now use this model for inference or further training.")
    
    print(f"\nTo load your model later:")
    print(f"from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    
    # Cleanup
    fine_tuner.cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        print("Please check your GPU memory and dependencies.")
        raise
