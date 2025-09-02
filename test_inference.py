import os
import sys
import logging
import warnings
import csv

# Configure logging levels (less aggressive than stderr redirection)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Load model with error handling
try:
    model_dir = "./gec_model"
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = TFT5ForConditionalGeneration.from_pretrained(model_dir)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model directory exists and contains the required files.")
    sys.exit(1)

max_length = 64

def correct_grammar(sentence):
    """
    Correct grammar in a given sentence using T5 model
    
    Args:
        sentence (str): Input sentence to correct
        
    Returns:
        str: Grammar-corrected sentence
    """
    # Input validation
    if not sentence or not sentence.strip():
        return sentence
    
    # Truncate if too long (leave room for "fix: " prefix)
    if len(sentence.split()) > max_length - 10:
        sentence = ' '.join(sentence.split()[:max_length-10])
        print(f"Warning: Input truncated to fit model constraints")
    
    try:
        # Tokenize input
        input_ids = tokenizer(
            "fix: " + sentence, 
            return_tensors="tf", 
            padding='max_length', 
            truncation=True, 
            max_length=max_length
        ).input_ids
        
        # Generate correction
        output = model.generate(
            input_ids, 
            max_length=max_length,
            num_beams=4,  # Beam search for better quality
            early_stopping=True,
            do_sample=False
        )
        
        # Extract output
        if hasattr(output, 'sequences'):
            output_ids = output.sequences[0]
        else:
            output_ids = output[0]
            
        # Decode result
        decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
        return decoded
        
    except Exception as e:
        print(f"Error during correction: {e}")
        return sentence  # Return original if correction fails

# Test sentences
test_sentences = [
    "tell what mistake is me doing",
]

print("Processing sentences...")
results = []

for i, sent in enumerate(test_sentences, 1):
    print(f"Processing sentence {i}/{len(test_sentences)}: {sent}")
    try:
        corrected = correct_grammar(sent)
        results.append((sent, corrected))
    except Exception as e:
        print(f"Error processing sentence '{sent}': {e}")
        results.append((sent, sent))  # Keep original if error occurs

# Display results
print("\n" + "="*60)
print("GRAMMAR CORRECTION RESULTS")
print("="*60)

for sent, corrected in results:
    print(f"Original:  {sent}")
    print(f"Corrected: {corrected}")
    print("-" * 40)

# Save to CSV with error handling
try:
    # Prepare data for CSV
    csv_data = []
    for sent, corrected in results:
        csv_data.append([sent, corrected])
    
    # Check if file exists to determine if headers are needed
    file_exists = os.path.isfile("self_training_data.csv")
    
    # Write to CSV
    with open("self_training_data.csv", "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write headers if file is new
        if not file_exists:
            writer.writerow(["input", "target"])
            print("Created new CSV file with headers")
        
        # Write data
        writer.writerows(csv_data)
        print(f"Successfully saved {len(csv_data)} corrections to self_training_data.csv")
        
except Exception as e:
    print(f"Error saving to CSV: {e}")

print("\nProcessing complete!")
