# Grammar Error Correction (GEC) with T5

A comprehensive Grammar Error Correction system built using T5 (Text-to-Text Transfer Transformer) model. This project provides end-to-end training, evaluation, and inference capabilities for automated grammar correction.

## 📁 Project Structure

```
gec_model/                          # Main project directory
├── config.json                     # Model configuration file
├── generation_config.json          # Text generation parameters
├── special_tokens_map.json         # Special tokens mapping
├── spiece.model                    # SentencePiece tokenizer model
├── tf_model.h5                     # Trained TensorFlow model weights
├── tokenizer_config.json           # Tokenizer configuration
├── train_gec_t5.py                 # Main training script
├── test_inference.py               # Inference and testing script
├── preprocessing_pipeline.py       # Data preprocessing utilities
└── convert_parquet_to_csv.py      # Data format conversion script

processed_data/                     # Processed dataset directory
├── train_dataset.parquet          # Training data
├── val_dataset.parquet            # Validation data
└── test_dataset.parquet           # Test data

datasets/                          # Raw dataset collections
├── archive5/                      # Additional archive dataset
├── c4_200m/                       # C4 corpus subset (200M samples)
├── coedit/                        # CoEdIT dataset
├── fce/                           # First Certificate in English dataset
├── jfleg/                         # JHU FLuency-Extended GUG dataset
├── lang8/                         # Lang8 learner corpus
├── lang8.bea19/                   # Lang8 BEA-2019 shared task version
└── wi+locness/                    # Write & Improve + LOCNESS corpus
```

## 🗂️ Datasets Used

This project leverages multiple high-quality grammar error correction datasets:

### Primary Datasets
- **FCE (First Certificate in English)**: Cambridge learner corpus with error annotations
- **JFLEG (JHU FLuency-Extended GUG)**: Fluency-focused error correction dataset
- **Lang8**: Large-scale learner corpus from the Lang8 language learning platform
- **Lang8.BEA19**: Refined version used in the BEA-2019 shared task
- **W&I+LOCNESS**: Write & Improve and LOCNESS corpus combination
- **CoEdIT**: Comprehensive editing dataset for text improvement
- **C4_200M**: Large-scale subset of the Common Crawl C4 corpus (200M samples)

### Dataset Characteristics
- **Training samples**: Variable (depends on preprocessing)
- **Languages**: Primarily English
- **Error types**: Grammar, spelling, fluency, coherence
- **Format**: Source-target pairs (incorrect → correct)

## 🚀 Features

- **T5-based Architecture**: Leverages pre-trained T5-base model
- **Comprehensive Training**: Custom training loop with progress tracking
- **Multi-metric Evaluation**: BLEU, ROUGE-L, accuracy, and R² scores
- **Robust Inference**: Error-handling and beam search optimization
- **Data Pipeline**: Complete preprocessing and format conversion
- **GPU Support**: Automatic GPU detection and memory growth configuration
- **Self-training**: Generates additional training data from corrections

## 📋 Requirements

```python
tensorflow>=2.8.0
transformers>=4.20.0
pandas>=1.5.0
scikit-learn>=1.1.0
tqdm>=4.64.0
nltk>=3.7
rouge-score>=0.1.2
numpy>=1.21.0
```

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd gec_model
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (for evaluation)
```python
import nltk
nltk.download('punkt')
```

## 🎯 Usage

### Training

Run the training script to train the GEC model:

```bash
python train_gec_t5.py
```

**Training Configuration:**
- **Model**: T5-base (220M parameters)
- **Batch Size**: 16
- **Learning Rate**: 3e-4
- **Epochs**: 10
- **Max Length**: 64 tokens
- **Optimizer**: Adam

### Inference

Test the trained model on new sentences:

```bash
python test_inference.py
```

**Example Usage:**
```python
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Load model
tokenizer = T5Tokenizer.from_pretrained("./gec_model")
model = TFT5ForConditionalGeneration.from_pretrained("./gec_model")

# Correct grammar
def correct_grammar(sentence):
    input_ids = tokenizer("fix: " + sentence, return_tensors="tf", 
                         max_length=64, truncation=True, padding='max_length').input_ids
    output = model.generate(input_ids, max_length=64, num_beams=4)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test
print(correct_grammar("She go to school every day."))
# Output: "She goes to school every day."
```

### Data Preprocessing

Process raw datasets into training format:

```bash
python preprocessing_pipeline.py
```

Convert between data formats:

```bash
python convert_parquet_to_csv.py
```

## 📊 Model Performance

The model is evaluated using multiple metrics:

- **Sequence-level Accuracy**: Exact match between predicted and target sentences
- **Token-level Accuracy**: Individual token correction accuracy
- **BLEU Score**: N-gram overlap with reference corrections
- **ROUGE-L**: Longest common subsequence F1 score
- **R² Score**: Regression score for token-level predictions
- **Average Loss**: Training/validation loss

## 🔧 Customization

### Hyperparameters
Modify training parameters in `train_gec_t5.py`:
```python
BATCH_SIZE = 16        # Batch size
EPOCHS = 10           # Training epochs
max_length = 64       # Maximum sequence length
learning_rate = 3e-4  # Adam learning rate
```

### Model Architecture
Switch to different T5 variants:
```python
model = TFT5ForConditionalGeneration.from_pretrained("t5-small")  # 60M params
model = TFT5ForConditionalGeneration.from_pretrained("t5-large")  # 770M params
```

### Dataset Integration
Add custom datasets by modifying the data loading section:
```python
# Load your custom dataset
custom_df = pd.read_csv("your_dataset.csv")
# Ensure columns: ['source', 'target']
```

## 🐛 Troubleshooting

### Common Issues

1. **GPU Memory Error**
   - Reduce batch size or max_length
   - Enable memory growth in GPU configuration

2. **Model Loading Error**
   - Ensure model files exist in `./gec_model/`
   - Check file permissions and paths

3. **CUDA/CPU Compatibility**
   - Set `CUDA_VISIBLE_DEVICES="-1"` to force CPU usage
   - Install appropriate TensorFlow GPU version

### Performance Optimization

- **GPU Training**: Ensure CUDA-compatible GPU and drivers
- **Memory Efficiency**: Use gradient checkpointing for large models
- **Batch Processing**: Increase batch size on high-memory systems
- **Mixed Precision**: Enable FP16 for faster training

## 📈 Model Output Examples

```
Original:  "tell what mistake is me doing"
Corrected: "tell me what mistake I am making"

Original:  "She go to school every day."
Corrected: "She goes to school every day."

Original:  "He have a apple."
Corrected: "He has an apple."

Original:  "They is playing football."
Corrected: "They are playing football."
```

## 📝 File Outputs

The system generates several output files:
- **`trained_gec_model/`**: Saved model and tokenizer
- **`self_training_data.csv`**: Self-generated training examples
- **Training logs**: Console output with loss and metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers**: For the T5 model implementation
- **Dataset Contributors**: FCE, JFLEG, Lang8, W&I+LOCNESS, CoEdIT, C4 creators
- **TensorFlow Team**: For the deep learning framework
- **Research Community**: For advancing grammar error correction research

## 📚 References

1. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
2. Napoles, C., et al. (2017). "JFLEG: A Fluency Corpus and Benchmarks for Automatic Grammatical Error Correction"
3. Bryant, C., et al. (2019). "The BEA-2019 Shared Task on Grammatical Error Correction"
4. Raheja, V. & Alikaniotis, D. (2023). "CoEdIT: Text Editing by Task-Specific Instruction Tuning"

---

**Last Updated**: September 2025  
**Version**: 1.0.0  
**Author**: John Anish G