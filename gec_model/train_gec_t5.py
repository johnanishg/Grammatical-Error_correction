import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import logging
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from sklearn.model_selection import train_test_split
import numpy as np
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    logger.info(f"GPU devices found: {len(physical_devices)}")
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.warning(f"GPU memory growth setting failed: {e}")
else:
    logger.warning("No GPU devices found, using CPU")

train_df = pd.read_parquet("processed_data/train_dataset.parquet")
val_df = pd.read_parquet("processed_data/val_dataset.parquet")
test_df = pd.read_parquet("processed_data/test_dataset.parquet")

def is_valid_string(val):
    return isinstance(val, str) and val.strip() != ''

train_pairs = []
for _, row in train_df.iterrows():
    src = row['source']
    tgt = row['target']
    if isinstance(src, str) and isinstance(tgt, str):
        src_stripped = src.strip()
        tgt_stripped = tgt.strip()
        if src_stripped and tgt_stripped and src_stripped != tgt_stripped:
            train_pairs.append({'input': src_stripped, 'target': tgt_stripped})

val_pairs = []
for _, row in val_df.iterrows():
    src = row['source']
    tgt = row['target']
    if isinstance(src, str) and isinstance(tgt, str):
        src_stripped = src.strip()
        tgt_stripped = tgt.strip()
        if src_stripped and tgt_stripped and src_stripped != tgt_stripped:
            val_pairs.append({'input': src_stripped, 'target': tgt_stripped})

test_pairs = []
for _, row in test_df.iterrows():
    src = row['source']
    tgt = row['target']
    if isinstance(src, str) and isinstance(tgt, str):
        src_stripped = src.strip()
        tgt_stripped = tgt.strip()
        if src_stripped and tgt_stripped and src_stripped != tgt_stripped:
            test_pairs.append({'input': src_stripped, 'target': tgt_stripped})

logger.info(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}, Test pairs: {len(test_pairs)}")

# 3. Tokenization
logger.info("Loading T5 tokenizer...")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
max_length = 64
BATCH_SIZE = 16

def encode(example):
    input_text = str(example['input'])
    target_text = str(example['target'])
    input_enc = tokenizer(
        text="fix: " + input_text,
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    target_enc = tokenizer(
        text=target_text,
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    return {
        'input_ids': input_enc['input_ids'],
        'attention_mask': input_enc['attention_mask'],
        'labels': target_enc['input_ids'],
    }

def df_to_tf_dataset(pairs, batch_size=32):
    records = []
    total = len(pairs)
    for idx, example in enumerate(tqdm(pairs, total=total, desc='Tokenizing')):
        records.append(encode(example))
        if (idx+1) % max(1, total // 10) == 0:
            print(f"Tokenization: {100*(idx+1)//total}% done ({idx+1}/{total})")
    def gen():
        for rec in records:
            yield (
                {
                    'input_ids': np.array(rec['input_ids'], dtype=np.int32),
                    'attention_mask': np.array(rec['attention_mask'], dtype=np.int32),
                },
                np.array(rec['labels'], dtype=np.int32)
            )
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                'input_ids': tf.TensorSpec((max_length,), tf.int32),
                'attention_mask': tf.TensorSpec((max_length,), tf.int32),
            },
            tf.TensorSpec((max_length,), tf.int32)
        )
    ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

logger.info("Preparing tf.data datasets...")
train_dataset = df_to_tf_dataset(train_pairs, batch_size=BATCH_SIZE)
val_dataset = df_to_tf_dataset(val_pairs, batch_size=BATCH_SIZE)
test_dataset = df_to_tf_dataset(test_pairs, batch_size=BATCH_SIZE)


logger.info("Loading TFT5ForConditionalGeneration model...")
model = TFT5ForConditionalGeneration.from_pretrained("t5-base")
if isinstance(model, tuple):
    model = model[0]

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=3e-4)

@tf.function
def train_step(input_ids, attention_mask, labels):
    m = model
    if isinstance(m, tuple):
        m = m[0]
    with tf.GradientTape() as tape:
        outputs = m(input_ids=input_ids, attention_mask=attention_mask, labels=labels, training=True)
        loss = outputs.loss
    gradients = tape.gradient(loss, m.trainable_variables)
    optimizer.apply_gradients(zip(gradients, m.trainable_variables))
    return loss

EPOCHS = 10

logger.info("Starting training...")
for epoch in range(EPOCHS):
    losses = []
    try:
        total_batches = tf.data.experimental.cardinality(train_dataset).numpy()
        if total_batches == tf.data.experimental.UNKNOWN_CARDINALITY:
            total_batches = None
    except:
        total_batches = None
    for batch_idx, batch in enumerate(train_dataset):
        inputs = batch[0]
        labels = batch[1]
        m = model
        if isinstance(m, tuple):
            m = m[0]
        loss = train_step(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels
        )
        if loss is not None:
            try:
                losses.append(loss.numpy())
            except Exception:
                losses.append(float(loss))
        if total_batches and (batch_idx+1) % max(1, total_batches // 10) == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}: {100*(batch_idx+1)//total_batches}% batches done ({batch_idx+1}/{total_batches})")
        elif (batch_idx+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}: Processed {batch_idx+1} batches")
    logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {np.mean(losses) if losses else 'N/A'}")

logger.info("Training complete!")


m = model
if isinstance(m, tuple):
    m = m[0]
m.save_pretrained('./trained_gec_model')
tokenizer.save_pretrained('./trained_gec_model')
logger.info("Model and tokenizer saved to ./trained_gec_model")


from sklearn.metrics import accuracy_score, r2_score

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def evaluate_model(model, tokenizer, dataset, max_length=64):
    all_preds = []
    all_labels = []
    all_token_preds = []
    all_token_labels = []
    losses = []
    for batch in tqdm(dataset, desc='Evaluating'):
        inputs = batch[0]
        labels = batch[1].numpy()
        
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length
        )
      
        outputs_for_loss = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels,
            training=False
        )
        loss = outputs_for_loss.loss
        if loss is not None:
            try:
                losses.append(loss.numpy())
            except Exception:
                losses.append(float(loss))
       
        for pred_ids, label_ids in zip(outputs, labels):
            pred_str = tokenizer.decode(pred_ids, skip_special_tokens=True)
            label_str = tokenizer.decode(label_ids, skip_special_tokens=True)
            all_preds.append(pred_str)
            all_labels.append(label_str)
           
            pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids)
            label_tokens = tokenizer.convert_ids_to_tokens(label_ids)
            # Pad to same length
            min_len = min(len(pred_tokens), len(label_tokens))
            all_token_preds.extend(pred_tokens[:min_len])
            all_token_labels.extend(label_tokens[:min_len])
    
    seq_acc = accuracy_score(all_labels, all_preds)

    token_acc = accuracy_score(all_token_labels, all_token_preds)
    
    all_token_preds_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in all_token_preds]
    all_token_labels_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in all_token_labels]
    r2 = r2_score(all_token_labels_ids, all_token_preds_ids)
    avg_loss = np.mean(losses) if losses else float('nan')

   
    references = [[ref.split()] for ref in all_labels]
    hypotheses = [pred.split() for pred in all_preds]
    bleu = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(all_labels, all_preds)]
    avg_rouge_l = np.mean(rouge_l_scores)

    print(f"\nEvaluation Results:")
    print(f"  Sequence-level Accuracy: {seq_acc:.4f}")
    print(f"  Token-level Accuracy: {token_acc:.4f}")
    print(f"  R2 Score (token-level): {r2:.4f}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  BLEU Score: {bleu:.4f}")
    print(f"  ROUGE-L F1 Score: {avg_rouge_l:.4f}")
    return {
        'seq_acc': seq_acc,
        'token_acc': token_acc,
        'r2': r2,
        'avg_loss': avg_loss,
        'bleu': bleu,
        'rougeL': avg_rouge_l
    }

logger.info("Evaluating model on test set...")
evaluate_model(m, tokenizer, test_dataset, max_length=max_length)


def correct_grammar(sentence):
    m = model
    if isinstance(m, tuple):
        m = m[0]
    input_ids = tokenizer("fix: " + sentence, return_tensors="tf", padding='max_length', truncation=True, max_length=max_length).input_ids
    output = m.generate(input_ids, max_length=max_length)
    if hasattr(output, 'sequences'):
        output_ids = tf.gather(output.sequences, 0)
    else:
        output_ids = tf.gather(output, 0)
    output_ids = tf.convert_to_tensor(output_ids).numpy()
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    return decoded

if __name__ == "__main__":
    test_sentences = [
        "She go to school every day.",
        "He have a apple.",
        "They is playing football."
    ]
    for sent in test_sentences:
        print(f"Original: {sent}")
        print(f"Corrected: {correct_grammar(sent)}\n")