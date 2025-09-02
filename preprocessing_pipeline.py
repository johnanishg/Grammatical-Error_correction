import os
import json
import pandas as pd
import tensorflow as tf
import numpy as np
import re
import ast
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetStats:
    """Container for dataset statistics."""
    name: str
    num_samples: int
    avg_source_length: float
    avg_target_length: float
    error_patterns: Dict[str, int]
    vocab_size: int
    unique_errors: int

class GECPreprocessingPipeline:
    """Comprehensive preprocessing pipeline for Grammar Error Correction datasets."""
    
    def __init__(self, datasets_dir: str = "datasets", output_dir: str = "processed_data"):
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.max_length = 512
        self.min_length = 5
        self.batch_size = 64
        self.buffer_size = 10000
        
        # Statistics tracking
        self.dataset_stats = {}
        self.error_patterns = defaultdict(int)
        self.vocab_counter = Counter()
        
        # TensorFlow configuration
        self._configure_tensorflow()
        
    def _configure_tensorflow(self):
        """Configure TensorFlow for optimal performance."""
        # Enable GPU memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured {len(gpus)} GPU(s) for memory growth")
            except RuntimeError as e:
                logger.warning(f"GPU configuration error: {e}")
        
        # Set mixed precision policy for better performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
    def discover_datasets(self) -> List[Tuple[str, str]]:
        """Automatically discover datasets in the datasets directory."""
        datasets = []
        for root, dirs, files in os.walk(self.datasets_dir):
            for file in files:
                file_path = Path(root) / file
                # Skip .m2 files in fce and wi+locness
                if file.endswith('.m2') and (
                    'fce' in str(file_path).lower() or 'wi+locness' in str(file_path).lower()
                ):
                    continue
                # Only add JFLEG .tsv files, not .parquet
                if 'jfleg' in str(file_path).lower() and file.endswith('.tsv'):
                    datasets.append((str(file_path), 'tsv'))
                elif file.endswith('.json'):
                    datasets.append((str(file_path), 'json'))
                elif file.endswith('.m2'):
                    datasets.append((str(file_path), 'm2'))
                elif file.endswith('.parquet') and 'jfleg' not in str(file_path).lower():
                    datasets.append((str(file_path), 'parquet'))
                elif file.endswith('.tsv') and 'jfleg' not in str(file_path).lower():
                    datasets.append((str(file_path), 'tsv'))
        # Add wi+locness test set as special case
        wi_test_path = self.datasets_dir / 'wi+locness' / 'test' / 'ABCN.test.bea19.orig'
        if wi_test_path.exists():
            datasets.append((str(wi_test_path), 'wi_test'))
        logger.info(f"Discovered {len(datasets)} dataset files")
        return datasets
    
    def read_json_dataset(self, file_path: str) -> List[Tuple[str, str]]:
        """Read a JSON dataset, handling both standard JSON and JSON Lines formats."""
        import json
        pairs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            # If data is a list or dict, process recursively
            pairs = self._process_standard_json(data)
            logger.info(f"Read {len(pairs)} pairs from standard JSON in {file_path}")
            return pairs
        except Exception as e:
            logger.info(f"Attempting to read {file_path} as JSON Lines format due to error: {e}")
            try:
                pairs = self._read_json_lines(file_path)
                logger.info(f"Read {len(pairs)} pairs from JSON Lines in {file_path}")
                return pairs
            except Exception as e2:
                logger.error(f"Error reading JSON file {file_path}: {e2}")
                return []
    
    def _process_standard_json(self, data) -> List[Tuple[str, str]]:
        # If the top-level is a list, process each item individually
        pairs = []
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    pairs.extend(self._process_json_entry(item, line_num=idx))
                else:
                    logger.warning(f"Skipping non-dict/non-list item at top-level index {idx}: {type(item)}. Item preview: {str(item)[:200]}")
            return pairs
        else:
            return self._process_json_entry(data, line_num=0)
    
    def _process_json_entry(self, entry, line_num):
        pairs = []
        if isinstance(entry, dict):
            # Handle different entry structures for learner corpora
            if 'text' in entry and 'edits' in entry:
                source = entry['text']
                edits_field = entry['edits']
                if source is None or edits_field is None:
                    logger.warning(f"Line {line_num}: Skipping entry with None source or edits.")
                    return pairs
                all_edits = []
                for edit_group in edits_field:
                    if isinstance(edit_group, list) and len(edit_group) == 2 and isinstance(edit_group[1], list):
                        for e in edit_group[1]:
                            if isinstance(e, list) and len(e) >= 3:
                                replacement = e[2] if e[2] is not None else ''
                                all_edits.append({
                                    'start': e[0],
                                    'end': e[1],
                                    'replacement': replacement
                                })
                target = self._apply_edits(source, all_edits)
                if target is None:
                    logger.warning(f"Line {line_num}: Skipping entry with None target after applying edits.")
                    return pairs
                pairs.append((source, target))
            elif 'text' in entry and 'corrections' in entry:
                source = entry['text']
                corrections = entry['corrections']
                if source is None or corrections is None:
                    logger.warning(f"Line {line_num}: Skipping entry with None source or corrections.")
                    return pairs
                if isinstance(corrections, list) and corrections:
                    target = corrections[0]
                else:
                    target = corrections
                if target is None:
                    logger.warning(f"Line {line_num}: Skipping entry with None target in corrections.")
                    return pairs
                pairs.append((source, target))
            elif 'original' in entry and 'corrected' in entry:
                source = entry['original']
                target = entry['corrected']
                if source is None or target is None:
                    logger.warning(f"Line {line_num}: Skipping entry with None original or corrected.")
                    return pairs
                pairs.append((source, target))
            else:
                logger.debug(f"Unrecognized entry format in line {line_num}: {list(entry.keys())}")
        elif isinstance(entry, list):
            for idx, item in enumerate(entry):
                # Defensive: Only process dict or list items, skip others
                if isinstance(item, (dict, list)):
                    pairs.extend(self._process_json_entry(item, line_num + idx))
                else:
                    logger.warning(f"Skipping non-dict/non-list item in list at line {line_num + idx}: {type(item)}. Item preview: {str(item)[:200]}")
        else:
            # Log the type and a preview of the entry for debugging
            preview = str(entry)
            if len(preview) > 200:
                preview = preview[:200] + '...'
            logger.warning(f"Skipping non-dict/non-list entry at line {line_num}: {type(entry)}. Entry preview: {preview}")
        return pairs

    def _read_json_lines(self, file_path: str) -> List[Tuple[str, str]]:
        """Read JSON Lines format datasets (one JSON object or list of objects per line, recursively)."""
        pairs = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    pairs.extend(self._process_json_entry(entry, line_num))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in {file_path} at line {line_num}: {e}")
                    continue
        return pairs
    
    def _apply_edits(self, text: str, edits: List[Dict]) -> str:
        if text is None or edits is None:
            logger.warning("_apply_edits called with None text or edits.")
            return None
        # Sort edits by start position in reverse order
        sorted_edits = sorted(edits, key=lambda x: x['start'], reverse=True)
        
        result = text
        for edit in sorted_edits:
            start = edit['start']
            end = edit['end']
            replacement = edit.get('replacement', '')
            result = result[:start] + replacement + result[end:]
            
        return result
    
    def read_m2_dataset(self, file_path: str) -> List[Tuple[str, str]]:
        """Read M2 format datasets."""
        pairs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                source = None
                target = None
                edits = []
                for line_num, line in enumerate(file, 1):
                    if not isinstance(line, str):
                        logger.warning(f"Line {line_num}: Skipping non-string line in M2 file.")
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('S '):
                        source = line[2:].strip()
                        target = source
                        edits = []
                    elif line.startswith('A '):
                        # Parse annotation
                        parts = line.split('|||')
                        if len(parts) >= 3:
                            indices = parts[0][2:].split()
                            if len(indices) == 2:
                                start_idx, end_idx = map(int, indices)
                                correction = parts[2]
                                error_type = parts[1] if len(parts) > 1 else 'UNK'
                                # Skip noop edits
                                if error_type != 'noop':
                                    edits.append({
                                        'start': start_idx,
                                        'end': end_idx,
                                        'correction': correction,
                                        'type': error_type
                                    })
                    elif line == '':  # Empty line signals end of entry
                        # Apply edits to current source
                        if source and edits:
                            target = self._apply_m2_edits(source, edits)
                            if target != source:  # Only add if there's a correction
                                pairs.append((source, target))
                            edits = []
                # Don't forget the last pair
                if source and edits:
                    target = self._apply_m2_edits(source, edits)
                    if target != source:  # Only add if there's a correction
                        pairs.append((source, target))
            return pairs
        except Exception as e:
            logger.error(f"Error reading M2 file {file_path}: {e}")
            return []
    
    def _apply_m2_edits(self, source: str, edits: List[Dict]) -> str:
        """Apply M2 format edits to source text."""
        if not edits:
            return source
            
        # Sort edits by start position in reverse order to avoid index shifting
        sorted_edits = sorted(edits, key=lambda x: x['start'], reverse=True)
        
        tokens = source.split()
        
        for edit in sorted_edits:
            start_idx = edit['start']
            end_idx = edit['end']
            correction = edit['correction']
            
            # Apply edit
            if start_idx <= len(tokens) and end_idx <= len(tokens):
                # Replace tokens from start_idx to end_idx with correction
                if correction == '-NONE-' or correction == '':
                    # Deletion
                    tokens = tokens[:start_idx] + tokens[end_idx:]
                else:
                    # Substitution or insertion
                    correction_tokens = correction.split() if correction else []
                    tokens = tokens[:start_idx] + correction_tokens + tokens[end_idx:]
                    
        return ' '.join(tokens)
    
    def read_parquet_dataset(self, file_path: str) -> List[Tuple[str, str]]:
        """Read Parquet format datasets."""
        try:
            df = pd.read_parquet(file_path)
            
            # Handle different column names
            source_col = None
            target_col = None
            
            for col in df.columns:
                if col.lower() in ['source', 'original', 'input', 'text', 'sentence']:
                    source_col = col
                elif col.lower() in ['target', 'corrected', 'output', 'correction', 'corrections']:
                    target_col = col
                    
            if source_col and target_col:
                pairs = []
                for _, row in df.iterrows():
                    source = row[source_col]
                    target = row[target_col]
                    
                    # Handle multiple corrections in a list
                    if isinstance(target, list):
                        # For training, we'll use the first correction
                        # In the future, this could be extended to generate multiple training pairs
                        if target:  # Check if list is not empty
                            target = target[0]
                        else:
                            continue  # Skip if corrections list is empty
                    
                    # Robust validation for source
                    if isinstance(source, pd.Series):
                        if not bool(source.to_numpy().any()):
                            continue
                        continue
                    elif isinstance(source, np.ndarray):
                        if not bool(source.any()):
                            continue
                        continue
                    elif not pd.api.types.is_scalar(source):
                        continue
                    else:
                        if pd.isna(source):
                            continue
                        if not isinstance(source, str):
                            continue
                        if source.strip() == '':
                            continue
                    # Robust validation for target
                    if isinstance(target, pd.Series):
                        if not bool(target.to_numpy().any()):
                            continue
                        continue
                    elif isinstance(target, np.ndarray):
                        if not bool(target.any()):
                            continue
                        continue
                    elif not pd.api.types.is_scalar(target):
                        continue
                    else:
                        if pd.isna(target):
                            continue
                        if not isinstance(target, str):
                            continue
                        if target.strip() == '':
                            continue
                        
                    pairs.append((str(source), str(target)))
                    
                return pairs
            else:
                logger.warning(f"Could not identify source/target columns in {file_path}. Available columns: {list(df.columns)}")
                return []
                
        except Exception as e:
            logger.error(f"Error reading Parquet file {file_path}: {e}")
            return []
    
    def read_tsv_dataset(self, file_path: str) -> List[Tuple[str, str]]:
        """Read TSV format datasets with enhanced handling for JFLEG, CoEdit, and other formats."""
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            logger.info(f"TSV file columns: {df.columns.tolist()}")

            # Handle different column names
            source_col = None
            target_col = None

            # CoEdit format
            if set(['original_instruction', 'input', 'output']).issubset(df.columns):
                source_col = 'input'
                target_col = 'output'
            else:
                for col in df.columns:
                    if col.lower() in ['source', 'original', 'input', 'text', 'sentence']:
                        source_col = col
                    elif col.lower() in ['target', 'corrected', 'output', 'correction', 'corrections']:
                        target_col = col

            if source_col and target_col:
                pairs = []
                for _, row in df.iterrows():
                    source = row[source_col]
                    target = row[target_col]

                    # Handle multiple corrections in JFLEG format
                    if isinstance(target, str) and target.strip().startswith('[') and target.strip().endswith(']'):
                        try:
                            corrections_list = ast.literal_eval(target)
                            if isinstance(corrections_list, list) and corrections_list:
                                target = corrections_list[0]
                            else:
                                continue
                        except Exception:
                            logger.warning(f"Could not parse corrections: {target}")
                            continue

                    # Robust validation for source
                    if isinstance(source, pd.Series):
                        if not bool(source.to_numpy().any()):
                            continue
                        continue
                    elif isinstance(source, np.ndarray):
                        if not bool(source.any()):
                            continue
                        continue
                    elif not pd.api.types.is_scalar(source):
                        continue
                    else:
                        if pd.isna(source):
                            continue
                        if not isinstance(source, str):
                            continue
                        if source.strip() == '':
                            continue
                    # Robust validation for target
                    if isinstance(target, pd.Series):
                        if not bool(target.to_numpy().any()):
                            continue
                        continue
                    elif isinstance(target, np.ndarray):
                        if not bool(target.any()):
                            continue
                        continue
                    elif not pd.api.types.is_scalar(target):
                        continue
                    else:
                        if pd.isna(target):
                            continue
                        if not isinstance(target, str):
                            continue
                        if target.strip() == '':
                            continue

                    pairs.append((str(source).strip(), str(target).strip()))

                return pairs
            else:
                logger.warning(f"Could not identify source/target columns in {file_path}. Available columns: {list(df.columns)}")
                return []
        except Exception as e:
            logger.error(f"Error reading TSV file {file_path}: {e}")
            return []
    
    def read_wilocness_test(self, file_path: str) -> List[Tuple[str, str]]:
        """Read wi+locness test set as source-only (target is empty string)."""
        pairs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pairs.append((line, ''))
            return pairs
        except Exception as e:
            logger.error(f"Error reading wi+locness test file {file_path}: {e}")
            return []
    
    def extract_error_patterns(self, source: str, target: str) -> List[str]:
        """Extract error patterns from source-target pairs."""
        patterns = []
        
        # Simple pattern extraction based on word differences
        source_words = source.split()
        target_words = target.split()
        
        # Length difference pattern
        if len(source_words) != len(target_words):
            if len(source_words) > len(target_words):
                patterns.append("deletion")
            else:
                patterns.append("insertion")
        
        # Word-level substitution patterns
        for i, (src_word, tgt_word) in enumerate(zip(source_words, target_words)):
            if src_word != tgt_word:
                patterns.append(f"substitution_{src_word.lower()}_{tgt_word.lower()}")
                
                # Common error patterns
                if src_word.lower() in ['a', 'an', 'the'] or tgt_word.lower() in ['a', 'an', 'the']:
                    patterns.append("article_error")
                elif src_word.endswith('s') != tgt_word.endswith('s'):
                    patterns.append("plural_error")
                elif abs(len(src_word) - len(tgt_word)) == 1:
                    patterns.append("spelling_error")
                    
        return patterns
    
    def compute_statistics(self, data: List[Tuple[str, str]], dataset_name: str) -> DatasetStats:
        """Compute comprehensive statistics for a dataset."""
        if not data:
            return DatasetStats(dataset_name, 0, 0, 0, {}, 0, 0)
        
        source_lengths = []
        target_lengths = []
        error_patterns = defaultdict(int)
        vocab = set()
        
        for source, target in data:
            source_len = len(source.split())
            target_len = len(target.split())
            
            source_lengths.append(source_len)
            target_lengths.append(target_len)
            
            # Extract error patterns
            patterns = self.extract_error_patterns(source, target)
            for pattern in patterns:
                error_patterns[pattern] += 1
                self.error_patterns[pattern] += 1
            
            # Update vocabulary
            vocab.update(source.split())
            vocab.update(target.split())
            self.vocab_counter.update(source.split())
            self.vocab_counter.update(target.split())
        
        stats = DatasetStats(
            name=dataset_name,
            num_samples=len(data),
            avg_source_length=float(np.mean(source_lengths)),
            avg_target_length=float(np.mean(target_lengths)),
            error_patterns=dict(error_patterns),
            vocab_size=len(vocab),
            unique_errors=len(error_patterns)
        )
        
        return stats
    
    def filter_and_clean_data(self, data: List[Tuple[str, str]], dataset_name: Optional[str] = None) -> List[Tuple[str, str]]:
        """Filter and clean the dataset. For wi+locness, skip all filtering except whitespace normalization."""
        cleaned_data = []
        if dataset_name is None:
            dataset_name = ''
        is_wilocness = 'wi+locness' in dataset_name.lower()
        for source, target in data:
            # Ensure source and target are strings
            if source is None:
                source = ''
            if target is None:
                target = ''
            # Clean text
            source = re.sub(r'\s+', ' ', source.strip())
            target = re.sub(r'\s+', ' ', target.strip())
            if is_wilocness:
                # For wi+locness, do not skip any samples
                cleaned_data.append((source, target))
            else:
                # Skip if either is empty
                if not source or not target:
                    continue
                # Filter by length
                source_len = len(source.split())
                target_len = len(target.split())
                if (self.min_length <= source_len <= self.max_length and 
                    self.min_length <= target_len <= self.max_length):
                    cleaned_data.append((source, target))
        return cleaned_data
    
    def create_tensorflow_dataset(self, data: List[Tuple[str, str]]) -> tf.data.Dataset:
        """Create optimized TensorFlow dataset."""
        # Defensive: filter out any malformed entries
        filtered = []
        for pair in data:
            if (isinstance(pair, (tuple, list)) and len(pair) == 2):
                s, t = pair
                s = '' if s is None else str(s)
                t = '' if t is None else str(t)
                filtered.append((s, t))
        if not filtered:
            return tf.data.Dataset.from_tensor_slices({
                'source': [],
                'target': []
            })
        sources, targets = zip(*filtered)
        dataset = tf.data.Dataset.from_tensor_slices({
            'source': list(sources),
            'target': list(targets)
        })
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def save_processed_data(self, data: List[Tuple[str, str]], filename: str):
        """Save processed data to disk."""
        output_path = self.output_dir / filename
        
        # Save as both pickle and parquet for flexibility
        with open(output_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(data, f)
            
        df = pd.DataFrame(data, columns=['source', 'target'])
        df.to_parquet(output_path.with_suffix('.parquet'), index=False)
        
        logger.info(f"Saved processed data to {output_path}")
    
    def save_statistics(self):
        """Save comprehensive statistics."""
        stats_path = self.output_dir / 'statistics.json'
        
        # Prepare statistics for JSON serialization
        stats_data = {
            'dataset_stats': {
                name: {
                    'name': stats.name,
                    'num_samples': stats.num_samples,
                    'avg_source_length': stats.avg_source_length,
                    'avg_target_length': stats.avg_target_length,
                    'error_patterns': stats.error_patterns,
                    'vocab_size': stats.vocab_size,
                    'unique_errors': stats.unique_errors
                }
                for name, stats in self.dataset_stats.items()
            },
            'global_error_patterns': dict(self.error_patterns),
            'global_vocab_size': len(self.vocab_counter),
            'most_common_words': self.vocab_counter.most_common(100)
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
            
        logger.info(f"Saved statistics to {stats_path}")
    
    def process_all_datasets(self) -> List[Tuple[str, str]]:
        """Process all discovered datasets."""
        datasets = self.discover_datasets()
        all_data = []
        for file_path, file_type in tqdm(datasets, desc="Processing datasets"):
            logger.info(f"Processing {file_path} (type: {file_type})")
            # Read dataset based on type
            if file_type == 'json':
                data = self.read_json_dataset(file_path)
                if not data:
                    logger.warning(f"No valid data parsed from {file_path}, skipping.")
                    continue
            elif file_type == 'm2':
                data = self.read_m2_dataset(file_path)
            elif file_type == 'parquet':
                data = self.read_parquet_dataset(file_path)
            elif file_type == 'tsv':
                data = self.read_tsv_dataset(file_path)
            elif file_type == 'wi_test':
                data = self.read_wilocness_test(file_path)
            else:
                logger.warning(f"Unknown file type: {file_type}")
                continue
            # Clean and filter data (pass dataset name for special handling)
            filter_name = str(file_path)
            stats_name = Path(file_path).stem
            data = self.filter_and_clean_data(data, dataset_name=filter_name)
            # Compute statistics
            stats = self.compute_statistics(data, stats_name)
            self.dataset_stats[stats_name] = stats
            logger.info(f"Processed {len(data)} samples from {stats_name}")
            all_data.extend(data)
        # Save processed data and statistics
        self.save_processed_data(all_data, 'combined_dataset')
        self.save_statistics()
        logger.info(f"Total processed samples: {len(all_data)}")
        return all_data
    
    def create_train_val_test_split(self, data: List[Tuple[str, str]], 
                                  train_ratio: float = 0.8, 
                                  val_ratio: float = 0.1) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create train/validation/test splits as TensorFlow datasets."""
        # Shuffle data
        np.random.shuffle(data)
        
        # Calculate split indices
        n_samples = len(data)
        train_idx = int(n_samples * train_ratio)
        val_idx = int(n_samples * (train_ratio + val_ratio))
        
        # Split data
        train_data = data[:train_idx]
        val_data = data[train_idx:val_idx]
        test_data = data[val_idx:]
        
        # Create TensorFlow datasets
        train_dataset = self.create_tensorflow_dataset(train_data)
        val_dataset = self.create_tensorflow_dataset(val_data)
        test_dataset = self.create_tensorflow_dataset(test_data)
        
        # Save splits
        self.save_processed_data(train_data, 'train_dataset')
        self.save_processed_data(val_data, 'val_dataset')
        self.save_processed_data(test_data, 'test_dataset')
        
        logger.info(f"Created splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_dataset, val_dataset, test_dataset


def main():
    """Main execution function."""
    # Initialize pipeline
    pipeline = GECPreprocessingPipeline()
    
    # Process all datasets
    logger.info("Starting preprocessing pipeline...")
    all_data = pipeline.process_all_datasets()
    
    # Create train/val/test splits
    train_ds, val_ds, test_ds = pipeline.create_train_val_test_split(all_data)
    
    # Print summary
    logger.info("\n=== PREPROCESSING SUMMARY ===")
    logger.info(f"Total samples processed: {len(all_data)}")
    logger.info(f"Datasets processed: {len(pipeline.dataset_stats)}")
    logger.info(f"Global vocabulary size: {len(pipeline.vocab_counter)}")
    logger.info(f"Unique error patterns: {len(pipeline.error_patterns)}")
    
    # Print top error patterns
    logger.info("\nTop 10 error patterns:")
    for pattern, count in Counter(pipeline.error_patterns).most_common(10):
        logger.info(f"  {pattern}: {count}")
    
    logger.info("\nPreprocessing completed successfully!")
    logger.info(f"Processed data saved to: {pipeline.output_dir}")
    
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    main()
