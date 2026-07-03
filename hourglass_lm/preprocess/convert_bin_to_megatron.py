#!/usr/bin/env python3
"""
Convert raw .bin token files to Megatron IndexedDataset format.

Each input .bin file is treated as a single document (a flat array of token IDs).
The output is a single Megatron-compatible dataset with .bin + .idx files.

Usage:
    # Convert all .bin files in a directory
    python convert_bin_to_megatron.py \
        --input-dir /path/to/olmo-stage1/data/ \
        --output-prefix /path/to/output/merged_data \
        --dtype uint32

    # Specify individual files
    python convert_bin_to_megatron.py \
        --input-files /path/to/data_chunks_0.bin /path/to/data_chunks_1.bin \
        --output-prefix /path/to/output/merged_data

    # Preview without converting
    python convert_bin_to_megatron.py \
        --input-dir /path/to/data/ \
        --output-prefix /tmp/test \
        --dry-run
"""

import argparse
import glob
import os
import re
import sys
import numpy as np

# Add Megatron-LM to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEGATRON_LM_PATH = os.path.join(SCRIPT_DIR, "..", "Megatron-Bridge", "3rdparty", "Megatron-LM")
if os.path.exists(MEGATRON_LM_PATH):
    sys.path.insert(0, MEGATRON_LM_PATH)

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder


# Input dtype (what the raw .bin files store)
DTYPE_MAP = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}

# Megatron DType enum only supports: uint8, int8, int16, int32, int64, float64, float32, uint16
# Map unsupported input dtypes to the closest supported Megatron dtype
MEGATRON_DTYPE_MAP = {
    np.uint32: np.int32,  # uint32 -> int32 (vocab size < 2^31, safe)
}


def natural_sort_files(files: list) -> list:
    """Sort files by the numerical INDEX in patterns like 'data_chunks_{INDEX}.bin'.
    
    Examples:
        data_chunks_0.bin -> INDEX 0
        data_chunks_10.bin -> INDEX 10
        data_chunks_9.bin -> INDEX 9
    
    Falls back to alphabetical sort if the pattern is not recognized.
    """
    def extract_index(filepath: str) -> tuple:
        """Extract the numerical index from a filename.
        
        Returns a tuple (index, filepath) where index is an int,
        or (float('inf'), filepath) if extraction fails (for fallback sorting).
        """
        basename = os.path.basename(filepath)
        # Try to match pattern like "data_chunks_<NUMBER>.bin"
        match = re.search(r'_(\d+)\.bin$', basename)
        if match:
            return (int(match.group(1)), filepath)
        # Fallback: sort by filename
        return (float('inf'), filepath)
    
    sorted_files = sorted(files, key=extract_index)
    return sorted_files


def peek_file(filepath: str, dtype: np.dtype, num_tokens: int = 20):
    """Preview the first few tokens of a .bin file."""
    data = np.memmap(filepath, dtype=dtype, mode='r')
    print(f"  File: {os.path.basename(filepath)}")
    print(f"  Total tokens: {len(data):,}")
    print(f"  First {num_tokens} tokens: {data[:num_tokens].tolist()}")
    print(f"  Token range: [{data.min()}, {data.max()}]")
    return len(data)


def convert(input_files: list, output_prefix: str, dtype: np.dtype, seq_len: int = 2048, dry_run: bool = False):
    """Convert raw .bin files to Megatron IndexedDataset format.
    
    Args:
        input_files: List of .bin file paths (each is one document)
        output_prefix: Output path prefix (will create .bin and .idx)
        dtype: numpy dtype of token IDs in the input files
        seq_len: Sequence length for splitting tokens (default: 2048)
        dry_run: If True, only preview without converting
    """
    print(f"\n{'='*80}")
    print(f"Converting {len(input_files)} .bin files to Megatron IndexedDataset")
    print(f"{'='*80}")
    print(f"  Input dtype: {dtype}")
    print(f"  Output prefix: {output_prefix}")
    print(f"  Output files: {output_prefix}.bin, {output_prefix}.idx")
    print()
    
    # Natural sort by numerical INDEX in filenames (e.g., data_chunks_0, data_chunks_10, ...)
    # This ensures files are processed in the correct order, matching OLMo's ordering.
    input_files = natural_sort_files(input_files)
    print(f"  File order after natural sort:")
    for i, f in enumerate(input_files[:10]):
        print(f"    {i}: {os.path.basename(f)}")
    if len(input_files) > 10:
        print(f"    ... and {len(input_files) - 10} more files")
    print()
    
    total_tokens = 0
    total_docs = 0
    
    if dry_run:
        print("--- DRY RUN: Preview only ---\n")
        for f in input_files[:5]:  # Preview first 5 files
            n = peek_file(f, dtype)
            total_tokens += n
            total_docs += 1
            print()
        
        if len(input_files) > 5:
            # Count remaining files
            for f in input_files[5:]:
                data = np.memmap(f, dtype=dtype, mode='r')
                total_tokens += len(data)
                total_docs += 1
            print(f"  ... and {len(input_files) - 5} more files")
        
        print(f"\nSummary:")
        print(f"  Total documents: {total_docs}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Estimated output size: ~{total_tokens * np.dtype(dtype).itemsize / 1e9:.2f} GB")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Map to Megatron-supported dtype (e.g. uint32 -> int32)
    input_dtype = dtype
    megatron_dtype = MEGATRON_DTYPE_MAP.get(dtype, dtype)
    if megatron_dtype != input_dtype:
        print(f"  Note: Input dtype {np.dtype(input_dtype)} -> Megatron dtype {np.dtype(megatron_dtype)}")
    
    # Build the Megatron IndexedDataset
    builder = IndexedDatasetBuilder(
        bin_path=f"{output_prefix}.bin",
        dtype=megatron_dtype,
    )
    
    print(f"  Sequence length: {seq_len}")
    print()
    
    for i, filepath in enumerate(input_files):
        # Read the raw token IDs in original dtype, then cast to Megatron dtype
        data = np.memmap(filepath, dtype=input_dtype, mode='r')
        tokens = np.array(data, dtype=megatron_dtype)  # Cast to Megatron-supported dtype
        
        # Split into sequences of seq_len, drop the remainder
        num_seqs = len(tokens) // seq_len
        usable_tokens = num_seqs * seq_len
        tokens = tokens[:usable_tokens]
        
        # 1 file = 1 document, containing num_seqs sequences of length seq_len
        lengths = [seq_len] * num_seqs
        builder.add_document(tokens, lengths=lengths)
        
        total_tokens += len(tokens)
        total_docs += 1
        
        if (i + 1) % 10 == 0 or (i + 1) == len(input_files):
            print(f"  Processed {i+1}/{len(input_files)} files "
                  f"({total_tokens:,} tokens, {total_docs} documents)")
    
    # Finalize — this creates the .idx file
    builder.finalize(f"{output_prefix}.idx")
    
    print(f"\n{'='*80}")
    print(f"Conversion complete!")
    print(f"{'='*80}")
    print(f"  Documents: {total_docs}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Total sequences (tokens / {seq_len}): {total_tokens // seq_len:,}")
    print(f"  Output: {output_prefix}.bin ({os.path.getsize(f'{output_prefix}.bin') / 1e9:.2f} GB)")
    print(f"  Index:  {output_prefix}.idx ({os.path.getsize(f'{output_prefix}.idx') / 1e6:.2f} MB)")
    print()
    print(f"To use in training, set in YAML:")
    print(f"  dataset:")
    print(f"    blend:")
    print(f"      - {output_prefix}")
    print(f'    split: "99,1,0"')


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw .bin token files to Megatron IndexedDataset format",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing .bin files to convert"
    )
    group.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        help="Individual .bin files to convert"
    )
    
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output path prefix (creates <prefix>.bin and <prefix>.idx)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint32",
        choices=list(DTYPE_MAP.keys()),
        help="Data type of token IDs in input .bin files (default: uint32)"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Sequence length to split tokens into (default: 2048)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files without converting"
    )
    
    args = parser.parse_args()
    
    pattern = ["*.bin", "*.npy"]
    input_files = []
    # Collect input files
    if args.input_dir:
        for p in pattern:
            input_files.extend(glob.glob(os.path.join(args.input_dir, "**", p), recursive=True))
        if not input_files:
            print(f"Error: No files matching '{args.pattern}' found in {args.input_dir}")
            sys.exit(1)
        # Sort by numerical INDEX in filenames (natural sort) to match OLMo ordering
        input_files = natural_sort_files(input_files)
        print(f"Found {len(input_files)} files in {args.input_dir}")
    else:
        input_files = args.input_files
        # Also naturally sort explicitly provided files
        input_files = natural_sort_files(input_files)
        for f in input_files:
            if not os.path.exists(f):
                print(f"Error: File not found: {f}")
                sys.exit(1)
    
    dtype = DTYPE_MAP[args.dtype]
    convert(input_files, args.output_prefix, dtype, args.sequence_length, args.dry_run)


if __name__ == "__main__":
    main()
