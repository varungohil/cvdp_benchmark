#!/usr/bin/env python3

"""
DeepSeek API Evaluation Script for CVDP Benchmark

This script runs CVDP benchmark evaluations using DeepSeek's API.
DeepSeek provides an OpenAI-compatible API, making integration straightforward.

Usage:
    # Single run with DeepSeek Chat (default)
    python run_deepseek.py -f dataset.jsonl --llm

    # Use a specific DeepSeek model
    python run_deepseek.py -f dataset.jsonl --deepseek-model deepseek-coder --llm

    # Multi-sampling for pass@k evaluation
    python run_deepseek.py -f dataset.jsonl --deepseek-model deepseek-chat -n 5 --llm

    # Run a single problem
    python run_deepseek.py -f dataset.jsonl -i problem_0001 --llm

Available models:
    - deepseek-chat (default, also aliased as deepseek-v3)
    - deepseek-coder
    - deepseek-reasoner (also aliased as deepseek-r1)

Environment:
    Set DEEPSEEK_API_KEY in your .env file or as an environment variable.

Reference: https://platform.deepseek.com/api-docs
"""

import argparse
import os
import sys
import subprocess
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.config_manager import config
from src.argparse_common import add_common_arguments, clean_filename
from src.logging_util import setup_logging, cleanup_logging
from src.llm_lib.model_factory import ModelFactory
from src.llm_lib.deepseek_llm import DeepSeek_Instance, DEEPSEEK_MODELS


class DeepSeekModelFactory(ModelFactory):
    """
    Custom model factory that adds DeepSeek model support.
    """
    
    def __init__(self):
        super().__init__()
        
        # Register DeepSeek models
        for model_name in DEEPSEEK_MODELS.keys():
            self.model_types[model_name] = self._create_deepseek_instance
        
        # Also register the full model names
        for full_name in DEEPSEEK_MODELS.values():
            if full_name not in self.model_types:
                self.model_types[full_name] = self._create_deepseek_instance
    
    def _create_deepseek_instance(self, model_name: str, context, key: Optional[str], **kwargs):
        """Create a DeepSeek model instance."""
        return DeepSeek_Instance(context=context, key=key, model=model_name)


def write_custom_factory_file(output_path: str) -> str:
    """
    Write a temporary custom factory file for use with run_benchmark.py.
    
    Args:
        output_path: Directory to write the factory file
        
    Returns:
        Path to the created factory file
    """
    factory_code = '''
# Auto-generated DeepSeek custom factory for CVDP benchmark
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_lib.model_factory import ModelFactory
from src.llm_lib.deepseek_llm import DeepSeek_Instance, DEEPSEEK_MODELS

class CustomModelFactory(ModelFactory):
    """Custom model factory with DeepSeek support."""
    
    def __init__(self):
        super().__init__()
        
        # Register DeepSeek models
        for model_name in DEEPSEEK_MODELS.keys():
            self.model_types[model_name] = self._create_deepseek_instance
        
        for full_name in DEEPSEEK_MODELS.values():
            if full_name not in self.model_types:
                self.model_types[full_name] = self._create_deepseek_instance
    
    def _create_deepseek_instance(self, model_name, context, key, **kwargs):
        return DeepSeek_Instance(context=context, key=key, model=model_name)
'''
    
    factory_path = os.path.join(output_path, "_deepseek_factory.py")
    with open(factory_path, 'w') as f:
        f.write(factory_code)
    
    return factory_path


def run_single_benchmark(args: argparse.Namespace, factory_path: str) -> bool:
    """
    Run a single benchmark evaluation.
    
    Args:
        args: Parsed command-line arguments
        factory_path: Path to the custom factory file
        
    Returns:
        bool: True if successful
    """
    cmd = ["python", "run_benchmark.py"]
    
    cmd.extend(["-f", args.filename])
    cmd.extend(["--model", args.deepseek_model])
    cmd.extend(["--custom-factory", factory_path])
    cmd.extend(["-p", args.prefix])
    cmd.append("--llm")
    
    if args.threads:
        cmd.extend(["-t", str(args.threads)])
    if args.id:
        cmd.extend(["-i", args.id])
    if args.force_agentic:
        cmd.append("--force-agentic")
    if args.force_copilot:
        cmd.append("--force-copilot")
    if args.answers:
        cmd.extend(["-a", args.answers])
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed: {e}")
        return False


def run_multi_sample_benchmark(args: argparse.Namespace, factory_path: str) -> bool:
    """
    Run multi-sample benchmark for pass@k evaluation.
    
    Args:
        args: Parsed command-line arguments
        factory_path: Path to the custom factory file
        
    Returns:
        bool: True if successful
    """
    cmd = ["python", "run_samples.py"]
    
    cmd.extend(["-f", args.filename])
    cmd.extend(["-n", str(args.n_samples)])
    cmd.extend(["-k", str(args.k_threshold)])
    cmd.extend(["--model", args.deepseek_model])
    cmd.extend(["--custom-factory", factory_path])
    cmd.extend(["-p", args.prefix])
    cmd.append("--llm")
    
    if args.threads:
        cmd.extend(["-t", str(args.threads)])
    if args.id:
        cmd.extend(["-i", args.id])
    if args.force_agentic:
        cmd.append("--force-agentic")
    if args.force_copilot:
        cmd.append("--force-copilot")
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Multi-sample benchmark failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run CVDP benchmark using DeepSeek API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run with default model (deepseek-chat)
  python run_deepseek.py -f dataset.jsonl --llm
  
  # Use DeepSeek Coder model
  python run_deepseek.py -f dataset.jsonl --deepseek-model deepseek-coder --llm
  
  # Use DeepSeek Reasoner (R1)
  python run_deepseek.py -f dataset.jsonl --deepseek-model deepseek-r1 --llm
  
  # Multi-sampling for pass@5 evaluation
  python run_deepseek.py -f dataset.jsonl -n 5 -k 5 --llm
  
  # Run single problem
  python run_deepseek.py -f dataset.jsonl -i cvdp_copilot_counter_0001 --llm

Available model aliases:
  deepseek-chat     -> DeepSeek V3 Chat model (default)
  deepseek-v3       -> DeepSeek V3 Chat model
  deepseek-coder    -> DeepSeek Coder model
  deepseek-reasoner -> DeepSeek R1 Reasoner model
  deepseek-r1       -> DeepSeek R1 Reasoner model

Environment:
  Set DEEPSEEK_API_KEY in your .env file or environment.
        """
    )
    
    # DeepSeek-specific arguments
    parser.add_argument("--deepseek-model", type=str, default="deepseek-chat",
                       help="DeepSeek model to use (default: deepseek-chat)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                       help="Maximum tokens for model response (default: 8192)")
    
    # Sample arguments for multi-sampling
    parser.add_argument("-n", "--n-samples", type=int, default=1,
                       help="Number of samples to run (default: 1, use >1 for pass@k)")
    parser.add_argument("-k", "--k-threshold", type=int, default=1,
                       help="Pass@k threshold (default: 1)")
    
    # Add common benchmark arguments
    add_common_arguments(parser)
    
    args = parser.parse_args()
    
    # Force LLM mode
    args.llm = True
    
    # Clean up filename
    args.filename = clean_filename(args.filename)
    
    # Validate DeepSeek API key
    api_key = config.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found.")
        print("Set it in your .env file or as an environment variable:")
        print("  export DEEPSEEK_API_KEY=your_api_key_here")
        print("  # or add to .env:")
        print("  DEEPSEEK_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Validate model name
    model = args.deepseek_model
    if model not in DEEPSEEK_MODELS and model not in DEEPSEEK_MODELS.values():
        print(f"Warning: Unknown DeepSeek model '{model}'. Proceeding anyway...")
        print(f"Known models: {', '.join(DEEPSEEK_MODELS.keys())}")
    
    # Set up output directory
    base_prefix = args.prefix or config.get("BENCHMARK_PREFIX")
    os.makedirs(base_prefix, exist_ok=True)
    
    # Set up logging
    setup_logging(base_prefix)
    
    # Write custom factory file
    factory_path = write_custom_factory_file(base_prefix)
    
    print("=" * 60)
    print("CVDP Benchmark - DeepSeek API Mode")
    print("=" * 60)
    print(f"Dataset: {args.filename}")
    print(f"Model: {args.deepseek_model}")
    print(f"Samples: {args.n_samples}")
    if args.n_samples > 1:
        print(f"Pass@k threshold: {args.k_threshold}")
    print(f"Output prefix: {args.prefix}")
    print("=" * 60)
    
    try:
        # Run appropriate benchmark mode
        if args.n_samples > 1:
            success = run_multi_sample_benchmark(args, factory_path)
        else:
            success = run_single_benchmark(args, factory_path)
        
        if success:
            print("\n" + "=" * 60)
            print("BENCHMARK COMPLETE")
            print("=" * 60)
            print(f"Results saved to: {args.prefix}")
        else:
            sys.exit(1)
            
    finally:
        # Clean up temporary factory file
        if os.path.exists(factory_path):
            os.remove(factory_path)
        
        cleanup_logging()


if __name__ == "__main__":
    main()
