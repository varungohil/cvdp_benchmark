#!/usr/bin/env python3

"""
Anthropic Batch API Orchestrator for CVDP Benchmark

This script automates the Anthropic Message Batches API workflow:
1. Export phase: Collect all prompts using local_export mode
2. Submit batch to Anthropic Message Batches API
3. Poll for completion
4. Transform responses to import format
5. Import phase: Evaluate using local_import mode

Usage:
    python run_batch_anthropic.py -f dataset.jsonl --batch-model claude-sonnet-4-5 -n 5 --llm

Benefits:
    - 50% cost reduction compared to synchronous API calls
    - Automated end-to-end workflow
    - Supports multi-sampling for pass@k evaluation

Reference: https://platform.claude.com/docs/en/build-with-claude/batch-processing
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)

from src.config_manager import config
from src.argparse_common import add_common_arguments, add_validation_checks, clean_filename
from src.logging_util import setup_logging, cleanup_logging


# Available Claude models for batch processing
CLAUDE_MODELS = {
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-haiku-3-5": "claude-3-5-haiku-20241022",
    "claude-haiku-3": "claude-3-haiku-20240307",
    # Add more model aliases as needed
}


class AnthropicBatchOrchestrator:
    """Orchestrates the Anthropic Message Batches API workflow."""
    
    def __init__(self, api_key: str, batch_model: str = "claude-sonnet-4-5"):
        """
        Initialize the batch orchestrator.
        
        Args:
            api_key: Anthropic API key
            batch_model: Model to use for batch inference
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Resolve model alias to full model name
        self.batch_model = CLAUDE_MODELS.get(batch_model, batch_model)
        print(f"Using model: {self.batch_model}")
    
    def load_prompts(self, input_file: str) -> List[Dict]:
        """
        Load and deduplicate prompts from exported file.
        
        Args:
            input_file: Path to the exported prompts JSONL file
            
        Returns:
            List of unique prompt dictionaries
        """
        prompts = []
        seen_ids = set()
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                data = json.loads(line)
                problem_id = data['id']
                
                if problem_id not in seen_ids:
                    seen_ids.add(problem_id)
                    prompts.append(data)
        
        print(f"Loaded {len(prompts)} unique prompts from {input_file}")
        return prompts
    
    def create_batch_requests(
        self, 
        prompts: List[Dict],
        n_samples: int = 1,
        max_tokens: int = 8192
    ) -> List[Request]:
        """
        Create Anthropic batch request objects from prompts.
        
        Args:
            prompts: List of prompt dictionaries
            n_samples: Number of completions per prompt (for multi-sampling)
            max_tokens: Maximum tokens for response
            
        Returns:
            List of Request objects for the batch
        """
        requests = []
        
        for prompt_data in prompts:
            problem_id = prompt_data['id']
            system_prompt = prompt_data.get('system', '')
            user_prompt = prompt_data.get('user', '')
            
            # Create n_samples requests for each problem
            for sample_idx in range(n_samples):
                custom_id = f"{problem_id}__sample_{sample_idx + 1}"
                
                # Build the request params
                params = MessageCreateParamsNonStreaming(
                    model=self.batch_model,
                    max_tokens=max_tokens,
                    messages=[{
                        "role": "user",
                        "content": user_prompt
                    }]
                )
                
                # Add system message if present
                if system_prompt:
                    params["system"] = system_prompt
                
                # Add temperature for sampling diversity (only if n_samples > 1)
                if n_samples > 1:
                    params["temperature"] = 0.7
                
                request = Request(
                    custom_id=custom_id,
                    params=params
                )
                requests.append(request)
        
        print(f"Created {len(requests)} batch requests ({len(prompts)} problems x {n_samples} samples)")
        return requests
    
    def submit_batch(
        self, 
        requests: List[Request],
        description: str = ""
    ) -> str:
        """
        Submit a batch to Anthropic.
        
        Args:
            requests: List of Request objects
            description: Optional description for the batch
            
        Returns:
            Batch ID
        """
        print(f"Submitting batch with {len(requests)} requests...")
        
        # Anthropic batch API has a limit of 100,000 requests or 256 MB
        if len(requests) > 100000:
            raise ValueError(f"Batch size {len(requests)} exceeds maximum of 100,000 requests")
        
        message_batch = self.client.messages.batches.create(
            requests=requests
        )
        
        batch_id = message_batch.id
        print(f"Created batch with ID: {batch_id}")
        print(f"Processing status: {message_batch.processing_status}")
        
        return batch_id
    
    def poll_batch_status(
        self, 
        batch_id: str, 
        poll_interval: int = 60,
        timeout: int = 86400  # 24 hours
    ) -> Dict[str, Any]:
        """
        Poll batch status until completion.
        
        Args:
            batch_id: Batch ID to poll
            poll_interval: Seconds between polls
            timeout: Maximum seconds to wait
            
        Returns:
            dict: Final batch status
        """
        print(f"Polling batch status (interval: {poll_interval}s, timeout: {timeout}s)...")
        
        start_time = time.time()
        last_status = None
        
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            
            # Print status updates
            if status != last_status or True:  # Always print for progress
                counts = batch.request_counts
                processing = counts.processing if counts else 0
                succeeded = counts.succeeded if counts else 0
                errored = counts.errored if counts else 0
                canceled = counts.canceled if counts else 0
                expired = counts.expired if counts else 0
                total = processing + succeeded + errored + canceled + expired
                
                elapsed = int(time.time() - start_time)
                print(f"[{elapsed}s] Status: {status} | "
                      f"Succeeded: {succeeded}/{total} | "
                      f"Errored: {errored} | "
                      f"Processing: {processing}")
                last_status = status
            
            # Check terminal states
            if status == "ended":
                counts = batch.request_counts
                print(f"\nBatch completed!")
                print(f"  Succeeded: {counts.succeeded}")
                print(f"  Errored: {counts.errored}")
                print(f"  Canceled: {counts.canceled}")
                print(f"  Expired: {counts.expired}")
                
                return {
                    "status": status,
                    "results_url": batch.results_url,
                    "request_counts": {
                        "succeeded": counts.succeeded,
                        "errored": counts.errored,
                        "canceled": counts.canceled,
                        "expired": counts.expired
                    }
                }
            elif status == "canceling":
                print("Batch is being canceled...")
            
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Batch did not complete within {timeout}s")
            
            time.sleep(poll_interval)
    
    def retrieve_results(
        self, 
        batch_id: str,
        output_file: str,
        n_samples: int = 1
    ) -> int:
        """
        Retrieve and transform batch results to import format.
        
        Args:
            batch_id: Batch ID to retrieve results for
            output_file: Path to write import-formatted responses
            n_samples: Number of samples (for context)
            
        Returns:
            int: Number of successful responses
        """
        print(f"Retrieving results for batch: {batch_id}")
        
        responses = []
        errors = []
        
        # Stream results from the batch
        for result in self.client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            
            # Parse custom_id to get problem_id and sample index
            if '__sample_' in custom_id:
                problem_id, sample_part = custom_id.rsplit('__sample_', 1)
                sample_idx = int(sample_part) - 1
            else:
                problem_id = custom_id
                sample_idx = 0
            
            result_type = result.result.type
            
            if result_type == "succeeded":
                # Extract text content from message
                message = result.result.message
                content_parts = []
                for block in message.content:
                    if hasattr(block, 'text'):
                        content_parts.append(block.text)
                
                completion = "\n".join(content_parts)
                
                responses.append({
                    'id': problem_id,
                    'completion': completion,
                    'sample_index': sample_idx
                })
            elif result_type == "errored":
                error = result.result.error
                error_type = error.type if hasattr(error, 'type') else 'unknown'
                error_msg = str(error)
                errors.append({
                    'custom_id': custom_id,
                    'error_type': error_type,
                    'error': error_msg
                })
                print(f"Warning: Request {custom_id} errored: {error_type}")
            elif result_type == "canceled":
                print(f"Warning: Request {custom_id} was canceled")
            elif result_type == "expired":
                print(f"Warning: Request {custom_id} expired")
        
        # Sort by problem_id and sample_index for consistent ordering
        responses.sort(key=lambda x: (x['id'], x['sample_index']))
        
        # Write import format
        with open(output_file, 'w', encoding='utf-8') as f:
            for resp in responses:
                f.write(json.dumps({
                    'id': resp['id'],
                    'completion': resp['completion']
                }) + '\n')
        
        # Save errors if any
        if errors:
            error_file = output_file.replace('.jsonl', '_errors.jsonl')
            with open(error_file, 'w', encoding='utf-8') as f:
                for err in errors:
                    f.write(json.dumps(err) + '\n')
            print(f"Errors saved to: {error_file}")
        
        num_problems = len(set(r['id'] for r in responses))
        print(f"Retrieved {len(responses)} responses for {num_problems} problems")
        
        return len(responses)


def run_export_phase(args: argparse.Namespace, prompts_file: str) -> bool:
    """
    Run the export phase to collect prompts.
    
    Args:
        args: Parsed command-line arguments
        prompts_file: Path to save exported prompts
        
    Returns:
        bool: True if successful
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Exporting prompts")
    print("=" * 60)
    
    # Clear the prompts file if it exists
    if os.path.exists(prompts_file):
        os.remove(prompts_file)
        print(f"Cleared existing prompts file: {prompts_file}")
    
    # Use run_benchmark.py directly with n=1 to get unique prompts
    cmd = ["python", "run_benchmark.py"]
    
    cmd.extend(["-f", args.filename])
    cmd.extend(["--model", "local_export"])
    cmd.extend(["--prompts-responses-file", prompts_file])
    cmd.extend(["-p", os.path.join(args.prefix, "export_run")])
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
        print(f"Export phase failed: {e}")
        return False


def run_import_phase(args: argparse.Namespace, responses_file: str) -> bool:
    """
    Run the import phase to evaluate responses.
    
    Args:
        args: Parsed command-line arguments
        responses_file: Path to responses file
        
    Returns:
        bool: True if successful
    """
    print("\n" + "=" * 60)
    print("PHASE 4: Importing responses and evaluating")
    print("=" * 60)
    
    cmd = ["python", "run_samples.py"]
    
    cmd.extend(["-f", args.filename])
    cmd.extend(["-n", str(args.n_samples)])
    cmd.extend(["-k", str(args.k_threshold)])
    cmd.extend(["--model", "local_import"])
    cmd.extend(["--prompts-responses-file", responses_file])
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
        print(f"Import phase failed: {e}")
        return False


def run_batch_workflow(args: argparse.Namespace) -> None:
    """
    Run the complete Anthropic batch API workflow.
    
    Args:
        args: Parsed command-line arguments
    """
    # Create batch working directory
    batch_dir = os.path.join(args.prefix, "batch_files")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Define file paths
    prompts_file = os.path.join(batch_dir, "exported_prompts.jsonl")
    responses_file = os.path.join(batch_dir, "responses.jsonl")
    
    # Get API key
    api_key = config.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: No Anthropic API key found. Set ANTHROPIC_API_KEY in .env or environment variable.")
        sys.exit(1)
    
    # Initialize orchestrator
    orchestrator = AnthropicBatchOrchestrator(api_key, args.batch_model)
    
    # Check if we should resume from a specific phase
    if args.resume_from:
        phase = args.resume_from.lower()
        if phase == "submit":
            print(f"Resuming from submit phase (using existing {prompts_file})")
        elif phase == "poll":
            if not args.batch_id:
                print("Error: --batch-id required when resuming from poll phase")
                sys.exit(1)
            print(f"Resuming from poll phase (batch ID: {args.batch_id})")
        elif phase == "import":
            print(f"Resuming from import phase (using existing {responses_file})")
        else:
            print(f"Error: Unknown resume phase: {phase}")
            sys.exit(1)
    else:
        phase = None
    
    batch_id = args.batch_id
    
    # Phase 1: Export prompts
    if not phase or phase == "export":
        if not run_export_phase(args, prompts_file):
            sys.exit(1)
        
        if not os.path.exists(prompts_file):
            print(f"Error: Export did not create prompts file: {prompts_file}")
            sys.exit(1)
    
    # Phase 2: Create and submit batch
    if not phase or phase in ["export", "submit"]:
        print("\n" + "=" * 60)
        print("PHASE 2: Submitting to Anthropic Message Batches API")
        print("=" * 60)
        
        # Load prompts
        prompts = orchestrator.load_prompts(prompts_file)
        
        if not prompts:
            print("Error: No prompts to submit")
            sys.exit(1)
        
        # Create batch requests
        requests = orchestrator.create_batch_requests(
            prompts,
            n_samples=args.n_samples,
            max_tokens=args.max_tokens
        )
        
        # Submit batch
        batch_id = orchestrator.submit_batch(
            requests,
            description=f"CVDP Benchmark - {os.path.basename(args.filename)}"
        )
        
        # Save batch info for potential resume
        batch_info_file = os.path.join(batch_dir, "batch_info.json")
        with open(batch_info_file, 'w') as f:
            json.dump({
                "batch_id": batch_id,
                "created_at": datetime.now().isoformat(),
                "model": orchestrator.batch_model,
                "n_samples": args.n_samples,
                "num_requests": len(requests)
            }, f, indent=2)
        print(f"Batch info saved to: {batch_info_file}")
    
    # Phase 3: Poll for completion
    if not phase or phase in ["export", "submit", "poll"]:
        print("\n" + "=" * 60)
        print("PHASE 3: Waiting for batch completion")
        print("=" * 60)
        
        if not batch_id:
            # Try to load from batch_info.json
            batch_info_file = os.path.join(batch_dir, "batch_info.json")
            if os.path.exists(batch_info_file):
                with open(batch_info_file, 'r') as f:
                    batch_info = json.load(f)
                    batch_id = batch_info.get('batch_id')
        
        if not batch_id:
            print("Error: No batch ID available. Use --batch-id or run from submit phase.")
            sys.exit(1)
        
        result = orchestrator.poll_batch_status(
            batch_id,
            poll_interval=args.poll_interval,
            timeout=args.batch_timeout
        )
        
        # Retrieve and transform results
        print("\n" + "-" * 40)
        print("Retrieving results...")
        print("-" * 40)
        
        num_responses = orchestrator.retrieve_results(
            batch_id,
            responses_file,
            n_samples=args.n_samples
        )
        
        if num_responses == 0:
            print("Error: No successful responses from batch")
            sys.exit(1)
    
    # Phase 4: Import and evaluate
    if not run_import_phase(args, responses_file):
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("BATCH WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.prefix}")


def main():
    parser = argparse.ArgumentParser(
        description="Run CVDP benchmark using Anthropic Message Batches API for 50% cost savings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark with 5 samples using Claude Sonnet 4.5
  python run_batch_anthropic.py -f dataset.jsonl --batch-model claude-sonnet-4-5 -n 5 --llm
  
  # Run single problem with Claude Haiku 3.5
  python run_batch_anthropic.py -f dataset.jsonl --batch-model claude-haiku-3-5 -i problem_0001 --llm
  
  # Resume from polling phase (if interrupted)
  python run_batch_anthropic.py -f dataset.jsonl --resume-from poll --batch-id msgbatch_abc123

Available model aliases:
  claude-sonnet-4-5  -> claude-sonnet-4-5-20250514
  claude-sonnet-4    -> claude-sonnet-4-20250514
  claude-opus-4      -> claude-opus-4-20250514
  claude-haiku-3-5   -> claude-3-5-haiku-20241022
  claude-haiku-3     -> claude-3-haiku-20240307
        """
    )
    
    # Batch-specific arguments
    parser.add_argument("--batch-model", type=str, default="claude-sonnet-4-5",
                       help="Model to use for batch inference (default: claude-sonnet-4-5)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                       help="Maximum tokens for model response (default: 8192)")
    parser.add_argument("--poll-interval", type=int, default=60,
                       help="Seconds between batch status polls (default: 60)")
    parser.add_argument("--batch-timeout", type=int, default=86400,
                       help="Maximum seconds to wait for batch completion (default: 86400 = 24h)")
    parser.add_argument("--resume-from", type=str, choices=["submit", "poll", "import"],
                       help="Resume from a specific phase")
    parser.add_argument("--batch-id", type=str,
                       help="Batch ID to resume polling (required with --resume-from poll)")
    
    # Sample arguments
    parser.add_argument("-n", "--n-samples", type=int, default=1,
                       help="Number of samples to run (default: 1)")
    parser.add_argument("-k", "--k-threshold", type=int, default=1,
                       help="Pass@k threshold (default: 1)")
    
    # Add common arguments
    add_common_arguments(parser)
    
    args = parser.parse_args()
    
    # Force LLM mode for batch workflow
    args.llm = True
    
    # Clear model args for validation
    args.model = None
    args.prompts_responses_file = None
    
    # Clean up filename
    args.filename = clean_filename(args.filename)
    
    # Set up logging
    base_prefix = args.prefix or config.get("BENCHMARK_PREFIX")
    os.makedirs(base_prefix, exist_ok=True)
    setup_logging(base_prefix)
    
    print("=" * 60)
    print("CVDP Benchmark - Anthropic Message Batches API Mode")
    print("=" * 60)
    print(f"Dataset: {args.filename}")
    print(f"Model: {args.batch_model}")
    print(f"Samples: {args.n_samples}")
    print(f"Pass@k threshold: {args.k_threshold}")
    print(f"Output prefix: {args.prefix}")
    print("=" * 60)
    
    # Run the workflow
    run_batch_workflow(args)


if __name__ == "__main__":
    main()
