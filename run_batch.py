#!/usr/bin/env python3

"""
Batch API Orchestrator for CVDP Benchmark

This script automates the OpenAI Batch API workflow:
1. Export phase: Collect all prompts using local_export mode
2. Transform prompts to OpenAI Batch API format
3. Submit batch to OpenAI
4. Poll for completion
5. Transform responses to import format
6. Import phase: Evaluate using local_import mode

Usage:
    python run_batch.py -f dataset.jsonl --batch-model gpt-4o -n 5 --llm

Benefits:
    - 50% cost reduction compared to synchronous API calls
    - Automated end-to-end workflow
    - Supports multi-sampling for pass@k evaluation
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import openai

from src.config_manager import config
from src.argparse_common import add_common_arguments, add_validation_checks, clean_filename
from src.logging_util import setup_logging, cleanup_logging


class BatchAPIOrchestrator:
    """Orchestrates the OpenAI Batch API workflow."""
    
    def __init__(self, api_key: str, batch_model: str = "gpt-4o"):
        """
        Initialize the batch orchestrator.
        
        Args:
            api_key: OpenAI API key
            batch_model: Model to use for batch inference (e.g., gpt-4o, gpt-4o-mini)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.batch_model = batch_model
        
    def transform_prompts_to_batch_format(
        self, 
        input_file: str, 
        output_file: str,
        n_samples: int = 1,
        logprobs: bool = False,
        top_logprobs: int = 5
    ) -> int:
        """
        Transform exported prompts to OpenAI Batch API format.
        
        Args:
            input_file: Path to the exported prompts JSONL file
            output_file: Path to write the batch-formatted JSONL file
            n_samples: Number of completions per prompt (for multi-sampling)
            logprobs: Whether to include log probabilities in response
            top_logprobs: Number of most likely tokens to return (1-20)
            
        Returns:
            int: Number of requests created
        """
        requests = []
        seen_problem_ids = set()  # Track unique problem IDs to avoid duplicates
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                data = json.loads(line)
                problem_id = data['id']
                
                # Skip duplicate problem IDs (can happen if export ran multiple times)
                if problem_id in seen_problem_ids:
                    continue
                seen_problem_ids.add(problem_id)
                
                system_prompt = data.get('system', '')
                user_prompt = data.get('user', '')
                
                # Create n_samples requests for each problem (for pass@k sampling)
                for sample_idx in range(n_samples):
                    # Use custom_id to track problem_id and sample index
                    custom_id = f"{problem_id}__sample_{sample_idx + 1}"
                    
                    # Build request body
                    body = {
                        "model": self.batch_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                    }
                    
                    # Only set temperature for models that support it
                    # gpt-5, o3, o1 models only support default temperature
                    if not any(x in self.batch_model.lower() for x in ['gpt-5', 'o3', 'o1']):
                        body["temperature"] = 0.7 if n_samples > 1 else 0.0
                    
                    # Add logprobs if requested
                    if logprobs:
                        body["logprobs"] = True
                        body["top_logprobs"] = min(max(top_logprobs, 1), 20)  # Clamp to valid range 1-20
                    
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body
                    }
                    requests.append(request)
        
        # Write batch format file
        with open(output_file, 'w', encoding='utf-8') as f:
            for request in requests:
                f.write(json.dumps(request) + '\n')
        
        num_problems = len(seen_problem_ids)
        print(f"Created {len(requests)} batch requests from {num_problems} unique problems ({n_samples} samples each)")
        return len(requests)
    
    def upload_batch_file(self, file_path: str) -> str:
        """
        Upload batch file to OpenAI.
        
        Args:
            file_path: Path to the batch JSONL file
            
        Returns:
            str: File ID from OpenAI
        """
        print(f"Uploading batch file: {file_path}")
        
        with open(file_path, 'rb') as f:
            file_response = self.client.files.create(
                file=f,
                purpose="batch"
            )
        
        file_id = file_response.id
        print(f"Uploaded file with ID: {file_id}")
        return file_id
    
    def create_batch(self, file_id: str, description: str = "") -> str:
        """
        Create a batch job.
        
        Args:
            file_id: OpenAI file ID
            description: Optional description for the batch
            
        Returns:
            str: Batch ID
        """
        print("Creating batch job...")
        
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": description or f"CVDP Benchmark batch - {datetime.now().isoformat()}"
            }
        )
        
        batch_id = batch.id
        print(f"Created batch with ID: {batch_id}")
        return batch_id
    
    def poll_batch_status(
        self, 
        batch_id: str, 
        poll_interval: int = 30,
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
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            
            # Print status updates
            if status != last_status:
                completed = batch.request_counts.completed if batch.request_counts else 0
                total = batch.request_counts.total if batch.request_counts else 0
                failed = batch.request_counts.failed if batch.request_counts else 0
                
                elapsed = int(time.time() - start_time)
                print(f"[{elapsed}s] Status: {status} | Progress: {completed}/{total} | Failed: {failed}")
                last_status = status
            
            # Check terminal states
            if status == "completed":
                print("Batch completed successfully!")
                return {
                    "status": status,
                    "output_file_id": batch.output_file_id,
                    "error_file_id": batch.error_file_id,
                    "request_counts": {
                        "total": batch.request_counts.total,
                        "completed": batch.request_counts.completed,
                        "failed": batch.request_counts.failed
                    }
                }
            elif status in ["failed", "expired", "cancelled"]:
                error_msg = f"Batch {status}"
                if hasattr(batch, 'errors') and batch.errors:
                    error_msg += f": {batch.errors}"
                raise RuntimeError(error_msg)
            
            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Batch did not complete within {timeout}s")
            
            time.sleep(poll_interval)
    
    def download_results(self, output_file_id: str, output_path: str) -> None:
        """
        Download batch results from OpenAI.
        
        Args:
            output_file_id: OpenAI file ID for results
            output_path: Local path to save results
        """
        print(f"Downloading results to: {output_path}")
        
        content = self.client.files.content(output_file_id)
        
        with open(output_path, 'wb') as f:
            f.write(content.read())
        
        print("Download complete")
    
    def transform_results_to_import_format(
        self, 
        batch_results_file: str, 
        output_file: str,
        n_samples: int = 1
    ) -> int:
        """
        Transform batch results to the format expected by local_import.
        
        Args:
            batch_results_file: Path to downloaded batch results
            output_file: Path to write import-formatted responses
            n_samples: Number of samples (for organizing multi-sample responses)
            
        Returns:
            int: Number of responses processed
        """
        responses = []
        
        with open(batch_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                data = json.loads(line)
                custom_id = data['custom_id']
                
                # Parse custom_id to get problem_id and sample index
                # Format: "problem_id__sample_N"
                if '__sample_' in custom_id:
                    problem_id, sample_part = custom_id.rsplit('__sample_', 1)
                    sample_idx = int(sample_part) - 1  # Convert to 0-based
                else:
                    problem_id = custom_id
                    sample_idx = 0
                
                # Extract completion from response
                if data.get('response', {}).get('status_code') == 200:
                    body = data['response']['body']
                    if 'choices' in body and body['choices']:
                        completion = body['choices'][0]['message']['content']
                        responses.append({
                            'id': problem_id,
                            'completion': completion,
                            'sample_index': sample_idx
                        })
                else:
                    error = data.get('error', {})
                    print(f"Warning: Request {custom_id} failed: {error}")
        
        # Sort by problem_id and sample_index to ensure consistent ordering
        responses.sort(key=lambda x: (x['id'], x['sample_index']))
        
        # Write import format (multiple completions per problem for multi-sampling)
        with open(output_file, 'w', encoding='utf-8') as f:
            for resp in responses:
                # Write in the format expected by LocalInferenceModel._load_responses()
                f.write(json.dumps({
                    'id': resp['id'],
                    'completion': resp['completion']
                }) + '\n')
        
        print(f"Processed {len(responses)} responses for {len(set(r['id'] for r in responses))} problems")
        return len(responses)


def run_export_phase(args: argparse.Namespace, prompts_file: str) -> bool:
    """
    Run the export phase to collect prompts.
    
    NOTE: We run with n_samples=1 to collect unique prompts only once.
    Multi-sampling is handled in the batch transform phase by creating
    multiple requests per problem with unique custom_ids.
    
    Args:
        args: Parsed command-line arguments
        prompts_file: Path to save exported prompts
        
    Returns:
        bool: True if successful
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Exporting prompts")
    print("=" * 60)
    
    # Clear the prompts file if it exists to avoid duplicates
    if os.path.exists(prompts_file):
        os.remove(prompts_file)
        print(f"Cleared existing prompts file: {prompts_file}")
    
    # Use run_benchmark.py directly with n=1 to get unique prompts
    # Multi-sampling is handled in the batch transform phase
    cmd = ["python", "run_benchmark.py"]
    
    # Add common arguments
    cmd.extend(["-f", args.filename])
    cmd.extend(["--model", "local_export"])
    cmd.extend(["--prompts-responses-file", prompts_file])
    cmd.extend(["-p", os.path.join(args.prefix, "export_run")])
    cmd.append("--llm")
    
    # Add optional arguments
    if args.threads:
        cmd.extend(["-t", str(args.threads)])
    if args.id:
        cmd.extend(["-i", args.id])
    if args.force_agentic:
        cmd.append("--force-agentic")
    if args.force_copilot:
        cmd.append("--force-copilot")
    
    print(f"Command: {' '.join(cmd)}")
    print(f"(Running single pass to collect unique prompts; multi-sampling handled in batch phase)")
    
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
    
    # Build command for run_samples.py with local_import
    cmd = ["python", "run_samples.py"]
    
    # Add common arguments
    cmd.extend(["-f", args.filename])
    cmd.extend(["-n", str(args.n_samples)])
    cmd.extend(["-k", str(args.k_threshold)])
    cmd.extend(["--model", "local_import"])
    cmd.extend(["--prompts-responses-file", responses_file])
    cmd.extend(["-p", args.prefix])
    cmd.append("--llm")
    
    # Add optional arguments
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
    Run the complete batch API workflow.
    
    Args:
        args: Parsed command-line arguments
    """
    # Create batch working directory
    batch_dir = os.path.join(args.prefix, "batch_files")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Define file paths
    prompts_file = os.path.join(batch_dir, "exported_prompts.jsonl")
    batch_input_file = os.path.join(batch_dir, "batch_input.jsonl")
    batch_output_file = os.path.join(batch_dir, "batch_output.jsonl")
    responses_file = os.path.join(batch_dir, "responses.jsonl")
    
    # Get API key
    api_key = config.get("OPENAI_USER_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: No OpenAI API key found. Set OPENAI_USER_KEY in .env or OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Check if we should resume from a specific phase
    if args.resume_from:
        phase = args.resume_from.lower()
        if phase == "transform":
            print(f"Resuming from transform phase (using existing {prompts_file})")
        elif phase == "submit":
            print(f"Resuming from submit phase (using existing {batch_input_file})")
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
    
    # Initialize orchestrator
    orchestrator = BatchAPIOrchestrator(api_key, args.batch_model)
    
    # Phase 1: Export prompts (skip if resuming from later phase)
    if not phase or phase == "export":
        if not run_export_phase(args, prompts_file):
            sys.exit(1)
        
        if not os.path.exists(prompts_file):
            print(f"Error: Export did not create prompts file: {prompts_file}")
            sys.exit(1)
    
    # Phase 2: Transform to batch format
    if not phase or phase in ["export", "transform"]:
        print("\n" + "=" * 60)
        print("PHASE 2: Transforming to OpenAI Batch API format")
        print("=" * 60)
        
        num_requests = orchestrator.transform_prompts_to_batch_format(
            prompts_file, 
            batch_input_file,
            n_samples=args.n_samples,
            logprobs=args.logprobs,
            top_logprobs=args.top_logprobs
        )
        
        if num_requests == 0:
            print("Error: No requests to submit")
            sys.exit(1)
    
    # Phase 3: Submit and poll batch
    if not phase or phase in ["export", "transform", "submit"]:
        print("\n" + "=" * 60)
        print("PHASE 3: Submitting to OpenAI Batch API")
        print("=" * 60)
        
        # Upload file
        file_id = orchestrator.upload_batch_file(batch_input_file)
        
        # Create batch
        batch_id = orchestrator.create_batch(
            file_id, 
            description=f"CVDP Benchmark - {os.path.basename(args.filename)}"
        )
        
        # Save batch ID for potential resume
        batch_info_file = os.path.join(batch_dir, "batch_info.json")
        with open(batch_info_file, 'w') as f:
            json.dump({
                "batch_id": batch_id,
                "file_id": file_id,
                "created_at": datetime.now().isoformat(),
                "model": args.batch_model,
                "n_samples": args.n_samples,
                "logprobs": args.logprobs,
                "top_logprobs": args.top_logprobs if args.logprobs else None
            }, f, indent=2)
        print(f"Batch info saved to: {batch_info_file}")
    else:
        batch_id = args.batch_id
    
    # Poll for completion
    if not phase or phase in ["export", "transform", "submit", "poll"]:
        print("\n" + "-" * 40)
        print("Waiting for batch completion...")
        print("-" * 40)
        
        result = orchestrator.poll_batch_status(
            batch_id,
            poll_interval=args.poll_interval,
            timeout=args.batch_timeout
        )
        
        # Download results
        if result.get("output_file_id"):
            orchestrator.download_results(result["output_file_id"], batch_output_file)
        else:
            print("Error: No output file from batch")
            sys.exit(1)
        
        # Transform to import format
        orchestrator.transform_results_to_import_format(
            batch_output_file,
            responses_file,
            n_samples=args.n_samples
        )
    
    # Phase 4: Import and evaluate
    if not run_import_phase(args, responses_file):
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("BATCH WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.prefix}")


def main():
    parser = argparse.ArgumentParser(
        description="Run CVDP benchmark using OpenAI Batch API for 50% cost savings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark with 5 samples using gpt-4o
  python run_batch.py -f dataset.jsonl --batch-model gpt-4o -n 5 --llm
  
  # Run single problem with gpt-4o-mini
  python run_batch.py -f dataset.jsonl --batch-model gpt-4o-mini -i problem_0001 --llm
  
  # Resume from polling phase (if interrupted)
  python run_batch.py -f dataset.jsonl --resume-from poll --batch-id batch_abc123
        """
    )
    
    # Batch-specific arguments
    parser.add_argument("--batch-model", type=str, default="gpt-4o",
                       help="Model to use for batch inference (default: gpt-4o)")
    parser.add_argument("--poll-interval", type=int, default=30,
                       help="Seconds between batch status polls (default: 30)")
    parser.add_argument("--batch-timeout", type=int, default=86400,
                       help="Maximum seconds to wait for batch completion (default: 86400 = 24h)")
    parser.add_argument("--resume-from", type=str, choices=["transform", "submit", "poll", "import"],
                       help="Resume from a specific phase")
    parser.add_argument("--batch-id", type=str,
                       help="Batch ID to resume polling (required with --resume-from poll)")
    parser.add_argument("--logprobs", action="store_true",
                       help="Include log probabilities in the response")
    parser.add_argument("--top-logprobs", type=int, default=5,
                       help="Number of most likely tokens to return at each position (1-20, default: 5). Only used when --logprobs is set.")
    
    # Sample arguments (from run_samples.py)
    parser.add_argument("-n", "--n-samples", type=int, default=1,
                       help="Number of samples to run (default: 1)")
    parser.add_argument("-k", "--k-threshold", type=int, default=1,
                       help="Pass@k threshold (default: 1)")
    
    # Add common arguments
    add_common_arguments(parser)
    
    args = parser.parse_args()
    
    # Override model to use batch workflow (not the model arg from common args)
    # The --model arg from common args is ignored; we use --batch-model
    if args.model and args.model not in ['local_export', 'local_import']:
        print(f"Note: --model {args.model} ignored. Using --batch-model {args.batch_model} for batch inference.")
    
    # Force LLM mode for batch workflow
    args.llm = True
    
    # Apply validation (skip local inference checks since we handle that internally)
    args.model = None  # Temporarily clear for validation
    args.prompts_responses_file = None
    
    # Clean up filename
    args.filename = clean_filename(args.filename)
    
    # Set up logging
    base_prefix = args.prefix or config.get("BENCHMARK_PREFIX")
    os.makedirs(base_prefix, exist_ok=True)
    setup_logging(base_prefix)
    
    print("=" * 60)
    print("CVDP Benchmark - OpenAI Batch API Mode")
    print("=" * 60)
    print(f"Dataset: {args.filename}")
    print(f"Model: {args.batch_model}")
    print(f"Samples: {args.n_samples}")
    print(f"Pass@k threshold: {args.k_threshold}")
    print(f"Logprobs: {args.logprobs}" + (f" (top {args.top_logprobs})" if args.logprobs else ""))
    print(f"Output prefix: {args.prefix}")
    print("=" * 60)
    
    # Run the workflow
    run_batch_workflow(args)


if __name__ == "__main__":
    main()
