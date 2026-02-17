#!/usr/bin/env python3
"""
Visualize LLM output tokens colored by their token-level entropy.

Creates an HTML file where each token is highlighted based on entropy:
- Green: Low entropy (model was confident, one option dominated)
- Yellow: Medium entropy
- Orange/Red: High entropy (model was uncertain, many options similarly likely)

Token entropy measures the model's confusion at each step of generation.
Formula: H(t) = -sum(p_i * ln(p_i)) for all top K candidates
"""

import argparse
import json
import html
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def calculate_token_entropy(top_logprobs: List[Dict[str, Any]]) -> float:
    """
    Calculate token-level entropy from top_logprobs.
    
    Entropy measures the model's uncertainty. High entropy means the model
    considered many different tokens equally likely.
    
    Formula: H(t) = -sum(p_i * ln(p_i)) for top K candidates
    
    Args:
        top_logprobs: List of dicts with 'logprob' keys for top candidates
    
    Returns:
        Entropy value (0 = perfectly confident, higher = more uncertain)
    """
    if not top_logprobs:
        return 0.0
    
    # Convert logprobs to probabilities
    probs = []
    for candidate in top_logprobs:
        logprob = candidate.get("logprob", -100)
        # Clamp to avoid numerical issues
        logprob = max(logprob, -100)
        p = math.exp(logprob)
        probs.append(p)
    
    probs = np.array(probs)
    
    # Normalize probabilities to sum to 1 (handles truncation to top K)
    prob_sum = np.sum(probs)
    if prob_sum > 0:
        probs_normalized = probs / prob_sum
    else:
        return 0.0
    
    # Calculate entropy: -sum(p * log(p))
    # Avoid log(0) by filtering out zero probabilities
    entropy = 0.0
    for p in probs_normalized:
        if p > 0:
            entropy -= p * np.log(p)
    
    return float(entropy)


def entropy_to_color(entropy: float, min_entropy: float = 0.0, max_entropy: float = 2.5) -> str:
    """
    Convert an entropy value to an RGB color.
    
    Uses a gradient from green (low entropy/confident) -> yellow -> orange -> red (high entropy/uncertain).
    
    Args:
        entropy: The entropy value (0 = confident, higher = uncertain)
        min_entropy: The minimum entropy (typically 0)
        max_entropy: The maximum entropy to consider (ln(K) for K candidates, ~2.3 for 10 candidates)
    
    Returns:
        RGB color string in format "rgb(r, g, b)"
    """
    # Clamp entropy to range
    entropy = max(min_entropy, min(max_entropy, entropy))
    
    # Normalize to 0-1 range (0 = low entropy/green, 1 = high entropy/red)
    # Note: this is INVERTED from logprob - high entropy = uncertain = red
    normalized = (entropy - min_entropy) / (max_entropy - min_entropy)
    
    # Invert so that low entropy = green, high entropy = red
    normalized = 1.0 - normalized
    
    # Color gradient: red -> orange -> yellow -> green
    if normalized < 0.25:
        # Red to orange
        t = normalized / 0.25
        r, g, b = 255, int(128 * t), 128
    elif normalized < 0.5:
        # Orange to yellow
        t = (normalized - 0.25) / 0.25
        r, g, b = 255, int(128 + 127 * t), int(128 - 28 * t)
    elif normalized < 0.75:
        # Yellow to light green
        t = (normalized - 0.5) / 0.25
        r, g, b = int(255 - 55 * t), 255, int(100 + 55 * t)
    else:
        # Light green to green
        t = (normalized - 0.75) / 0.25
        r, g, b = int(200 - 50 * t), 255, int(155 - 55 * t)
    
    return f"rgb({r}, {g}, {b})"


def logprob_to_probability(logprob: float) -> float:
    """Convert logprob to probability percentage."""
    return math.exp(logprob) * 100


def extract_tokens_from_response(response: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Extract token information from a batch response.
    
    Returns list of dicts with 'token', 'logprob', and optional 'top_logprobs'.
    """
    try:
        body = response.get("response", {}).get("body", {})
        choices = body.get("choices", [])
        if not choices:
            return None
        
        logprobs_data = choices[0].get("logprobs", {})
        if not logprobs_data:
            return None
        
        content = logprobs_data.get("content", [])
        return content if content else None
    except (KeyError, IndexError, TypeError):
        return None


def get_response_metadata(response: Dict[str, Any], pass_fail_results: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    """Extract metadata from the response for display."""
    custom_id = response.get("custom_id", "unknown")
    model = response.get("response", {}).get("body", {}).get("model", "unknown")
    
    # Parse problem_id and sample from custom_id
    problem_id, sample_num = parse_custom_id(custom_id)
    
    # Look up pass/fail status
    passed = None  # None = unknown
    if pass_fail_results is not None:
        passed = pass_fail_results.get(problem_id)
    
    return {
        "custom_id": custom_id,
        "model": model,
        "problem_id": problem_id,
        "sample_num": sample_num,
        "passed": passed
    }


def generate_token_html(tokens: List[Dict[str, Any]], show_tooltip: bool = True) -> Tuple[str, List[float]]:
    """
    Generate HTML for a list of tokens with color highlighting based on entropy.
    
    Args:
        tokens: List of token dicts with 'token', 'logprob', and 'top_logprobs' keys
        show_tooltip: Whether to show tooltip with entropy/probability info on hover
    
    Returns:
        Tuple of (HTML string with styled tokens, list of entropy values)
    """
    html_parts = []
    entropies = []
    
    for token_info in tokens:
        token = token_info.get("token", "")
        logprob = token_info.get("logprob", 0.0)
        top_logprobs = token_info.get("top_logprobs", [])
        
        # Calculate entropy from top_logprobs
        entropy = calculate_token_entropy(top_logprobs)
        entropies.append(entropy)
        
        # Get color based on entropy (not logprob)
        color = entropy_to_color(entropy)
        probability = logprob_to_probability(logprob)
        
        # Escape HTML special characters
        escaped_token = html.escape(token)
        
        # Handle special characters for display
        display_token = escaped_token.replace("\n", "<br>")
        display_token = display_token.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
        display_token = display_token.replace(" ", "&nbsp;")
        
        # Build tooltip content
        tooltip_parts = [
            f"Token: {repr(token)}",
            f"Entropy: {entropy:.4f}",
            f"Logprob: {logprob:.4f}",
            f"Probability: {probability:.2f}%"
        ]
        
        # Add top alternatives if available
        if top_logprobs and len(top_logprobs) > 1:
            tooltip_parts.append("---")
            tooltip_parts.append("Top alternatives:")
            for alt in top_logprobs[:5]:
                alt_token = alt.get("token", "")
                alt_logprob = alt.get("logprob", 0.0)
                alt_prob = logprob_to_probability(alt_logprob)
                tooltip_parts.append(f"  {repr(alt_token)}: {alt_prob:.2f}%")
        
        tooltip = "&#10;".join(tooltip_parts)
        
        if show_tooltip:
            html_parts.append(
                f'<span class="token" style="background-color: {color};" '
                f'title="{tooltip}">{display_token}</span>'
            )
        else:
            html_parts.append(
                f'<span class="token" style="background-color: {color};">{display_token}</span>'
            )
    
    return "".join(html_parts), entropies


def generate_legend_html() -> str:
    """Generate HTML for the entropy color legend."""
    legend_items = []
    steps = 10
    
    for i in range(steps + 1):
        # Entropy from 0 (confident) to 2.5 (uncertain)
        entropy = 2.5 * (1 - i / steps)  # Reversed: left=high entropy, right=low
        color = entropy_to_color(entropy)
        legend_items.append(f'<div class="legend-item" style="background-color: {color};">'
                          f'{entropy:.2f}</div>')
    
    return f'''
    <div class="legend">
        <div class="legend-title">Entropy Scale (H = -Σ p·ln(p))</div>
        <div class="legend-bar">
            {"".join(legend_items)}
        </div>
        <div class="legend-labels">
            <span>High Entropy (Uncertain)</span>
            <span>Low Entropy (Confident)</span>
        </div>
        <div class="legend-note">
            Entropy measures model confusion. Low entropy = model was confident about the token choice.
            High entropy = model considered multiple alternatives equally likely.
        </div>
    </div>
    '''


def generate_html_page(responses: List[Dict[str, Any]], title: str = "Logprob Visualization",
                       pass_fail_results: Optional[Dict[str, bool]] = None) -> str:
    """
    Generate a complete HTML page with all responses visualized.
    
    Args:
        responses: List of parsed JSONL response objects
        title: Page title
        pass_fail_results: Optional dict mapping problem_id -> passed (True/False)
    
    Returns:
        Complete HTML document as string
    """
    css = '''
    <style>
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        .legend {
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            display: inline-block;
        }
        
        .legend-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        .legend-bar {
            display: flex;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .legend-item {
            padding: 8px 12px;
            font-size: 11px;
            text-align: center;
            min-width: 50px;
        }
        
        .legend-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 5px;
            font-size: 12px;
            color: #666;
        }
        
        .legend-note {
            margin-top: 10px;
            font-size: 11px;
            color: #888;
            font-style: italic;
            max-width: 500px;
        }
        
        .response {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .response-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        
        .response-id {
            font-weight: bold;
            color: #2c3e50;
            font-size: 14px;
        }
        
        .response-model {
            color: #7f8c8d;
            font-size: 12px;
            background: #ecf0f1;
            padding: 4px 10px;
            border-radius: 4px;
        }
        
        .pass-fail-badge {
            font-size: 12px;
            font-weight: bold;
            padding: 4px 12px;
            border-radius: 4px;
            margin-left: 10px;
        }
        
        .badge-pass {
            background: #27ae60;
            color: white;
        }
        
        .badge-fail {
            background: #e74c3c;
            color: white;
        }
        
        .badge-unknown {
            background: #95a5a6;
            color: white;
        }
        
        .response.passed {
            border-left: 4px solid #27ae60;
        }
        
        .response.failed {
            border-left: 4px solid #e74c3c;
        }
        
        .response-content {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.8;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .token {
            padding: 2px 0;
            border-radius: 2px;
            cursor: default;
            transition: opacity 0.2s;
        }
        
        .token:hover {
            opacity: 0.8;
            outline: 1px solid #333;
        }
        
        .stats {
            margin-top: 15px;
            padding-top: 10px;
            border-top: 1px solid #eee;
            font-size: 12px;
            color: #666;
        }
        
        .stats-item {
            display: inline-block;
            margin-right: 20px;
        }
        
        .no-logprobs {
            color: #e74c3c;
            font-style: italic;
        }
        
        .summary {
            background: #3498db;
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .summary h2 {
            margin: 0 0 10px 0;
            font-size: 18px;
        }
        
        .summary-stats {
            display: flex;
            gap: 30px;
        }
        
        .summary-stat {
            text-align: center;
        }
        
        .summary-stat-value {
            font-size: 24px;
            font-weight: bold;
        }
        
        .summary-stat-label {
            font-size: 12px;
            opacity: 0.9;
        }
        
        .controls {
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .control-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .control-group label {
            font-size: 14px;
            color: #555;
        }
        
        input[type="number"] {
            width: 80px;
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        button {
            padding: 8px 16px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        button:hover {
            background: #2980b9;
        }
        
        .filter-input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            width: 250px;
        }
        
        .filter-btn {
            padding: 6px 12px;
            margin-left: 5px;
        }
        
        .filter-btn.btn-pass {
            background: #27ae60;
        }
        
        .filter-btn.btn-pass:hover {
            background: #219a52;
        }
        
        .filter-btn.btn-fail {
            background: #e74c3c;
        }
        
        .filter-btn.btn-fail:hover {
            background: #c0392b;
        }
    </style>
    '''
    
    js = '''
    <script>
        function filterResponses() {
            const filterText = document.getElementById('filter-input').value.toLowerCase();
            const responses = document.querySelectorAll('.response');
            
            responses.forEach(resp => {
                const id = resp.querySelector('.response-id').textContent.toLowerCase();
                const content = resp.querySelector('.response-content').textContent.toLowerCase();
                
                if (id.includes(filterText) || content.includes(filterText)) {
                    resp.style.display = 'block';
                } else {
                    resp.style.display = 'none';
                }
            });
        }
        
        function jumpToResponse(index) {
            const responses = document.querySelectorAll('.response');
            if (index > 0 && index <= responses.length) {
                responses[index - 1].scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }
        
        function filterByStatus(status) {
            const responses = document.querySelectorAll('.response');
            
            responses.forEach(resp => {
                if (status === 'all') {
                    resp.style.display = 'block';
                } else if (status === 'passed') {
                    resp.style.display = resp.classList.contains('passed') ? 'block' : 'none';
                } else if (status === 'failed') {
                    resp.style.display = resp.classList.contains('failed') ? 'block' : 'none';
                }
            });
        }
    </script>
    '''
    
    # Generate response sections
    response_sections = []
    total_tokens = 0
    total_entropy_sum = 0
    responses_with_logprobs = 0
    passed_count = 0
    failed_count = 0
    unknown_count = 0
    
    for i, response in enumerate(responses, 1):
        tokens = extract_tokens_from_response(response)
        metadata = get_response_metadata(response, pass_fail_results)
        
        # Determine pass/fail status and badge
        passed = metadata.get("passed")
        if passed is True:
            badge_html = '<span class="pass-fail-badge badge-pass">PASS</span>'
            response_class = "response passed"
            passed_count += 1
        elif passed is False:
            badge_html = '<span class="pass-fail-badge badge-fail">FAIL</span>'
            response_class = "response failed"
            failed_count += 1
        else:
            badge_html = '<span class="pass-fail-badge badge-unknown">?</span>'
            response_class = "response"
            unknown_count += 1
        
        if tokens:
            responses_with_logprobs += 1
            token_html, entropies = generate_token_html(tokens)
            num_tokens = len(tokens)
            total_tokens += num_tokens
            
            # Calculate entropy stats
            avg_entropy = sum(entropies) / len(entropies) if entropies else 0
            min_entropy = min(entropies) if entropies else 0
            max_entropy = max(entropies) if entropies else 0
            total_entropy_sum += sum(entropies)
            
            # Also calculate logprob stats for reference
            logprobs = [t.get("logprob", 0) for t in tokens]
            avg_logprob = sum(logprobs) / len(logprobs)
            
            stats_html = f'''
            <div class="stats">
                <span class="stats-item"><strong>Tokens:</strong> {num_tokens}</span>
                <span class="stats-item"><strong>Avg Entropy:</strong> {avg_entropy:.4f}</span>
                <span class="stats-item"><strong>Min Entropy:</strong> {min_entropy:.4f}</span>
                <span class="stats-item"><strong>Max Entropy:</strong> {max_entropy:.4f}</span>
                <span class="stats-item"><strong>Avg Logprob:</strong> {avg_logprob:.4f}</span>
            </div>
            '''
        else:
            token_html = '<span class="no-logprobs">No logprobs available for this response</span>'
            stats_html = ''
        
        response_sections.append(f'''
        <div class="{response_class}" id="response-{i}">
            <div class="response-header">
                <span class="response-id">#{i}: {html.escape(metadata["custom_id"])}</span>
                <div>
                    {badge_html}
                    <span class="response-model">{html.escape(metadata["model"])}</span>
                </div>
            </div>
            <div class="response-content">{token_html}</div>
            {stats_html}
        </div>
        ''')
    
    # Calculate overall stats
    avg_overall_entropy = total_entropy_sum / total_tokens if total_tokens > 0 else 0
    
    pass_rate = (passed_count / (passed_count + failed_count) * 100) if (passed_count + failed_count) > 0 else 0
    
    summary_html = f'''
    <div class="summary">
        <h2>Summary</h2>
        <div class="summary-stats">
            <div class="summary-stat">
                <div class="summary-stat-value">{len(responses)}</div>
                <div class="summary-stat-label">Total Responses</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value" style="color: #2ecc71;">{passed_count}</div>
                <div class="summary-stat-label">Passed</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value" style="color: #e74c3c;">{failed_count}</div>
                <div class="summary-stat-label">Failed</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">{pass_rate:.1f}%</div>
                <div class="summary-stat-label">Pass Rate</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">{total_tokens:,}</div>
                <div class="summary-stat-label">Total Tokens</div>
            </div>
            <div class="summary-stat">
                <div class="summary-stat-value">{avg_overall_entropy:.4f}</div>
                <div class="summary-stat-label">Avg Entropy</div>
            </div>
        </div>
    </div>
    '''
    
    controls_html = '''
    <div class="controls">
        <div class="control-group">
            <label>Filter:</label>
            <input type="text" id="filter-input" class="filter-input" 
                   placeholder="Filter by ID or content..." oninput="filterResponses()">
        </div>
        <div class="control-group">
            <label>Show:</label>
            <button onclick="filterByStatus('all')" class="filter-btn">All</button>
            <button onclick="filterByStatus('passed')" class="filter-btn btn-pass">Passed</button>
            <button onclick="filterByStatus('failed')" class="filter-btn btn-fail">Failed</button>
        </div>
        <div class="control-group">
            <label>Jump to:</label>
            <input type="number" id="jump-input" min="1" placeholder="#">
            <button onclick="jumpToResponse(document.getElementById('jump-input').value)">Go</button>
        </div>
    </div>
    '''
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    {css}
</head>
<body>
    <div class="container">
        <h1>{html.escape(title)}</h1>
        {summary_html}
        {generate_legend_html()}
        {controls_html}
        {"".join(response_sections)}
    </div>
    {js}
</body>
</html>
'''
    
    return html_content


def load_jsonl(filepath: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load responses from a JSONL file."""
    responses = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                try:
                    responses.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {i+1}: {e}")
    return responses


def parse_custom_id(custom_id: str) -> Tuple[str, str]:
    """
    Parse custom_id to extract problem_id and sample_num.
    
    Example: "cvdp_copilot_64b66b_decoder_0001__sample_1" 
             -> ("cvdp_copilot_64b66b_decoder_0001", "sample_1")
    """
    if "__" in custom_id:
        parts = custom_id.rsplit("__", 1)
        return parts[0], parts[1]
    return custom_id, ""


def load_pass_fail_from_report(report_path: Path) -> Dict[str, bool]:
    """
    Load pass/fail status from a benchmark report.txt file.
    
    Returns dict mapping problem_id -> passed (True/False)
    """
    results = {}
    
    if not report_path.exists():
        print(f"Warning: Report file not found: {report_path}")
        return results
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse failing problems section
    in_failing = False
    in_passing = False
    
    for line in content.split('\n'):
        if '=== Failing Problems ===' in line:
            in_failing = True
            in_passing = False
            continue
        if '=== Passing Problems ===' in line:
            in_failing = False
            in_passing = True
            continue
        
        # Look for problem IDs in the table rows
        # Format: | 1   | cvdp_copilot_64b66b_decoder_0011 | ...
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            for part in parts:
                if part.startswith('cvdp_copilot_'):
                    # This is a problem ID
                    if in_failing:
                        results[part] = False
                    elif in_passing:
                        results[part] = True
    
    return results


def load_pass_fail_from_work_dir(work_dir: Path, sample_num: str) -> Dict[str, bool]:
    """
    Load pass/fail status from work directory structure.
    
    Looks for work_dir/sample_X/report.txt
    """
    # Try sample-specific report first
    sample_report = work_dir / sample_num / "report.txt"
    if sample_report.exists():
        return load_pass_fail_from_report(sample_report)
    
    # Fall back to generic report
    generic_report = work_dir / "report.txt"
    if generic_report.exists():
        return load_pass_fail_from_report(generic_report)
    
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LLM output tokens colored by their token-level entropy."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the JSONL file containing batch output"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output HTML file path (default: input_file.html)"
    )
    parser.add_argument(
        "-t", "--title",
        type=str,
        default="Token Entropy Visualization",
        help="Title for the HTML page"
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="Limit number of responses to process"
    )
    parser.add_argument(
        "-w", "--work-dir",
        type=Path,
        default=None,
        help="Work directory containing report.txt files for pass/fail status"
    )
    parser.add_argument(
        "-r", "--report",
        type=Path,
        default=None,
        help="Path to report.txt file for pass/fail status"
    )
    parser.add_argument(
        "-s", "--sample",
        type=str,
        default="sample_1",
        help="Sample name to use for loading reports (default: sample_1)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1
    
    # Set default output path
    if args.output is None:
        args.output = args.input_file.with_suffix('.html')
    
    print(f"Loading responses from: {args.input_file}")
    responses = load_jsonl(args.input_file, limit=args.limit)
    print(f"Loaded {len(responses)} responses")
    
    if not responses:
        print("Error: No responses found in input file")
        return 1
    
    # Load pass/fail results if available
    pass_fail_results = None
    if args.report:
        print(f"Loading pass/fail status from: {args.report}")
        pass_fail_results = load_pass_fail_from_report(args.report)
        print(f"Loaded {len(pass_fail_results)} problem results")
    elif args.work_dir:
        print(f"Loading pass/fail status from work dir: {args.work_dir} (sample: {args.sample})")
        pass_fail_results = load_pass_fail_from_work_dir(args.work_dir, args.sample)
        print(f"Loaded {len(pass_fail_results)} problem results")
    
    print(f"Generating HTML visualization...")
    html_content = generate_html_page(responses, title=args.title, pass_fail_results=pass_fail_results)
    
    print(f"Writing output to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Done! Open {args.output} in a browser to view the visualization.")
    return 0


if __name__ == "__main__":
    exit(main())
