"""
DeepSeek LLM implementation using OpenAI-compatible API.

DeepSeek provides an OpenAI-compatible API, so we use the openai library
with a custom base_url pointing to DeepSeek's API endpoint.

Reference: https://platform.deepseek.com/api-docs
"""

import openai
import os
import logging
import time
from typing import Optional, Any
from src.config_manager import config
from src.model_helpers import ModelHelpers

logging.basicConfig(level=logging.INFO)

# DeepSeek API configuration
DEEPSEEK_API_BASE = "https://api.deepseek.com"

# Available DeepSeek models
DEEPSEEK_MODELS = {
    "deepseek-chat": "deepseek-chat",
    "deepseek-coder": "deepseek-coder", 
    "deepseek-reasoner": "deepseek-reasoner",
    # Aliases for convenience
    "deepseek-v3": "deepseek-chat",
    "deepseek-r1": "deepseek-reasoner",
}

RETRY_CODES = [429, 502, 503, 504]
WAIT_TIME = 1.5  # Seconds to wait between API calls for error resilience
MAX_RETRIES = 3


class DeepSeek_Instance:
    """
    DeepSeek LLM implementation using OpenAI-compatible API.
    """

    def __init__(self, context: str = "You are a helpful assistant.", key=None, model=None):
        """
        Initialize the DeepSeek model instance.
        
        Args:
            context: System context/prompt for the model
            key: DeepSeek API key (optional, will use DEEPSEEK_API_KEY from env if not provided)
            model: Model name to use (default: deepseek-chat)
        """
        # Resolve model alias
        if model is None:
            model = "deepseek-chat"
        self.model = DEEPSEEK_MODELS.get(model, model)
        
        self.context = context
        self.debug = False
        
        # Get API key from parameter, config, or environment
        api_key = key or config.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        
        if api_key is None:
            raise ValueError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY in .env or environment variable."
            )
        
        # Create OpenAI client with DeepSeek base URL
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_API_BASE
        )
        
        # Initialize model helpers
        self.helper = ModelHelpers()
        
        logging.info(f"Created DeepSeek Model instance. Using model: {self.model}")
        self.set_debug(False)

    def key(self, key: str):
        """Set a new API key."""
        self.client = openai.OpenAI(
            api_key=key,
            base_url=DEEPSEEK_API_BASE
        )

    @property
    def requires_evaluation(self) -> bool:
        """Whether this model requires harness evaluation."""
        return True

    def set_debug(self, debug: bool = True) -> None:
        """Enable or disable debug mode."""
        self.debug = debug
        logging.info(f"Debug mode {'enabled' if debug else 'disabled'}")

    def prompt(self, prompt: str, schema: str = None, prompt_log: str = "", 
               files: Optional[list] = None, timeout: int = 60, category: Optional[int] = None):
        """
        Send a prompt to the DeepSeek model and get a response.
        
        Args:
            prompt: The user prompt/query
            schema: Optional JSON schema for structured output
            prompt_log: Path to log the prompt (if not empty)
            files: List of expected output files (if any)
            timeout: Timeout in seconds for the API call (default: 60)
            category: Optional integer indicating the category/problem ID
            
        Returns:
            Tuple of (parsed_response, success_bool)
        """
        if self.client is None:
            raise ValueError("DeepSeek client not initialized")

        # Create system prompt using helper
        system_prompt = self.helper.create_system_prompt(self.context, schema, category)
        
        # Use timeout from config if not specified
        if timeout == 60:
            timeout = config.get("MODEL_TIMEOUT", 60)

        # Determine if we're expecting a single file (direct text mode)
        expected_single_file = files and len(files) == 1 and schema is None
        expected_file_name = files[0] if expected_single_file else None

        if self.debug:
            logging.debug(f"Requesting prompt using the model: {self.model}")
            logging.debug(f"System prompt: {system_prompt}")
            logging.debug(f"User prompt: {prompt}")
            if files:
                logging.debug(f"Expected files: {files}")
                if expected_single_file:
                    logging.debug(f"Using direct text mode for single file: {expected_file_name}")

        # Write prompt log if requested
        if prompt_log:
            try:
                os.makedirs(os.path.dirname(prompt_log), exist_ok=True)
                temp_log = f"{prompt_log}.tmp"
                with open(temp_log, "w+") as f:
                    f.write(system_prompt + "\n\n----------------------------------------\n" + prompt)
                os.replace(temp_log, prompt_log)
            except Exception as e:
                logging.error(f"Failed to write prompt log to {prompt_log}: {str(e)}")
                raise

        # Retry logic for transient errors
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                # Create chat completion request
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    timeout=timeout,
                    max_tokens=8192,  # DeepSeek default max
                )

                if self.debug:
                    logging.debug(f"Response received:\n{response}")

                for choice in response.choices:
                    message = choice.message
                    if self.debug:
                        logging.debug(f"  - Message: {message.content}")

                    content = message.content.strip()
                    
                    # Fix common JSON formatting issues if schema is expected
                    if not expected_single_file and schema is not None:
                        if content.startswith('{') and content.endswith('}'):
                            content = self.helper.fix_json_formatting(content)

                    # Parse and return the response
                    return self.helper.parse_model_response(content, files, expected_single_file)

            except openai.RateLimitError as e:
                last_error = e
                wait_time = WAIT_TIME * (2 ** attempt)  # Exponential backoff
                logging.warning(f"Rate limited (attempt {attempt + 1}/{MAX_RETRIES}). Waiting {wait_time}s...")
                time.sleep(wait_time)
                
            except openai.APIStatusError as e:
                if hasattr(e, 'status_code') and e.status_code in RETRY_CODES:
                    last_error = e
                    wait_time = WAIT_TIME * (2 ** attempt)
                    logging.warning(f"API error {e.status_code} (attempt {attempt + 1}/{MAX_RETRIES}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise ValueError(f"DeepSeek API error: {str(e)}")
                    
            except Exception as e:
                raise ValueError(f"Unable to get response from DeepSeek model: {str(e)}")

        # If we exhausted all retries
        raise ValueError(f"DeepSeek API failed after {MAX_RETRIES} retries: {str(last_error)}")
