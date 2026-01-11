"""
Model Manager - Handles LLM API interactions with retry logic and response processing.
Optimized for reliability and performance.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

# Constants
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0
JSON_CODE_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
THINK_TAG_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\n\s*\n')


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    api_key: str = ""
    api_url: str = ""
    model: str = ""
    name: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    batch_size: int = 10
    threads: int = 8
    timeout: int = 180
    is_inference: bool = False
    updated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "api_url": self.api_url,
            "model": self.model,
            "name": self.name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "batch_size": self.batch_size,
            "threads": self.threads,
            "timeout": self.timeout,
            "is_inference": self.is_inference,
            "updated": self.updated
        }

    @classmethod
    def from_env(cls, prefix: str, name: str) -> "ModelConfig":
        """Create config from environment variables."""
        return cls(
            api_key=os.getenv(f"{prefix}_API_KEY", ""),
            api_url=os.getenv(f"{prefix}_API_URL", ""),
            model=os.getenv(f"{prefix}_MODEL_NAME", ""),
            name=name,
            temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv(f"{prefix}_MAX_TOKENS", "4096")),
            batch_size=int(os.getenv(f"{prefix}_BATCH_SIZE", "10")),
            threads=int(os.getenv(f"{prefix}_THREADS", "8")),
            timeout=int(os.getenv(f"{prefix}_TIMEOUT", "180")),
            is_inference=os.getenv(f"{prefix}_IS_INFERENCE", "").lower() == "true",
            updated=False
        )


class ModelManager:
    """Manages API interactions with LLM providers."""

    __slots__ = ('_configs',)

    # Model definitions
    MODEL_DEFS = {
        "model_a": ("MODEL_A", "Model A (Primary Analyzer)"),
        "model_b": ("MODEL_B", "Model B (Critical Reviewer)"),
        "model_c": ("MODEL_C", "Model C (Final Arbitrator)")
    }

    def __init__(self):
        load_dotenv(override=True)
        self._configs: Dict[str, ModelConfig] = {}
        self._load_configs()

    def _load_configs(self) -> None:
        """Load all model configurations from environment."""
        for model_key, (prefix, name) in self.MODEL_DEFS.items():
            self._configs[model_key] = ModelConfig.from_env(prefix, name)
            if not self._configs[model_key].api_key:
                logging.warning(f"API key not found for {name}")

    def update_model_config(self, model_key: str, config: Dict[str, Any]) -> None:
        """Update model configuration."""
        if model_key not in self._configs:
            raise ValueError(f"Invalid model key: {model_key}")

        current = self._configs[model_key]
        for key, value in config.items():
            if hasattr(current, key):
                setattr(current, key, value)

    def get_config(self, model_key: str) -> Dict[str, Any]:
        """Get model configuration, refreshing from env if not manually updated."""
        load_dotenv(override=True)

        if model_key not in self._configs:
            return {}

        config = self._configs[model_key]
        if not config.updated:
            prefix, name = self.MODEL_DEFS.get(model_key, ("", ""))
            if prefix:
                self._configs[model_key] = ModelConfig.from_env(prefix, name)
                config = self._configs[model_key]

        return config.to_dict()

    def test_api_connection(self, model_key: str) -> str:
        """Test API connection for a specific model."""
        config = self._configs.get(model_key)
        if not config:
            return f"✗ Configuration not found for {model_key}"

        try:
            response = requests.post(
                f"{config.api_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.api_key}"
                },
                json={
                    "model": config.model,
                    "messages": [{"role": "user", "content": "test"}],
                    "temperature": config.temperature,
                    "max_tokens": 10
                },
                timeout=10
            )

            if response.status_code == 200:
                return f"✓ {config.name} connection successful"
            return f"✗ {config.name} failed: {response.status_code}"

        except requests.Timeout:
            return f"✗ {config.name} connection timeout"
        except Exception as e:
            return f"✗ {config.name} error: {str(e)}"

    def call_api(self, model_key: str, prompt: str) -> Dict:
        """Call API with retry mechanism."""
        config = self._configs.get(model_key)
        if not config:
            raise ValueError(f"Configuration not found for {model_key}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}"
        }

        data = {
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in analyzing medical literature based on PICOS criteria."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{config.api_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=config.timeout
                )

                if response.status_code == 200:
                    return self._process_response(model_key, response.text)

                last_error = f"API returned {response.status_code}: {response.text[:200]}"
                logging.error(f"{config.name} attempt {attempt + 1}: {last_error}")

            except requests.Timeout:
                last_error = "Request timeout"
                logging.error(f"{config.name} attempt {attempt + 1}: timeout")
            except Exception as e:
                last_error = str(e)
                logging.error(f"{config.name} attempt {attempt + 1}: {e}")

            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_RETRY_DELAY * (attempt + 1))

        raise RuntimeError(f"API call failed after {MAX_RETRIES} attempts: {last_error}")

    def _process_response(self, model_key: str, response: str) -> Dict:
        """Process API response and extract results."""
        try:
            response_obj = json.loads(response) if isinstance(response, str) else response

            config = self._configs.get(model_key)
            if config and config.is_inference:
                return self._process_inference_response(response_obj)

            return self._process_standard_response(response_obj, model_key)

        except Exception as e:
            logging.error(f"Response processing error for {model_key}: {e}")
            return self._get_default_response(model_key)

    def _process_standard_response(self, response_obj: Dict, model_key: str) -> Dict:
        """Process standard (non-inference) API response."""
        if not isinstance(response_obj, dict):
            return self._get_default_response(model_key)

        choices = response_obj.get("choices", [])
        if not choices:
            return self._get_default_response(model_key)

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            return self._get_default_response(model_key)

        # Extract JSON from markdown code blocks if present
        content = self._extract_json_content(content)

        try:
            result = json.loads(content)
            if not self._validate_results(result):
                return self._get_default_response(model_key)
            return result
        except json.JSONDecodeError as e:
            logging.error(f"JSON parse error: {e}")
            return self._get_default_response(model_key)

    def _process_inference_response(self, response_obj: Dict) -> Dict:
        """Process inference model response (with reasoning tags)."""
        if not isinstance(response_obj, dict):
            return {"results": []}

        for choice in response_obj.get("choices", []):
            content = choice.get("message", {}).get("content", "")
            if not content:
                continue

            content = self._extract_json_content(content)

            try:
                result = json.loads(content)
                if self._validate_results(result):
                    return result
            except json.JSONDecodeError:
                continue

        return {"results": []}

    def _extract_json_content(self, content: str) -> str:
        """Extract JSON from markdown code blocks."""
        match = JSON_CODE_BLOCK_PATTERN.search(content)
        return match.group(1).strip() if match else content

    def _validate_results(self, result: Dict) -> bool:
        """Validate that result contains valid results array."""
        if not isinstance(result, dict):
            return False
        results = result.get("results")
        if not isinstance(results, list) or not results:
            return False
        return all(
            isinstance(item, dict) and "Index" in item
            for item in results
        )

    def _get_default_response(self, model_key: str) -> Dict:
        """Get default response structure for failed API calls."""
        defaults = {
            "model_a": {
                "Index": "0",
                "A_P": "not applicable",
                "A_I": "not applicable",
                "A_C": "not applicable",
                "A_O": "not applicable",
                "A_S": "not applicable",
                "A_Decision": False,
                "A_Reason": "API call failed"
            },
            "model_b": {
                "Index": "0",
                "B_P": "not applicable",
                "B_I": "not applicable",
                "B_C": "not applicable",
                "B_O": "not applicable",
                "B_S": "not applicable",
                "B_Decision": False,
                "B_Reason": "API call failed"
            },
            "model_c": {
                "Index": "0",
                "C_Decision": False,
                "C_Reason": "API call failed"
            }
        }
        return {"results": [defaults.get(model_key, defaults["model_c"])]}

    @staticmethod
    def clean_inference_text(text: str) -> str:
        """Remove thinking tags and clean inference response text."""
        if not isinstance(text, str):
            return str(text)

        text = THINK_TAG_PATTERN.sub('', text)
        text = HTML_TAG_PATTERN.sub('', text)
        text = WHITESPACE_PATTERN.sub('\n\n', text.strip())
        return text
