import os
import json
import requests
import logging
import time
from typing import Dict, Any

class ModelManager:
    def __init__(self):
        self.model_configs = {
            "model_a": {
                "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
                "api_url": "https://api.deepseek.com/v1/chat/completions",
                "model": "deepseek-chat",
                "name": "Model A (Primary Analyzer)",
                "temperature": 0.3,
                "max_tokens": 2000,
                "batch_size": 10,
                "threads": 8,
                "timeout": 60
            },
            "model_b": {
                "api_key": os.getenv("QWEN_API_KEY", ""),
                "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                "model": "qwen-turbo",
                "name": "Model B (Critical Reviewer)",
                "temperature": 0.3,
                "max_tokens": 2000,
                "batch_size": 10,
                "threads": 8,
                "timeout": 60
            },
            "model_c": {
                "api_key": os.getenv("GPTGE_API_KEY", ""),
                "api_url": "https://api.gpt.ge/v1/chat/completions",
                "model": "gpt-4o",
                "name": "Model C (Final Arbitrator)",
                "temperature": 0.3,
                "max_tokens": 2000,
                "batch_size": 10,
                "threads": 8,
                "timeout": 60
            }
        }
        
        # Validate API keys
        for model_key, config in self.model_configs.items():
            if not config["api_key"]:
                logging.warning(f"API key not found for {config['name']}")
    
    def update_model_config(self, model_key: str, config: Dict[str, Any]) -> None:
        """Update model configuration"""
        if model_key not in self.model_configs:
            raise ValueError(f"Invalid model key: {model_key}")
        self.model_configs[model_key].update(config)
    
    def test_api_connection(self, model_key: str) -> str:
        """Test API connection"""
        config = self.model_configs.get(model_key)
        if not config:
            return f"❌ Configuration not found for {model_key}"
            
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config['api_key']}"
            }
            
            data = {
                "model": config["model"],
                "messages": [{"role": "user", "content": "test"}],
                "temperature": config["temperature"],
                "max_tokens": 10
            }
            
            response = requests.post(
                config["api_url"],
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                return f"✓ {config['name']} connection successful"
            else:
                return f"❌ {config['name']} connection failed: {response.status_code}"
                
        except Exception as e:
            return f"❌ {config['name']} connection error: {str(e)}"
    
    def call_api(self, model_key: str, prompt: str) -> Dict:
        """Call API with retry mechanism and improved error handling"""
        max_retries = 3
        retry_delay = 2
        config = self.model_configs[model_key]
        
        for attempt in range(max_retries):
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config['api_key']}"
                }
                
                data = {
                    "model": config["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": config["temperature"],
                    "max_tokens": config["max_tokens"]
                }
                
                # Log request data
                logging.info(f"Sending request to {config['name']}:")
                logging.info(f"URL: {config['api_url']}")
                logging.info(f"Headers: {headers}")
                logging.info(f"Request Data: {json.dumps(data, indent=2)}")
                
                response = requests.post(
                    config["api_url"],
                    headers=headers,
                    json=data,
                    timeout=config.get("timeout", 60)
                )
                
                # Log response details
                logging.info(f"Response Status: {response.status_code}")
                logging.info(f"Response Headers: {dict(response.headers)}")
                logging.info(f"Response Content: {response.text}")
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', retry_delay))
                    logging.warning(f"Rate limited by {config['name']}, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                
                # Try to parse JSON response
                try:
                    json_response = response.json()
                    logging.info(f"Parsed JSON Response: {json.dumps(json_response, indent=2)}")
                    return json_response
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON response: {str(e)}")
                    logging.error(f"Raw response content: {response.text}")
                    raise
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"API call failed for {config['name']}: {str(e)}, retrying...")
                    time.sleep(retry_delay)
                    continue
                raise Exception(f"API call failed for {config['name']}: {str(e)}")
    
    def get_config(self, model_key: str) -> Dict[str, Any]:
        """Get model configuration"""
        return self.model_configs.get(model_key, {}) 