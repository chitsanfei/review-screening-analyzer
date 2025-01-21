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
                "model": "qwen-plus-1127",
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
                
                response = requests.post(
                    config["api_url"],
                    headers=headers,
                    json=data,
                    timeout=config.get("timeout", 60)
                )
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', retry_delay))
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    raise Exception(f"Failed to parse JSON response: {str(e)}")
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception(f"API call failed for {config['name']}: {str(e)}")
    
    def get_config(self, model_key: str) -> Dict[str, Any]:
        """Get model configuration"""
        return self.model_configs.get(model_key, {}) 