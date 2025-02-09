import os
import json
import requests
import logging
import time
import re
from typing import Dict, Any

class ModelManager:
    def __init__(self):
        # 从环境变量加载基础配置
        self.model_configs = {
            "model_a": {
                "api_key": os.getenv("MODEL_A_API_KEY", ""),
                "api_url": os.getenv("MODEL_A_API_URL", ""),
                "model": os.getenv("MODEL_A_MODEL_NAME", ""),
                "name": "Model A (Primary Analyzer)",
                "temperature": float(os.getenv("MODEL_A_TEMPERATURE", "0.3")),
                "max_tokens": int(os.getenv("MODEL_A_MAX_TOKENS", "16384")),
                "batch_size": int(os.getenv("MODEL_A_BATCH_SIZE", "10")),
                "threads": int(os.getenv("MODEL_A_THREADS", "8")),
                "timeout": int(os.getenv("MODEL_A_TIMEOUT", "180")),
                "is_inference": os.getenv("MODEL_A_IS_INFERENCE", "").lower() == "true"
            },
            "model_b": {
                "api_key": os.getenv("MODEL_B_API_KEY", ""),
                "api_url": os.getenv("MODEL_B_API_URL", ""),
                "model": os.getenv("MODEL_B_MODEL_NAME", ""),
                "name": "Model B (Critical Reviewer)",
                "temperature": float(os.getenv("MODEL_B_TEMPERATURE", "0.3")),
                "max_tokens": int(os.getenv("MODEL_B_MAX_TOKENS", "16384")),
                "batch_size": int(os.getenv("MODEL_B_BATCH_SIZE", "10")),
                "threads": int(os.getenv("MODEL_B_THREADS", "8")),
                "timeout": int(os.getenv("MODEL_B_TIMEOUT", "180")),
                "is_inference": os.getenv("MODEL_B_IS_INFERENCE", "").lower() == "true"
            },
            "model_c": {
                "api_key": os.getenv("MODEL_C_API_KEY", ""),
                "api_url": os.getenv("MODEL_C_API_URL", ""),
                "model": os.getenv("MODEL_C_MODEL_NAME", ""),
                "name": "Model C (Final Arbitrator)",
                "temperature": float(os.getenv("MODEL_C_TEMPERATURE", "0.3")),
                "max_tokens": int(os.getenv("MODEL_C_MAX_TOKENS", "16384")),
                "batch_size": int(os.getenv("MODEL_C_BATCH_SIZE", "10")),
                "threads": int(os.getenv("MODEL_C_THREADS", "8")),
                "timeout": int(os.getenv("MODEL_C_TIMEOUT", "180")),
                "is_inference": os.getenv("MODEL_C_IS_INFERENCE", "").lower() == "true"
            }
        }
        
        # 验证 API keys
        for model_key, config in self.model_configs.items():
            if not config["api_key"]:
                logging.warning(f"API key not found for {config['name']}")
    
    def update_model_config(self, model_key: str, config: Dict[str, Any]) -> None:
        """Update model configuration"""
        if model_key not in self.model_configs:
            raise ValueError(f"Invalid model key: {model_key}")
        self.model_configs[model_key].update(config)
    
    def process_model_response(self, model_key: str, response: str) -> Dict:
        """根据模型类型处理响应"""
        config = self.model_configs[model_key]
        
        try:
            # 解析 JSON
            result = json.loads(response)
            
            # 检查是否为推理模型
            if config.get("is_inference", False):
                # 处理 OpenAI 格式的响应
                if "choices" in result:
                    for choice in result["choices"]:
                        if "message" in choice:
                            content = choice["message"].get("content", "")
                            if not content:
                                choice["message"]["content"] = json.dumps({"reviews": []})
                                continue
                            
                            try:
                                content_data = json.loads(content)
                                if "reviews" in content_data:
                                    processed_reviews = self.process_reviews(content_data, model_key)
                                    choice["message"]["content"] = json.dumps(processed_reviews)
                                else:
                                    choice["message"]["content"] = json.dumps({"reviews": []})
                            except json.JSONDecodeError:
                                raise
                    
                    return result
                
                # 直接的 reviews 格式响应
                elif "reviews" in result:
                    processed_reviews = self.process_reviews(result, model_key)
                    return {
                        "choices": [{
                            "message": {
                                "content": json.dumps(processed_reviews)
                            }
                        }]
                    }
                
                # 其他格式，返回空结果
                return {
                    "choices": [{
                        "message": {
                            "content": json.dumps({"reviews": []})
                        }
                    }]
                }
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing {model_key} response: {str(e)}")
            raise

    def process_reviews(self, result: Dict, model_key: str) -> Dict:
        """处理 reviews 格式的响应"""
        try:
            if not isinstance(result.get("reviews", []), list):
                logging.error("Invalid reviews format")
                return {"reviews": []}
            
            field_name = "B_Reason" if model_key == "model_b" else "C_Reason"
            for review in result["reviews"]:
                if field_name in review:
                    # 移除重复的 Reason 字段
                    if isinstance(review[field_name], list):
                        review[field_name] = review[field_name][-1]
                    
                    # 处理推理内容（移除 think 标签等）
                    review[field_name] = self.process_inference_response(review[field_name])
            
            return result
        except Exception as e:
            logging.error(f"Error processing reviews: {str(e)}")
            return {"reviews": []}
    
    def process_inference_response(self, response: str) -> str:
        """处理推理响应中的特殊标记"""
        try:
            if not isinstance(response, str):
                return response
                
            # 移除思考过程
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            
            # 移除 HTML 标签
            response = re.sub(r'<[^>]+>', '', response)
            
            # 清理多余空白
            response = re.sub(r'\n\s*\n', '\n\n', response.strip())
            
            return response
            
        except Exception as e:
            logging.error(f"Error processing inference response: {str(e)}")
            return response
    
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
                
                # 如果是推理模型，使用百炼的格式
                if config.get("is_inference", False):
                    data = {
                        "model": config["model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": config["temperature"],
                        "max_tokens": config["max_tokens"]
                    }
                else:
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
                
                # 检查响应内容是否为空
                if not response.text.strip():
                    raise Exception("Empty response content")
                
                # 根据模型类型处理响应
                return self.process_model_response(model_key, response.text)
                
            except requests.Timeout:
                logging.error(f"Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # 指数退避
                    continue
                raise Exception(f"API call timed out after {max_retries} attempts")
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception(f"API call failed for {config['name']}: {str(e)}")
                
        raise Exception(f"API call failed after {max_retries} attempts")
    
    def get_config(self, model_key: str) -> Dict[str, Any]:
        """Get model configuration"""
        return self.model_configs.get(model_key, {}) 