import os
import json
import requests
import logging
import time
import re
from typing import Dict, Any
from dotenv import load_dotenv

# Ensure .env file is loaded (with override enabled to pick up any modifications)
load_dotenv(override=True)

class ModelManager:
    def __init__(self):
        # Load base configuration from environment variables
        self.model_configs = {
            "model_a": {
                "api_key": os.getenv("MODEL_A_API_KEY", ""),
                "api_url": os.getenv("MODEL_A_API_URL", ""),
                "model": os.getenv("MODEL_A_MODEL_NAME", ""),
                "name": "Model A (Primary Analyzer)",
                "temperature": float(os.getenv("MODEL_A_TEMPERATURE", "0.3")),
                "max_tokens": int(os.getenv("MODEL_A_MAX_TOKENS", "4096")),
                "batch_size": int(os.getenv("MODEL_A_BATCH_SIZE", "10")),
                "threads": int(os.getenv("MODEL_A_THREADS", "8")),
                "timeout": int(os.getenv("MODEL_A_TIMEOUT", "180")),
                "is_inference": os.getenv("MODEL_A_IS_INFERENCE", "").lower() == "true",
                "updated": False  # flag to indicate if manually updated
            },
            "model_b": {
                "api_key": os.getenv("MODEL_B_API_KEY", ""),
                "api_url": os.getenv("MODEL_B_API_URL", ""),
                "model": os.getenv("MODEL_B_MODEL_NAME", ""),
                "name": "Model B (Critical Reviewer)",
                "temperature": float(os.getenv("MODEL_B_TEMPERATURE", "0.3")),
                "max_tokens": int(os.getenv("MODEL_B_MAX_TOKENS", "4096")),
                "batch_size": int(os.getenv("MODEL_B_BATCH_SIZE", "10")),
                "threads": int(os.getenv("MODEL_B_THREADS", "8")),
                "timeout": int(os.getenv("MODEL_B_TIMEOUT", "180")),
                "is_inference": os.getenv("MODEL_B_IS_INFERENCE", "").lower() == "true",
                "updated": False
            },
            "model_c": {
                "api_key": os.getenv("MODEL_C_API_KEY", ""),
                "api_url": os.getenv("MODEL_C_API_URL", ""),
                "model": os.getenv("MODEL_C_MODEL_NAME", ""),
                "name": "Model C (Final Arbitrator)",
                "temperature": float(os.getenv("MODEL_C_TEMPERATURE", "0.3")),
                "max_tokens": int(os.getenv("MODEL_C_MAX_TOKENS", "4096")),
                "batch_size": int(os.getenv("MODEL_C_BATCH_SIZE", "10")),
                "threads": int(os.getenv("MODEL_C_THREADS", "8")),
                "timeout": int(os.getenv("MODEL_C_TIMEOUT", "180")),
                "is_inference": os.getenv("MODEL_C_IS_INFERENCE", "").lower() == "true",
                "updated": False
            }
        }
        
        # Validate API keys
        for model_key, config in self.model_configs.items():
            if not config["api_key"]:
                logging.warning(f"API key not found for {config['name']}")
    
    def update_model_config(self, model_key: str, config: Dict[str, Any]) -> None:
        """Update model configuration."""
        if model_key not in self.model_configs:
            raise ValueError(f"Invalid model key: {model_key}")
        self.model_configs[model_key].update(config)
    
    def process_model_response(self, model_key: str, response: str) -> Dict:
        """
        Process response based on model type.
        
        Args:
            model_key: Identifier of the model.
            response: Raw response string from the API.
            
        Returns:
            Dict containing processed response data.
        """
        try:
            # Debug log for raw response
            logging.debug(f"Raw response from {model_key}: {response}")
            
            # Parse outer JSON
            response_obj = json.loads(response) if isinstance(response, str) else response
            
            # Process based on mode
            if self.model_configs[model_key].get("is_inference", False):
                logging.debug(f"Processing {model_key} response in inference mode")
                return self.process_inference_result(response_obj, model_key)
            
            # Get content from response
            if not isinstance(response_obj, dict) or "choices" not in response_obj or not response_obj["choices"]:
                logging.error(f"Invalid response format from {model_key}: {response_obj}")
                return self.get_default_response(model_key)
            
            content = response_obj["choices"][0].get("message", {}).get("content", "")
            if not content:
                logging.error(f"Empty content in {model_key} response")
                return self.get_default_response(model_key)
            
            logging.debug(f"Raw content from {model_key}: {content}")
            
            # Handle markdown code blocks
            if "```json" in content:
                pattern = r"```json\s*(.*?)\s*```"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    logging.debug(f"Extracted JSON from markdown: {content}")
            
            # Parse inner JSON
            try:
                result = json.loads(content)
                logging.debug(f"Parsed JSON result: {json.dumps(result, indent=2, ensure_ascii=False)}")
                
                # Ensure response has results field
                if "results" not in result:
                    logging.error(f"Missing 'results' field in {model_key} response")
                    return self.get_default_response(model_key)
                
                return result
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse {model_key} content JSON: {str(e)}")
                logging.error(f"Content was: {content}")
                return self.get_default_response(model_key)
            
        except Exception as e:
            logging.error(f"Error processing {model_key} response: {str(e)}")
            return self.get_default_response(model_key)

    def get_default_response(self, model_key: str) -> Dict:
        """
        Return default response format for each model type.
        
        Args:
            model_key: Identifier of the model.
            
        Returns:
            Dict containing default response structure.
        """
        if model_key == "model_a":
            return {
                "results": [{
                    "Index": "0",
                    "A_P": "not applicable",
                    "A_I": "not applicable",
                    "A_C": "not applicable",
                    "A_O": "not applicable",
                    "A_S": "not applicable",
                    "A_Decision": False,
                    "A_Reason": "API call failed or returned no results"
                }]
            }
        elif model_key == "model_b":
            return {
                "results": [{
                    "Index": "0",
                    "B_P": "not applicable",
                    "B_I": "not applicable",
                    "B_C": "not applicable",
                    "B_O": "not applicable",
                    "B_S": "not applicable",
                    "B_Decision": False,
                    "B_Reason": "API call failed or returned no results"
                }]
            }
        else:  # model_c
            return {
                "results": [{
                    "Index": "0",
                    "C_Decision": False,
                    "C_Reason": "API call failed or returned no results"
                }]
            }

    def process_inference_result(self, result: Dict, model_key: str) -> Dict:
        """
        Process inference model results.
        
        Args:
            result: Raw inference result.
            model_key: Identifier of the model.
            
        Returns:
            Dict containing processed inference results.
        """
        try:
            if not isinstance(result, dict) or "choices" not in result:
                logging.error(f"Invalid inference result format from {model_key}")
                return self.get_default_response(model_key)
            
            for choice in result["choices"]:
                if "message" not in choice:
                    logging.warning(f"Missing message in choice: {choice}")
                    continue
                
                content = choice["message"].get("content", "")
                if not content:
                    logging.warning(f"Empty content in {model_key} choice")
                    choice["message"]["content"] = json.dumps(self.get_default_response(model_key))
                    continue
                
                # Handle markdown code blocks
                if "```json" in content:
                    pattern = r"```json\s*(.*?)\s*```"
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        content = match.group(1).strip()
                        logging.debug(f"Extracted JSON from markdown in inference result: {content}")
                
                try:
                    content_data = json.loads(content)
                    logging.debug(f"Parsed inference content: {json.dumps(content_data, indent=2, ensure_ascii=False)}")
                    
                    # Validate and standardize content format
                    if "results" not in content_data:
                        logging.error(f"Missing 'results' field in {model_key} inference result")
                        content_data = self.get_default_response(model_key)
                    else:
                        # Ensure each result has the required fields
                        for result_item in content_data["results"]:
                            if model_key == "model_c":
                                if "Index" not in result_item or "C_Decision" not in result_item or "C_Reason" not in result_item:
                                    logging.error(f"Missing required fields in Model C result: {result_item}")
                                    continue
                                # Convert decision to boolean if it's a string
                                if isinstance(result_item["C_Decision"], str):
                                    result_item["C_Decision"] = result_item["C_Decision"].lower() == "true"
                    
                    choice["message"]["content"] = json.dumps(content_data)
                    
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse {model_key} inference content: {str(e)}")
                    logging.error(f"Content was: {content}")
                    choice["message"]["content"] = json.dumps(self.get_default_response(model_key))
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing {model_key} inference result: {str(e)}")
            return self.get_default_response(model_key)

    def process_reviews(self, result: Dict, model_key: str) -> Dict:
        """
        Process reviews format response.
        
        Args:
            result: Raw review data.
            model_key: Identifier of the model.
            
        Returns:
            Dict containing processed reviews.
        """
        try:
            if not isinstance(result.get("reviews", []), list):
                logging.error("Invalid reviews format")
                return {"reviews": []}
            
            field_name = "B_Reason" if model_key == "model_b" else "C_Reason"
            for review in result["reviews"]:
                if field_name in review:
                    # Remove duplicate Reason fields
                    if isinstance(review[field_name], list):
                        review[field_name] = review[field_name][-1]
                    
                    # Process inference content (remove think tags etc.)
                    review[field_name] = self.process_inference_response(review[field_name])
            
            return result
        except Exception as e:
            logging.error(f"Error processing reviews: {str(e)}")
            return {"reviews": []}
    
    def process_inference_response(self, response: str) -> str:
        """
        Process special markers in inference response.
        
        Args:
            response: Raw inference response string.
            
        Returns:
            Processed response string with special markers removed.
        """
        try:
            if not isinstance(response, str):
                return response
                
            # Remove thinking process
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            
            # Remove HTML tags
            response = re.sub(r'<[^>]+>', '', response)
            
            # Clean extra whitespace
            response = re.sub(r'\n\s*\n', '\n\n', response.strip())
            
            return response
            
        except Exception as e:
            logging.error(f"Error processing inference response: {str(e)}")
            return response
    
    def test_api_connection(self, model_key: str) -> str:
        """
        Test API connection for a specific model.
        
        Args:
            model_key: Identifier of the model to test.
            
        Returns:
            String indicating connection status.
        """
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
        """
        Call API with retry mechanism and improved error handling.
        
        Args:
            model_key: Identifier of the model to call.
            prompt: Prompt text to send to the API.
            
        Returns:
            Dict containing API response.
            
        Raises:
            Exception: If API call fails after retries.
        """
        config = self.model_configs.get(model_key)
        if not config:
            raise Exception(f"Configuration not found for {model_key}")
        
        max_retries = 3
        retry_delay = 1  # Initial retry delay in seconds
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key']}"
        }
        
        # Debug log for API call
        logging.debug(f"Calling {model_key} API with config: {json.dumps({k:v for k,v in config.items() if k != 'api_key'}, indent=2)}")
        
        # Construct standard request format
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in analyzing medical literature based on PICOS criteria."},
            {"role": "user", "content": prompt}
        ]
        
        data = {
            "model": config["model"],
            "messages": messages,
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"]
        }
        
        logging.debug(f"Request data for {model_key}: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    config["api_url"],
                    headers=headers,
                    json=data,
                    timeout=config["timeout"]
                )
                
                # Debug logs for API response
                logging.debug(f"API Response status: {response.status_code}")
                logging.debug(f"API Response headers: {dict(response.headers)}")
                if response.status_code != 200:
                    error_msg = f"API call failed for {config['name']}: {response.status_code} {response.reason}"
                    if response.text:
                        error_msg += f"\nResponse: {response.text}"
                    logging.error(error_msg)
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise Exception(error_msg)
                
                # Process response based on model type
                return self.process_model_response(model_key, response.text)
                
            except requests.Timeout:
                logging.error(f"Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                raise Exception(f"API call timed out after {max_retries} attempts")
                
            except Exception as e:
                error_msg = f"API call error for {config['name']}: {str(e)}"
                logging.error(error_msg)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception(error_msg)
                
        raise Exception(f"API call failed after {max_retries} attempts")
    
    def get_config(self, model_key: str) -> Dict[str, Any]:
        """
        Get model configuration.
        This method re-reads environment variables for models that haven't been manually updated.
        """
        # Reload environment variables from .env file to capture any modifications
        load_dotenv(override=True)
        if model_key not in self.model_configs:
            return {}
        config = self.model_configs[model_key]
        if not config.get("updated", False):
            # For models not manually updated, refresh config from environment variables
            if model_key == "model_a":
                refreshed_config = {
                    "api_key": os.getenv("MODEL_A_API_KEY", ""),
                    "api_url": os.getenv("MODEL_A_API_URL", ""),
                    "model": os.getenv("MODEL_A_MODEL_NAME", ""),
                    "temperature": float(os.getenv("MODEL_A_TEMPERATURE", "0.3")),
                    "max_tokens": int(os.getenv("MODEL_A_MAX_TOKENS", "4096")),
                    "batch_size": int(os.getenv("MODEL_A_BATCH_SIZE", "10")),
                    "threads": int(os.getenv("MODEL_A_THREADS", "8")),
                    "timeout": int(os.getenv("MODEL_A_TIMEOUT", "180")),
                    "is_inference": os.getenv("MODEL_A_IS_INFERENCE", "").lower() == "true",
                    "updated": False
                }
            elif model_key == "model_b":
                refreshed_config = {
                    "api_key": os.getenv("MODEL_B_API_KEY", ""),
                    "api_url": os.getenv("MODEL_B_API_URL", ""),
                    "model": os.getenv("MODEL_B_MODEL_NAME", ""),
                    "temperature": float(os.getenv("MODEL_B_TEMPERATURE", "0.3")),
                    "max_tokens": int(os.getenv("MODEL_B_MAX_TOKENS", "4096")),
                    "batch_size": int(os.getenv("MODEL_B_BATCH_SIZE", "10")),
                    "threads": int(os.getenv("MODEL_B_THREADS", "8")),
                    "timeout": int(os.getenv("MODEL_B_TIMEOUT", "180")),
                    "is_inference": os.getenv("MODEL_B_IS_INFERENCE", "").lower() == "true",
                    "updated": False
                }
            elif model_key == "model_c":
                refreshed_config = {
                    "api_key": os.getenv("MODEL_C_API_KEY", ""),
                    "api_url": os.getenv("MODEL_C_API_URL", ""),
                    "model": os.getenv("MODEL_C_MODEL_NAME", ""),
                    "temperature": float(os.getenv("MODEL_C_TEMPERATURE", "0.3")),
                    "max_tokens": int(os.getenv("MODEL_C_MAX_TOKENS", "4096")),
                    "batch_size": int(os.getenv("MODEL_C_BATCH_SIZE", "10")),
                    "threads": int(os.getenv("MODEL_C_THREADS", "8")),
                    "timeout": int(os.getenv("MODEL_C_TIMEOUT", "180")),
                    "is_inference": os.getenv("MODEL_C_IS_INFERENCE", "").lower() == "true",
                    "updated": False
                }
            else:
                refreshed_config = {}
            self.model_configs[model_key] = refreshed_config
            config = refreshed_config
        return config

    def process_analysis(self, result: Dict, model_key: str) -> Dict:
        """
        Process analysis format response.
        
        Args:
            result: Raw analysis data.
            model_key: Identifier of the model.
            
        Returns:
            Dict containing processed analysis.
        """
        try:
            if not isinstance(result.get("analysis", []), list):
                logging.error("Invalid analysis format")
                return {"analysis": []}
            
            # Process each analysis item
            for analysis in result["analysis"]:
                if "A_Reason" in analysis:
                    # Remove duplicate Reason fields
                    if isinstance(analysis["A_Reason"], list):
                        analysis["A_Reason"] = analysis["A_Reason"][-1]
                    
                    # Process inference content (remove think tags etc.)
                    analysis["A_Reason"] = self.process_inference_response(analysis["A_Reason"])
                
                # Ensure boolean fields are proper booleans
                if "A_Decision" in analysis:
                    analysis["A_Decision"] = bool(analysis["A_Decision"])
                
                # Ensure all PICOS fields are strings
                for field in ["A_P", "A_I", "A_C", "A_O", "A_S"]:
                    if field in analysis:
                        analysis[field] = str(analysis[field])
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing analysis: {str(e)}")
            return {"analysis": []}
