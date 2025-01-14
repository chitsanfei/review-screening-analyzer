import os
import json
import pandas as pd
import requests
import logging
import csv
from datetime import datetime
from typing import Dict, List, Any
import traceback
import gradio as gr
from dotenv import load_dotenv
import concurrent.futures
import time
import re

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
for directory in [DATA_DIR, LOG_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {directory}: {str(e)}")

# Logging configuration
try:
    log_file = os.path.join(LOG_DIR, f"picos_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
except Exception as e:
    print(f"Failed to initialize logging: {str(e)}")
    raise

# Global variables
analyzer = None
model_results = {}

class PICOSAnalyzer:
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
                "timeout": 60  # Add timeout setting
            },
            "model_b": {
                "api_key": os.getenv("QWEN_API_KEY", ""),
                "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                "model": "qwen-max",
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

        self.picos_criteria = {
            "population": "patients with hepatocellular carcinoma",
            "intervention": "immune checkpoint inhibitors (ICIs)",
            "comparison": "treatment without the studied ICIs or placebo",
            "outcome": "survival rate or response rate",
            "study_design": "randomized controlled trial"
        }
        
        self.prompts = {
            "model_a": """You are a medical research expert analyzing clinical trial abstracts.
Your task is to analyze each abstract and determine if it matches the PICOS criteria.

Target PICOS criteria:
- Population: {population}
- Intervention: {intervention}
- Comparison: {comparison}
- Outcome: {outcome}
- Study Design: {study_design}

Input abstracts:
{abstracts_json}

Each article in the input contains:
- index: article identifier
- abstract: the text to analyze

IMPORTANT: You must follow these strict JSON formatting rules:
1. Use double quotes for all strings
2. Ensure all strings are properly terminated
3. Use commas between array items and object properties
4. Do not use trailing commas
5. Keep the response concise and avoid unnecessary whitespace
6. Escape any special characters in strings
7. Use true/false (not True/False) for boolean values

Provide your analysis in this exact JSON format:
{{
  "analysis": [
    {{
      "index": "ARTICLE_INDEX",
      "P": "brief population description",
      "I": "brief intervention description",
      "C": "brief comparison description",
      "O": "brief outcome description",
      "S": "brief study design description",
      "matches_criteria": true/false,
      "reasoning": "brief reasoning for match/mismatch"
    }},
    ...
  ]
}}

Keep all descriptions brief and focused. Do not include line breaks or special characters in the text fields.
If any field is not found in the abstract, use "not specified" as the value.
Be strict in your evaluation and ensure the output is valid JSON format.""",
            "model_b": """You are a critical reviewer in a systematic review team.
Your task is to critically review the initial PICOS analysis and provide your own assessment.

Target PICOS criteria:
- Population: {population}
- Intervention: {intervention}
- Comparison: {comparison}
- Outcome: {outcome}
- Study Design: {study_design}

Articles for review:
{abstracts_json}

Each article in the input contains:
- index: article identifier
- abstract: original article abstract
- model_a_analysis:
  - P, I, C, O, S: extracted PICOS elements
  - matches_criteria: initial inclusion decision
  - reasoning: explanation for the decision

Your task is to:
1. Review the original abstract
2. Critically assess Model A's PICOS extraction and decision
3. Provide corrected PICOS elements if you disagree
4. Make your own inclusion decision with reasoning

IMPORTANT: You must follow these strict JSON formatting rules:
1. Use double quotes for all strings
2. Ensure all strings are properly terminated
3. Use commas between array items and object properties
4. Do not use trailing commas
5. Keep the response concise and avoid unnecessary whitespace
6. Escape any special characters in strings
7. Use true/false for inclusion_decision
8. Use "-" for unchanged PICOS elements

Return your analysis in this exact JSON format:
{{
  "reviews": [
    {{
      "index": "ARTICLE_INDEX",
      "inclusion_decision": true/false,
      "review_comments": "brief critical analysis of Model A's decision",
      "corrected_P": "-" or "brief corrected population",
      "corrected_I": "-" or "brief corrected intervention",
      "corrected_C": "-" or "brief corrected comparison",
      "corrected_O": "-" or "brief corrected outcome",
      "corrected_S": "-" or "brief corrected study design"
    }},
    ...
  ]
}}

Keep all descriptions brief and focused. Do not include line breaks or special characters in the text fields.
If you agree with Model A's PICOS extraction, use exactly "-" as the value.""",
            "model_c": """You are the final arbitrator in a systematic review process.
Your task is to resolve disagreements between Model A and Model B's analyses.

Target PICOS criteria:
- Population: {population}
- Intervention: {intervention}
- Comparison: {comparison}
- Outcome: {outcome}
- Study Design: {study_design}

Articles with disagreements:
{disagreements_json}

Each article in the input contains:
- index: article identifier
- abstract: original article abstract
- model_a_analysis:
  - P, I, C, O, S: extracted PICOS elements
  - matches_criteria: inclusion decision
  - reasoning: explanation for the decision
- model_b_analysis:
  - corrected_P, corrected_I, corrected_C, corrected_O, corrected_S: reviewed PICOS elements
  - inclusion_decision: reviewed decision
  - review_comments: critical analysis

Your task is to:
1. Review the original abstract
2. Consider both Model A and B's PICOS extractions
3. Consider both models' inclusion decisions and reasoning
4. Make a final decision on inclusion
5. Provide clear reasoning for your decision

IMPORTANT: You must follow these strict JSON formatting rules:
1. Use double quotes for all strings
2. Ensure all strings are properly terminated
3. Use commas between array items and object properties
4. Do not use trailing commas
5. Keep the response concise and avoid unnecessary whitespace
6. Escape any special characters in strings
7. Use true/false for final_decision

Return your decisions in this EXACT JSON format (no other text allowed):
{{
  "decisions": [
    {{
      "index": "ARTICLE_INDEX",
      "final_decision": true/false,
      "reasoning": "brief explanation considering both models' analyses"
    }},
    ...
  ]
}}

Keep all descriptions brief and focused. Do not include line breaks or special characters in the text fields.
Be thorough and objective in your final judgment."""
        }
    
    def update_model_config(self, model_key: str, config: Dict):
        """Update model configuration"""
        if model_key not in self.model_configs:
            raise ValueError(f"Invalid model key: {model_key}")
        self.model_configs[model_key].update(config)
    
    def update_prompt(self, model_key: str, prompt: str):
        """Update model prompt"""
        if model_key not in self.prompts:
            raise ValueError(f"Invalid model key: {model_key}")
        self.prompts[model_key] = prompt
    
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
    
    def process_batch(self, df: pd.DataFrame, model_key: str, previous_results: Dict = None) -> pd.DataFrame:
        """Process a batch of data"""
        config = self.model_configs[model_key]
        batch_size = config["batch_size"]
        threads = config["threads"]
        results = []
        total_rows = len(df)
        completed_rows = 0
        error_count = 0
        failed_indices = set()  # Track failed indices
        
        def update_progress():
            progress = (completed_rows / total_rows) * 100
            status_text = (f"Processing {model_key.upper()}: {completed_rows}/{total_rows} rows ({progress:.1f}%) "
                         f"- Errors: {error_count}")
            if hasattr(update_progress, 'last_status') and update_progress.last_status == status_text:
                return
            update_progress.last_status = status_text
            logging.info(status_text)
        
        def process_batch_data(batch_df: pd.DataFrame) -> List[Dict]:
            nonlocal error_count, failed_indices
            batch_results = []
            empty_results = []  # Store results for empty/invalid abstracts
            
            for idx, row in batch_df.iterrows():
                try:
                    is_empty = pd.isna(row["Abstract"]) or len(str(row["Abstract"]).strip()) < 50
                    if is_empty:
                        logging.warning(f"Row {idx}: Abstract too short or empty")
                        # Create placeholder result for empty abstract
                        empty_result = {
                            "index": str(idx)
                        }
                        if model_key == "model_a":
                            empty_result.update({
                                "P": "not applicable",
                                "I": "not applicable",
                                "C": "not applicable",
                                "O": "not applicable",
                                "S": "not applicable",
                                "matches_criteria": False,
                                "reasoning": "Abstract too short or empty"
                            })
                        elif model_key == "model_b":
                            empty_result.update({
                                "inclusion_decision": False,
                                "review_comments": "Abstract too short or empty",
                                "corrected_P": "-",
                                "corrected_I": "-",
                                "corrected_C": "-",
                                "corrected_O": "-",
                                "corrected_S": "-"
                            })
                        else:  # model_c
                            empty_result.update({
                                "final_decision": False,
                                "reasoning": "Abstract too short or empty"
                            })
                        empty_results.append(empty_result)
                        continue
                        
                    abstract = {
                        "index": str(idx),
                        "abstract": str(row["Abstract"]).strip()
                    }
                    
                    # For Model B and C, check if we have all required previous results
                    if model_key in ["model_b", "model_c"]:
                        if not previous_results or "model_a" not in previous_results:
                            raise Exception("Model A results required")
                        if idx not in previous_results["model_a"].index:
                            logging.warning(f"Missing Model A result for index {idx}")
                            failed_indices.add(idx)
                            continue
                            
                    if model_key == "model_c" and "model_b" not in previous_results:
                        raise Exception("Model B results required")
                        
                    batch_results.append(abstract)
                    
                except Exception as e:
                    logging.error(f"Error processing row {idx}: {str(e)}")
                    error_count += 1
                    failed_indices.add(idx)
            
            try:
                api_results = []
                if batch_results:  # Only call API if we have valid abstracts
                    if model_key == "model_a":
                        api_results = self._call_model_a(batch_results)
                    elif model_key == "model_b":
                        model_a_results = []
                        for abstract in batch_results:
                            idx = int(abstract["index"])
                            a_result = previous_results["model_a"].loc[idx].to_dict()
                            model_a_results.append(a_result)
                        api_results = self._call_model_b(batch_results, model_a_results)
                    else:  # model_c
                        model_a_results = []
                        model_b_results = []
                        for abstract in batch_results:
                            idx = int(abstract["index"])
                            # 确保获取正确的决策字段
                            a_result = previous_results["model_a"].loc[idx]
                            b_result = previous_results["model_b"].loc[idx]
                            
                            # 检查是否有分歧
                            if a_result["matches_criteria"] != b_result["inclusion_decision"]:
                                model_a_results.append(a_result.to_dict())
                                model_b_results.append(b_result.to_dict())
                                logging.debug(f"Found disagreement at index {idx}: A={a_result['matches_criteria']}, B={b_result['inclusion_decision']}")
                        
                        if model_a_results and model_b_results:  # 只有在有分歧时才调用 Model C
                            api_results = self._call_model_c(batch_results, model_a_results, model_b_results)
                        else:
                            # 如果没有分歧，创建空结果
                            api_results = []
                            for abstract in batch_results:
                                idx = abstract["index"]
                                a_result = previous_results["model_a"].loc[int(idx)]
                                api_results.append({
                                    "index": idx,
                                    "final_decision": a_result["matches_criteria"],
                                    "reasoning": "No disagreement between Model A and B"
                                })
                
                # Combine API results with empty results
                return api_results + empty_results
                
            except Exception as e:
                logging.error(f"Error in model call: {str(e)}")
                error_count += len(batch_results)
                for abstract in batch_results:
                    failed_indices.add(int(abstract["index"]))
                return empty_results  # Return empty results even if API call fails
        
        # Create batches
        batches = []
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size].copy()
            batches.append(batch_df)
        
        # Use thread pool to process batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for batch_df in batches:
                future = executor.submit(process_batch_data, batch_df)
                futures.append((batch_df.index, future))
            
            # Process results
            for batch_indices, future in futures:
                try:
                    batch_results = future.result()
                    if batch_results:
                        results.extend(batch_results)
                except Exception as e:
                    error_count += len(batch_indices)
                    for idx in batch_indices:
                        failed_indices.add(idx)
                    logging.error(f"Error processing batch: {str(e)}")
                finally:
                    completed_rows += len(batch_indices)
                    update_progress()
        
        if not results:
            raise Exception("No results were successfully processed")
        
        # Convert to DataFrame and validate results
        results_df = pd.DataFrame(results)
        
        # Ensure we have the index column and it's properly set
        if 'index' in results_df.columns:
            results_df['Index'] = results_df['index']
            results_df = results_df.drop('index', axis=1)
        results_df.set_index('Index', inplace=True)
        
        # Validate required columns
        required_columns = {
            "model_a": ["P", "I", "C", "O", "S", "matches_criteria", "reasoning"],
            "model_b": ["inclusion_decision", "review_comments", 
                       "corrected_P", "corrected_I", "corrected_C", "corrected_O", "corrected_S"],
            "model_c": ["final_decision", "reasoning"]
        }
        
        missing_columns = set(required_columns[model_key]) - set(results_df.columns)
        if missing_columns:
            raise Exception(f"Missing required columns in results: {missing_columns}")
        
        # Log failed indices
        if failed_indices:
            failed_list = sorted(list(failed_indices))
            logging.warning(f"Failed to process {len(failed_indices)} indices in {model_key}: {failed_list}")
            
        return results_df
    
    def merge_results(self, df: pd.DataFrame, model_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all model results"""
        # Create a copy of input DataFrame
        merged = df.copy()
        
        # Ensure index is properly set
        if 'Index' in merged.columns:
            merged.set_index('Index', inplace=True)
        
        # Add Model A results
        if "model_a" in model_results:
            model_a_df = model_results["model_a"].copy()
            if 'index' in model_a_df.columns:
                model_a_df = model_a_df.drop('index', axis=1)
            for col in model_a_df.columns:
                merged[f"A_{col}"] = model_a_df[col]
        
        # Add Model B results
        if "model_b" in model_results:
            model_b_df = model_results["model_b"].copy()
            if 'index' in model_b_df.columns:
                model_b_df = model_b_df.drop('index', axis=1)
            for col in model_b_df.columns:
                merged[f"B_{col}"] = model_b_df[col]
        
        # Add Model C results
        if "model_c" in model_results:
            model_c_df = model_results["model_c"].copy()
            if 'index' in model_c_df.columns:
                model_c_df = model_c_df.drop('index', axis=1)
            for col in model_c_df.columns:
                merged[f"C_{col}"] = model_c_df[col]
        
        return merged
    
    def _call_api(self, config: Dict, prompt: str, model_key: str) -> Dict:
        """Call API with retry mechanism and improved error handling"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Validate API configuration
                if not config.get('api_key') or not config.get('api_url') or not config.get('model'):
                    raise Exception(f"Invalid API configuration for {config.get('name', 'Unknown Model')}")

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config['api_key']}"
                }
                
                data = {
                    "model": config["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": config["temperature"],
                    "max_tokens": config["max_tokens"],
                    "response_format": {"type": "json_object"}
                }
                
                try:
                    logging.debug(f"Sending request to {config['name']} (Attempt {attempt + 1}/{max_retries})")
                    response = requests.post(
                        config["api_url"],
                        headers=headers,
                        json=data,
                        timeout=config.get("timeout", 60)
                    )
                    
                    # Check for rate limiting
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', retry_delay))
                        logging.warning(f"Rate limited by {config['name']}, waiting {retry_after} seconds")
                        time.sleep(retry_after)
                        continue
                        
                    response.raise_for_status()  # Raise exception for bad status codes
                    
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        logging.warning(f"API call timed out for {config['name']}, retrying...")
                        time.sleep(retry_delay)
                        continue
                    raise Exception(f"API call timed out for {config['name']} after {max_retries} attempts")
                    
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"API call failed for {config['name']}: {str(e)}, retrying...")
                        time.sleep(retry_delay)
                        continue
                    raise Exception(f"API call failed for {config['name']}: {str(e)}")

                try:
                    response_json = response.json()
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse response as JSON: {response.text}")
                    if attempt < max_retries - 1:
                        logging.warning("Retrying due to JSON parse error...")
                        time.sleep(retry_delay)
                        continue
                    raise Exception(f"Invalid JSON response from API: {str(e)}")
                
                # 记录原始响应
                logging.debug(f"Raw API response: {json.dumps(response_json, indent=2)}")
                
                # 处理不同的响应格式
                try:
                    if model_key == "model_c":
                        # 首先尝试解析content字段
                        if "choices" in response_json and response_json["choices"]:
                            content = response_json["choices"][0]["message"]["content"]
                            # 清理content中的非JSON内容
                            content = content.strip()
                            # 记录清理前的content
                            logging.debug(f"Original content: {content}")
                            
                            try:
                                # 直接尝试解析整个content
                                result = json.loads(content)
                                if isinstance(result, dict) and "decisions" in result:
                                    return result
                            except json.JSONDecodeError:
                                # 如果直接解析失败，尝试提取JSON部分
                                json_start = content.find('{')
                                json_end = content.rfind('}') + 1
                                if json_start != -1 and json_end > json_start:
                                    try:
                                        content = content[json_start:json_end]
                                        # 记录提取的JSON部分
                                        logging.debug(f"Extracted JSON content: {content}")
                                        result = json.loads(content)
                                        if isinstance(result, dict) and "decisions" in result:
                                            return result
                                    except json.JSONDecodeError as e:
                                        logging.error(f"Failed to parse extracted JSON: {str(e)}")
                                        raise ValueError(f"Invalid JSON in content: {content}")
                        
                        # 如果上述方法都失败，尝试从原始响应中获取decisions
                        if isinstance(response_json, dict) and "decisions" in response_json:
                            return response_json
                        
                        # 如果所有尝试都失败，记录详细信息并抛出异常
                        logging.error(f"Failed to extract decisions from response: {json.dumps(response_json, indent=2)}")
                        raise ValueError(f"Unable to find valid decisions in response")
                    
                    else:  # 处理其他模型的响应
                        if 'choices' not in response_json or not response_json['choices']:
                            raise ValueError(f"Invalid API response format: missing choices - {response_json}")
                        
                        content = response_json['choices'][0]['message']['content']
                        result = json.loads(content)
                        return result
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Error processing response: {str(e)}, retrying...")
                        time.sleep(retry_delay)
                        continue
                    raise ValueError(f"Failed to process API response: {str(e)}")
                
                if 'choices' not in response_json or not response_json['choices']:
                    raise Exception(f"Invalid API response format: missing choices - {response_json}")
                
                if 'message' not in response_json['choices'][0] or 'content' not in response_json['choices'][0]['message']:
                    raise Exception(f"Invalid API response structure: {response_json}")
                
                content = response_json['choices'][0]['message']['content']
                logging.debug(f"Extracted content: {content}")
                
                # Clean and validate JSON string
                content = content.strip()
                if not content.startswith('{'):
                    # Try to find the start of a JSON object in the content
                    json_start = content.find('{')
                    if json_start != -1:
                        content = content[json_start:]
                    else:
                        raise Exception(f"Response content is not a JSON object: {content}")
                
                # Try to parse JSON with improved error handling
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as e:
                    # If JSON parsing fails, attempt to fix common issues
                    logging.warning(f"Initial JSON parse failed, attempting to fix content: {content}")
                    
                    # 1. Remove all newlines and extra spaces
                    content = re.sub(r'\s+', ' ', content).strip()
                    
                    # 2. Ensure quotes are correctly paired
                    content = re.sub(r'(?<!\\)"', '\\"', content)
                    content = content.replace('\\"', '"')  # Reset all quotes
                    content = re.sub(r'([^\\])"([^"]*?)"', r'\1"\2"', content)  # Fix quote pairing
                    
                    # 3. Fix common JSON syntax errors
                    content = content.replace("'", '"')  # Replace single quotes with double quotes
                    content = re.sub(r',\s*}', '}', content)  # Remove trailing comma from objects
                    content = re.sub(r',\s*]', ']', content)  # Remove trailing comma from arrays
                    
                    logging.debug(f"Cleaned content: {content}")
                    
                    try:
                        result = json.loads(content)
                        logging.info(f"Successfully fixed and parsed JSON content")
                    except json.JSONDecodeError as e2:
                        if attempt < max_retries - 1:
                            logging.warning("Retrying due to JSON parse error after cleanup...")
                            time.sleep(retry_delay)
                            continue
                        raise Exception(f"Failed to parse API response as JSON after cleanup: {str(e2)}\nContent: {content}")
                
                # Validate model-specific response format
                try:
                    self._validate_model_response(result, model_key)
                    return result
                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Invalid response format: {str(e)}, retrying...")
                        time.sleep(retry_delay)
                        continue
                    raise
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"All attempts failed: {str(e)}")
                    raise
    
    def _validate_model_response(self, result: Dict, model_key: str) -> None:
        """Validate model-specific response format"""
        if model_key == "model_a":
            if not isinstance(result, dict) or 'analysis' not in result or not isinstance(result['analysis'], list):
                raise Exception("Invalid Model A response format: missing 'analysis' array")
            if not result['analysis']:
                raise Exception("Empty analysis array in Model A response")
            # Validate each analysis result
            for item in result['analysis']:
                if not isinstance(item, dict):
                    raise Exception(f"Invalid analysis item format: {item}")
                if 'index' not in item:
                    raise Exception(f"Missing 'index' in analysis item: {item}")
                required_fields = ["P", "I", "C", "O", "S", "matches_criteria", "reasoning"]
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    raise Exception(f"Missing fields in analysis item: {missing_fields}")
                
        elif model_key == "model_b":
            if not isinstance(result, dict) or "reviews" not in result:
                raise Exception("Invalid Model B response format: missing 'reviews' field")
            if not isinstance(result["reviews"], list):
                raise Exception("Invalid Model B response format: 'reviews' is not a list")
            if not result["reviews"]:
                raise Exception("Empty reviews array in Model B response")
            # Validate each result
            for item in result["reviews"]:
                required_fields = ["index", "inclusion_decision", "review_comments", 
                                 "corrected_P", "corrected_I", "corrected_C", 
                                 "corrected_O", "corrected_S"]
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    raise Exception(f"Missing fields in Model B result: {missing_fields}")
                
        else:  # model_c
            if not isinstance(result, dict) or "decisions" not in result:
                raise Exception("Invalid Model C response format: missing 'decisions' field")
            if not isinstance(result["decisions"], list):
                raise Exception("Invalid Model C response format: 'decisions' is not a list")
            if not result["decisions"]:
                raise Exception("Empty decisions array in Model C response")
            # Validate each decision
            for item in result["decisions"]:
                required_fields = ["index", "final_decision", "reasoning"]
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    raise Exception(f"Missing fields in Model C decision: {missing_fields}")
    
    def _call_model_a(self, abstracts: List[Dict]) -> List[Dict]:
        """Batch call Model A for analysis"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                config = self.model_configs["model_a"]
                # Prepare batch data
                abstracts_json = json.dumps([
                    {
                        "index": str(item["index"]),
                        "abstract": item["abstract"]
                    }
                    for item in abstracts
                ], indent=2)
                
                prompt = self.prompts["model_a"].format(
                    abstracts_json=abstracts_json,
                    **self.picos_criteria
                )
                
                logging.debug(f"Model A attempt {attempt + 1}/{max_retries}")
                response = self._call_api(config, prompt, "model_a")
                
                # Validate response format
                if not isinstance(response, dict) or 'analysis' not in response:
                    raise Exception("Invalid Model A response format: missing 'analysis' field")
                if not isinstance(response['analysis'], list):
                    raise Exception("Invalid Model A response format: 'analysis' is not a list")
                if not response['analysis']:
                    raise Exception("Empty analysis array in Model A response")
                
                # Validate each analysis result
                for item in response['analysis']:
                    if not isinstance(item, dict):
                        raise Exception(f"Invalid analysis item format: {item}")
                    if 'index' not in item:
                        raise Exception(f"Missing 'index' in analysis item: {item}")
                    required_fields = ["P", "I", "C", "O", "S", "matches_criteria", "reasoning"]
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        raise Exception(f"Missing fields in analysis item: {missing_fields}")
                
                return response['analysis']
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Model A attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"All Model A attempts failed: {str(e)}")
                    raise
    
    def _call_model_b(self, abstracts: List[Dict], model_a_results: List[Dict]) -> List[Dict]:
        """Batch call Model B for review"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                config = self.model_configs["model_b"]
                # Prepare batch data, including original abstract and Model A analysis
                analyses = []
                for abstract, a_result in zip(abstracts, model_a_results):
                    analyses.append({
                        "index": str(abstract["index"]),
                        "abstract": abstract["abstract"],
                        "model_a_analysis": {
                            "P": a_result["P"],
                            "I": a_result["I"],
                            "C": a_result["C"],
                            "O": a_result["O"],
                            "S": a_result["S"],
                            "matches_criteria": a_result["matches_criteria"],
                            "reasoning": a_result["reasoning"]
                        }
                    })
                
                abstracts_json = json.dumps(analyses, indent=2)
                prompt = self.prompts["model_b"].format(
                    abstracts_json=abstracts_json,
                    **self.picos_criteria
                )
                
                logging.debug(f"Model B attempt {attempt + 1}/{max_retries}")
                response = self._call_api(config, prompt, "model_b")
                
                # Validate response format
                if not isinstance(response, dict) or "reviews" not in response:
                    raise Exception("Invalid Model B response format: missing 'reviews' field")
                if not isinstance(response["reviews"], list):
                    raise Exception("Invalid Model B response format: 'reviews' is not a list")
                if not response["reviews"]:
                    raise Exception("Empty reviews array in Model B response")
                
                # Validate each result
                for item in response["reviews"]:
                    required_fields = ["index", "inclusion_decision", "review_comments", 
                                     "corrected_P", "corrected_I", "corrected_C", 
                                     "corrected_O", "corrected_S"]
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        raise Exception(f"Missing fields in Model B result: {missing_fields}")
                
                return response["reviews"]
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Model B attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"All Model B attempts failed: {str(e)}")
                    raise
    
    def _call_model_c(self, abstracts: List[Dict], model_a_results: List[Dict], model_b_results: List[Dict]) -> List[Dict]:
        """Batch call Model C for arbitration"""
        try:
            # Create index mapping for both Model A and B results
            a_index_map = {str(result.get('index', '')): result for result in model_a_results}
            b_index_map = {str(result.get('index', '')): result for result in model_b_results}
            
            if not a_index_map or not b_index_map:
                logging.error("Empty results from Model A or Model B")
                raise ValueError("Empty results from Model A or Model B")
            
            # 先收集所有分歧
            all_disagreements = []
            disagreement_indices = []  # 记录所有有分歧的索引
            for abstract in abstracts:
                try:
                    idx = str(abstract.get('index', ''))
                    result_a = a_index_map.get(idx)
                    result_b = b_index_map.get(idx)
                    
                    if not result_a or not result_b:
                        logging.warning(f"Missing results for index {idx} (A: {result_a is not None}, B: {result_b is not None})")
                        continue
                    
                    # Get decisions using get() to handle missing keys safely
                    a_decision = result_a.get('matches_criteria')
                    b_decision = result_b.get('inclusion_decision')
                    
                    if a_decision is None or b_decision is None:
                        logging.warning(f"Missing decision values for index {idx} (A: {a_decision}, B: {b_decision})")
                        continue
                    
                    if a_decision != b_decision:
                        disagreement = {
                            'index': idx,
                            'abstract': abstract.get('abstract', ''),
                            'model_a_analysis': {
                                'decision': a_decision,
                                'reasoning': result_a.get('reasoning', ''),
                                'P': result_a.get('P', ''),
                                'I': result_a.get('I', ''),
                                'C': result_a.get('C', ''),
                                'O': result_a.get('O', ''),
                                'S': result_a.get('S', '')
                            },
                            'model_b_analysis': {
                                'decision': b_decision,
                                'reasoning': result_b.get('review_comments', ''),
                                'P': result_b.get('corrected_P', ''),
                                'I': result_b.get('corrected_I', ''),
                                'C': result_b.get('corrected_C', ''),
                                'O': result_b.get('corrected_O', ''),
                                'S': result_b.get('corrected_S', '')
                            }
                        }
                        all_disagreements.append(disagreement)
                        disagreement_indices.append(idx)
                        logging.info(f"Found disagreement at index {idx}: A={a_decision}, B={b_decision}")
                except Exception as e:
                    logging.error(f"Error processing abstract {abstract.get('index', '')}: {str(e)}")
                    continue
            
            if not all_disagreements:
                logging.info("No disagreements found between Model A and B")
                return []  # 返回空列表而不是空的DataFrame
            
            # 记录找到的分歧数量
            logging.info(f"Found {len(all_disagreements)} disagreements to process")
            
            # 按批次处理分歧
            batch_size = self.model_configs["model_c"]["batch_size"]
            all_decisions = []
            
            for i in range(0, len(all_disagreements), batch_size):
                batch_disagreements = all_disagreements[i:i + batch_size]
                logging.info(f"Processing batch {i//batch_size + 1} with {len(batch_disagreements)} disagreements")
                
                try:
                    # Process disagreements with Model C
                    config = self.model_configs["model_c"]
                    disagreements_json = json.dumps(batch_disagreements, indent=2)
                    prompt = self.prompts["model_c"].format(
                        disagreements_json=disagreements_json,
                        **self.picos_criteria
                    )
                    
                    # 记录发送给API的输入内容
                    logging.info("=== Model C API Input ===")
                    logging.info(f"Batch disagreements:\n{disagreements_json}")
                    logging.info(f"Prompt:\n{prompt}")
                    
                    response = self._call_api(config, prompt, "model_c")
                    
                    # 记录API的原始响应
                    logging.info("=== Model C API Response ===")
                    logging.info(f"Response type: {type(response)}")
                    logging.info(f"Raw response:\n{json.dumps(response, indent=2)}")
                    
                    # 尝试从响应中获取decisions
                    if isinstance(response, dict) and 'decisions' in response:
                        decisions = response['decisions']
                        logging.info(f"Extracted decisions:\n{json.dumps(decisions, indent=2)}")
                        if isinstance(decisions, list):
                            for decision in decisions:
                                logging.info(f"Processing decision: {json.dumps(decision, indent=2)}")
                                if isinstance(decision, dict):
                                    # 获取所有字段
                                    index = str(decision.get('index', ''))
                                    final_decision = bool(decision.get('final_decision', False))
                                    
                                    # 处理可能重复的 reasoning 字段
                                    reasoning_values = []
                                    for key, value in decision.items():
                                        if key.startswith('reasoning'):
                                            if isinstance(value, str):
                                                reasoning_values.append(value)
                                    
                                    # 使用最长的 reasoning 值
                                    reasoning = max(reasoning_values, key=len) if reasoning_values else "No reasoning provided"
                                    
                                    # 清理 reasoning 文本
                                    reasoning = ' '.join(reasoning.split())
                                    
                                    processed_decision = {
                                        'index': index,
                                        'final_decision': final_decision,
                                        'reasoning': reasoning
                                    }
                                    logging.info(f"Processed decision: {json.dumps(processed_decision, indent=2)}")
                                    all_decisions.append(processed_decision)
                                else:
                                    logging.warning(f"Invalid decision format: {decision}")
                        else:
                            logging.warning(f"Decisions is not a list: {type(decisions)}")
                    else:
                        logging.warning(f"Response does not contain decisions field or is not a dict: {response}")
                
                except Exception as e:
                    logging.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                    # 为这个批次中的每个分歧创建一个错误结果
                    for disagreement in batch_disagreements:
                        idx = disagreement['index']
                        all_decisions.append({
                            'index': str(idx),
                            'final_decision': bool(a_index_map[idx]['matches_criteria']),
                            'reasoning': f"Error in Model C processing: {str(e)}"
                        })
            
            # 返回决策列表而不是DataFrame
            return all_decisions if all_decisions else []
            
        except Exception as e:
            logging.error(f"Error in Model C processing: {str(e)}")
            return []  # 返回空列表而不是空的DataFrame

# Initialize global analyzer instance
analyzer = PICOSAnalyzer()

def update_picos_criteria(p, i, c, o, s):
    """Update PICOS criteria"""
    try:
        global analyzer
        analyzer.picos_criteria.update({
            "population": p.strip(),
            "intervention": i.strip(),
            "comparison": c.strip(),
            "outcome": o.strip(),
            "study_design": s.strip()
        })
        return "✓ PICOS criteria updated successfully"
    except Exception as e:
        return f"❌ Error updating PICOS criteria: {str(e)}"

def create_gradio_interface():
    """Create Gradio interface"""
    global analyzer, model_results
    
    def parse_nbib(file) -> tuple:
        """Parse NBIB file and return results"""
        if not file or not os.path.exists(file.name):
            return None, "Invalid file"
            
        try:
            records = []
            record = {}
            authors = []
            current_field = None

            with open(file.name, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                return None, "Empty file"

            for line in lines:
                if line.startswith('TI  - '):
                    record['Title'] = line.replace('TI  - ', '').strip()
                    current_field = 'Title'
                elif line.startswith('AB  - '):
                    record['Abstract'] = line.replace('AB  - ', '').strip()
                    current_field = 'Abstract'
                elif line.startswith('AU  - '):
                    authors.append(line.replace('AU  - ', '').strip())
                    current_field = None
                elif line.startswith('LID - '):
                    if '[doi]' in line:
                        doi_part = line.replace('LID - ', '').strip()
                        record['DOI'] = doi_part.replace(' [doi]', '').strip()
                    current_field = None
                elif line.startswith('PMID- '):
                    if record:  # Save the previous record
                        record['Authors'] = '; '.join(authors)
                        records.append(record)
                        record = {}
                        authors = []
                    current_field = None
                elif line.startswith('      ') and current_field in ['Abstract', 'Title']:
                    record[current_field] += ' ' + line.strip()

            # Add the last record
            if record:
                record['Authors'] = '; '.join(authors)
                records.append(record)

            # Create DataFrame and add index
            df = pd.DataFrame(records)
            df.index.name = 'Index'
            
            # Save to CSV
            output_path = os.path.join(DATA_DIR, "extracted_data.csv")
            df.to_csv(output_path)

            # Prepare preview data
            preview = ""
            for i, record in enumerate(records[:3], 0):
                preview += f"\nRecord {i}:\n"
                preview += f"DOI: {record.get('DOI', '')[:50]}\n"
                preview += f"Title: {record.get('Title', '')[:100]}...\n"
                preview += f"Authors: {record.get('Authors', '')[:100]}...\n"
                preview += f"Abstract: {record.get('Abstract', '')[:200]}...\n"
                preview += "-" * 80 + "\n"
            
            preview += f"\nTotal records extracted: {len(records)}"
            
            return output_path, preview
            
        except Exception as e:
            return None, f"Error processing NBIB file: {str(e)}"
    
    def update_model_settings(model_key, api_url, api_key, model_name, temperature, max_tokens, batch_size, threads, prompt):
        """Update model settings"""
        try:
            config = {
                "api_url": api_url,
                "api_key": api_key,
                "model": model_name,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "batch_size": int(batch_size),
                "threads": int(threads)
            }
            analyzer.update_model_config(model_key, config)
            analyzer.update_prompt(model_key, prompt)
            return f"✓ Model settings updated"
        except Exception as e:
            return f"❌ Error updating settings: {str(e)}"
    
    def test_connection(model_key):
        """Test API connection"""
        return analyzer.test_api_connection(model_key)
    
    def process_model(input_file, model_key):
        """Process analysis for a single model"""
        try:
            # Read CSV file and ensure correct index
            df = pd.read_csv(input_file.name, index_col='Index')
            if df.index.name != 'Index':
                df.index.name = 'Index'
            
            if model_key in ["model_b", "model_c"] and not all(k in model_results for k in ["model_a", "model_b"][:{"model_b": 1, "model_c": 2}[model_key]]):
                return None, f"Previous model results required for {model_key.upper()}"
            
            # Start processing
            logging.info(f"Init Model: {model_key.upper()}...")
            results_df = analyzer.process_batch(df, model_key, model_results)
            model_results[model_key] = results_df
            
            # Save results with index
            output_path = os.path.join(DATA_DIR, f"{model_key}_results.csv")
            results_df.to_csv(output_path, index=True)
            
            completion_msg = f"{model_key.upper()} analysis completed: processed {len(df)} rows"
            logging.info(completion_msg)
            return output_path, completion_msg
        except Exception as e:
            error_msg = f"Error in {model_key.upper()}: {str(e)}"
            logging.error(error_msg)
            return None, error_msg
    
    def merge_all_results(input_file):
        """Merge all model results"""
        try:
            if not all(k in model_results for k in ["model_a", "model_b"]):
                return None, "Model A and B results required"
            
            # Read CSV file and ensure correct index
            df = pd.read_csv(input_file.name, index_col='Index')
            if df.index.name != 'Index':
                df.index.name = 'Index'
            
            merged_df = analyzer.merge_results(df, model_results)
            
            # Save results with index
            output_path = os.path.join(DATA_DIR, "final_results.csv")
            merged_df.to_csv(output_path)
            return output_path, "Results merged successfully"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def merge_results_with_files(input_file, model_a_file, model_b_file, model_c_file):
        """Merge all model results from files"""
        if not all([input_file, model_a_file, model_b_file]):
            return None, "Original file, Model A and B results are required"
        
        try:
            # Load all results
            model_a_results = pd.read_csv(model_a_file.name)
            model_b_results = pd.read_csv(model_b_file.name)
            model_c_results = pd.read_csv(model_c_file.name) if model_c_file else None
            
            # Ensure index column exists
            if "Index" not in model_a_results.columns:
                return None, "Model A results missing Index column"
            if "Index" not in model_b_results.columns:
                return None, "Model B results missing Index column"
            if model_c_results is not None and "Index" not in model_c_results.columns:
                return None, "Model C results missing Index column"
            
            # Set index
            model_a_results.set_index("Index", inplace=True)
            model_b_results.set_index("Index", inplace=True)
            if model_c_results is not None:
                model_c_results.set_index("Index", inplace=True)
            
            # Process original file
            df = pd.read_csv(input_file.name)
            if "Index" not in df.columns:
                df["Index"] = df.index.astype(str)
            
            # Validate all required indices exist
            df_indices = set(df["Index"].astype(str))
            missing_a = df_indices - set(model_a_results.index.astype(str))
            missing_b = df_indices - set(model_b_results.index.astype(str))
            
            if missing_a:
                return None, f"Missing Model A results for indices: {', '.join(sorted(missing_a))}"
            if missing_b:
                return None, f"Missing Model B results for indices: {', '.join(sorted(missing_b))}"
            
            # Merge results
            merged_df = df.copy()
            merged_df.set_index("Index", inplace=True)
            
            # Add Model A results (Decision and Reasoning first)
            merged_df['A_Decision'] = model_a_results['matches_criteria']
            merged_df['A_Reasoning'] = model_a_results['reasoning']
            merged_df['A_P'] = model_a_results['P']
            merged_df['A_I'] = model_a_results['I']
            merged_df['A_C'] = model_a_results['C']
            merged_df['A_O'] = model_a_results['O']
            merged_df['A_S'] = model_a_results['S']
            
            # Add Model B results (Decision and Reasoning first)
            merged_df['B_Decision'] = model_b_results['inclusion_decision']
            merged_df['B_Reasoning'] = model_b_results['review_comments']
            merged_df['B_P'] = model_b_results['corrected_P']
            merged_df['B_I'] = model_b_results['corrected_I']
            merged_df['B_C'] = model_b_results['corrected_C']
            merged_df['B_O'] = model_b_results['corrected_O']
            merged_df['B_S'] = model_b_results['corrected_S']
            
            # Add Model C results (only for cases with disagreement)
            if model_c_results is not None:
                # Initialize C columns
                merged_df['C_Decision'] = None
                merged_df['C_Reasoning'] = None
                
                # Fill C results only for cases with disagreement
                disagreement_mask = merged_df['A_Decision'] != merged_df['B_Decision']
                for idx in merged_df[disagreement_mask].index:
                    if idx in model_c_results.index:
                        merged_df.loc[idx, 'C_Decision'] = model_c_results.loc[idx, 'final_decision']
                        merged_df.loc[idx, 'C_Reasoning'] = model_c_results.loc[idx, 'reasoning']
            
            # Calculate final decision
            def get_final_decision(row):
                # If A and B agree, use their decision
                if row['A_Decision'] == row['B_Decision']:
                    return row['A_Decision']
                # If there's disagreement and C has a decision, use C's decision
                elif pd.notna(row.get('C_Decision')):
                    return row['C_Decision']
                # If there's disagreement but no C decision, return None
                return None
            
            # Apply final decision rule
            merged_df['Final_Decision'] = merged_df.apply(get_final_decision, axis=1)
            
            # Save results
            output_path = os.path.join(DATA_DIR, "final_results.csv")
            merged_df.to_csv(output_path, index=True)
            return output_path, "Results merged successfully"
        except Exception as e:
            return None, f"Error merging results: {str(e)}"
    
    def run_all_models(input_file):
        """Run analysis pipeline for all models"""
        try:
            # Read CSV file and ensure correct index
            df = pd.read_csv(input_file.name, index_col='Index')
            if df.index.name != 'Index':
                df.index.name = 'Index'
            
            total_steps = 4  # A, B, C, and merge
            current_step = 0
            
            def update_progress(step_name, status):
                nonlocal current_step
                current_step += 1
                progress = (current_step / total_steps) * 100
                return f"Step {current_step}/{total_steps} ({progress:.1f}%) - {step_name}: {status}"
            
            # Initialize output variables
            model_a_path = None
            model_b_path = None
            model_c_path = None
            final_path = None
            
            # Run Model A
            logging.info("Starting full analysis pipeline with Model A...")
            model_a_path, model_a_status = process_model(input_file, "model_a")
            if not model_a_path:
                yield model_a_path, model_b_path, model_c_path, final_path, f"Model A failed: {model_a_status}"
                return
            status = update_progress("Model A", "Completed")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            # Run Model B
            logging.info("Starting full analysis pipeline with Model B...")
            model_b_path, model_b_status = process_model(input_file, "model_b")
            if not model_b_path:
                yield model_a_path, model_b_path, model_c_path, final_path, f"Model B failed: {model_b_status}"
                return
            status = update_progress("Model B", "Completed")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            # Run Model C
            logging.info("Starting full analysis pipeline with Model C...")
            model_c_path, model_c_status = process_model(input_file, "model_c")
            if not model_c_path:
                yield model_a_path, model_b_path, model_c_path, final_path, f"Model C failed: {model_c_status}"
                return
            status = update_progress("Model C", "Completed")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            # Create temporary file object
            class TempFile:
                def __init__(self, path):
                    self.name = path
            
            # Merge results
            logging.info("Merging results...")
            final_path, merge_status = merge_results_with_files(
                input_file,
                TempFile(model_a_path),
                TempFile(model_b_path),
                TempFile(model_c_path) if model_c_path else None
            )
            if not final_path:
                yield model_a_path, model_b_path, model_c_path, final_path, f"Merge failed: {merge_status}"
                return
            status = update_progress("Merge", "Completed")
            
            completion_msg = f"All models completed successfully - Processed {len(df)} rows"
            logging.info(completion_msg)
            yield model_a_path, model_b_path, model_c_path, final_path, completion_msg
            
        except Exception as e:
            error_msg = f"Error in pipeline: {str(e)}"
            logging.error(error_msg)
            yield model_a_path, model_b_path, model_c_path, final_path, error_msg
    
    # Create Gradio interface
    with gr.Blocks(title="PICOS Analysis System") as interface:
        gr.Markdown("# PICOS Literature Analysis System")
        gr.Markdown("This system uses a multi-model approach to analyze medical literature abstracts.")
        
        with gr.Tab("NBIB Processing"):
            gr.Markdown("""
            ## NBIB File Processing
            Upload a .nbib file to extract and convert it to CSV format. The extracted data will include:
            - DOI
            - Title
            - Authors
            - Abstract
            """)
            
            with gr.Row():
                nbib_file = gr.File(label="Upload NBIB File", file_types=[".nbib"])
                process_btn = gr.Button("Process NBIB File")
            
            with gr.Row():
                preview = gr.Textbox(label="Preview", lines=20)
                csv_output = gr.File(label="Download CSV")
            
            process_btn.click(
                parse_nbib,
                inputs=[nbib_file],
                outputs=[csv_output, preview]
            )
        
        with gr.Tab("PICOS Criteria"):
            gr.Markdown("""
            ## PICOS Criteria Settings
            Define the standard PICOS criteria that will be used by all models.
            These criteria will be used to evaluate whether each article meets the requirements.
            """)
            
            with gr.Group("Standard PICOS Criteria"):
                population = gr.Textbox(label="Population", value=analyzer.picos_criteria["population"],
                                      placeholder="e.g., patients with hepatocellular carcinoma")
                intervention = gr.Textbox(label="Intervention", value=analyzer.picos_criteria["intervention"],
                                        placeholder="e.g., immunotherapy or targeted therapy")
                comparison = gr.Textbox(label="Comparison", value=analyzer.picos_criteria["comparison"],
                                      placeholder="e.g., standard therapy or placebo")
                outcome = gr.Textbox(label="Outcome", value=analyzer.picos_criteria["outcome"],
                                   placeholder="e.g., survival or response rate")
                study_design = gr.Textbox(label="Study Design", value=analyzer.picos_criteria["study_design"],
                                        placeholder="e.g., randomized controlled trial")
                
                update_picos_btn = gr.Button("Update PICOS Criteria")
                picos_status = gr.Textbox(label="Status")
                
                update_picos_btn.click(
                    update_picos_criteria,
                    inputs=[population, intervention, comparison, outcome, study_design],
                    outputs=picos_status
                )
        
        with gr.Tab("Model Settings"):
            for model_key in ["model_a", "model_b", "model_c"]:
                with gr.Group(f"{model_key.upper()} Settings"):
                    api_url = gr.Textbox(label="API URL", value=analyzer.model_configs[model_key]["api_url"])
                    api_key = gr.Textbox(label="API Key", value=analyzer.model_configs[model_key]["api_key"])
                    model_name = gr.Textbox(label="Model", value=analyzer.model_configs[model_key]["model"])
                    temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, value=analyzer.model_configs[model_key]["temperature"])
                    max_tokens = gr.Number(label="Max Tokens", value=analyzer.model_configs[model_key]["max_tokens"])
                    batch_size = gr.Number(label="Batch Size", value=analyzer.model_configs[model_key]["batch_size"])
                    threads = gr.Slider(label="Threads", minimum=1, maximum=32, step=1, value=analyzer.model_configs[model_key]["threads"])
                    prompt = gr.Textbox(label="Prompt Template", value=analyzer.prompts[model_key], lines=10)
                    
                    update_btn = gr.Button(f"Update {model_key.upper()} Settings")
                    test_btn = gr.Button(f"Test {model_key.upper()} Connection")
                    status = gr.Textbox(label="Status")
                    
                    update_btn.click(
                        update_model_settings,
                        inputs=[gr.Textbox(value=model_key, visible=False), api_url, api_key, model_name,
                               temperature, max_tokens, batch_size, threads, prompt],
                        outputs=status
                    )
                    test_btn.click(
                        test_connection,
                        inputs=[gr.Textbox(value=model_key, visible=False)],
                        outputs=status
                    )
        
        with gr.Tab("Analysis"):
            with gr.Row():
                input_file = gr.File(label="Original CSV File")
                model_a_input = gr.File(label="Model A Results")
                model_b_input = gr.File(label="Model B Results")
                model_c_input = gr.File(label="Model C Results")
            
            with gr.Row():
                model_a_btn = gr.Button("Run Model A")
                model_b_btn = gr.Button("Run Model B")
                model_c_btn = gr.Button("Run Model C")
                merge_btn = gr.Button("Merge Results")
                run_all_btn = gr.Button("Run All", variant="primary")
            
            status = gr.Textbox(label="Status")
            
            with gr.Row():
                model_a_output = gr.File(label="Model A Results")
                model_b_output = gr.File(label="Model B Results")
                model_c_output = gr.File(label="Model C Results")
                final_output = gr.File(label="Final Results")
            
            def load_results(file_path):
                """Load model results"""
                if not file_path:
                    return None
                try:
                    return pd.read_csv(file_path.name)
                except Exception as e:
                    logging.error(f"Error loading results: {str(e)}")
                    return None
            
            def process_model_a(input_file):
                """Process Model A"""
                if not input_file:
                    return None, "Please upload the original CSV file"
                return process_model(input_file, "model_a")
            
            def process_model_b(input_file, model_a_file):
                """Process Model B"""
                if not input_file:
                    return None, "Please upload the original CSV file"
                if not model_a_file:
                    return None, "Please upload Model A results"
                
                try:
                    # Load Model A results
                    model_a_results = pd.read_csv(model_a_file.name)
                    # Ensure index column exists
                    if "Index" not in model_a_results.columns:
                        return None, "Model A results missing Index column"
                    
                    # Set index
                    model_a_results.set_index("Index", inplace=True)
                    model_results["model_a"] = model_a_results
                    
                    # Process original file
                    df = pd.read_csv(input_file.name)
                    # Add index column
                    if "Index" not in df.columns:
                        df["Index"] = df.index.astype(str)
                    
                    # Validate all required indices exist
                    missing_indices = set(df["Index"].astype(str)) - set(model_a_results.index.astype(str))
                    if missing_indices:
                        return None, f"Missing Model A results for indices: {', '.join(sorted(missing_indices))}"
                    
                    return process_model(input_file, "model_b")
                except Exception as e:
                    return None, f"Error processing Model B: {str(e)}"
            
            def process_model_c(input_file, model_a_file, model_b_file):
                """Process Model C"""
                if not input_file:
                    return None, "Please upload the original CSV file"
                if not model_a_file:
                    return None, "Please upload Model A results"
                if not model_b_file:
                    return None, "Please upload Model B results"
                
                try:
                    # Load Model A and B results
                    model_a_results = pd.read_csv(model_a_file.name)
                    model_b_results = pd.read_csv(model_b_file.name)
                    
                    # Ensure index column exists
                    if "Index" not in model_a_results.columns:
                        return None, "Model A results missing Index column"
                    if "Index" not in model_b_results.columns:
                        return None, "Model B results missing Index column"
                    
                    # Set index
                    model_a_results.set_index("Index", inplace=True)
                    model_b_results.set_index("Index", inplace=True)
                    
                    # Process original file
                    df = pd.read_csv(input_file.name)
                    # Add index column
                    if "Index" not in df.columns:
                        df["Index"] = df.index.astype(str)
                    
                    # Validate all required indices exist
                    df_indices = set(df["Index"].astype(str))
                    missing_a = df_indices - set(model_a_results.index.astype(str))
                    missing_b = df_indices - set(model_b_results.index.astype(str))
                    
                    if missing_a:
                        return None, f"Missing Model A results for indices: {', '.join(sorted(missing_a))}"
                    if missing_b:
                        return None, f"Missing Model B results for indices: {', '.join(sorted(missing_b))}"
                    
                    model_results["model_a"] = model_a_results
                    model_results["model_b"] = model_b_results
                    return process_model(input_file, "model_c")
                except Exception as e:
                    return None, f"Error processing Model C: {str(e)}"
            
            def merge_results_with_files(input_file, model_a_file, model_b_file, model_c_file):
                """Merge all model results from files"""
                if not all([input_file, model_a_file, model_b_file]):
                    return None, "Original file, Model A and B results are required"
                
                try:
                    # Load all results
                    model_a_results = pd.read_csv(model_a_file.name)
                    model_b_results = pd.read_csv(model_b_file.name)
                    model_c_results = pd.read_csv(model_c_file.name) if model_c_file else None
                    
                    # Ensure index column exists
                    if "Index" not in model_a_results.columns:
                        return None, "Model A results missing Index column"
                    if "Index" not in model_b_results.columns:
                        return None, "Model B results missing Index column"
                    if model_c_results is not None and "Index" not in model_c_results.columns:
                        return None, "Model C results missing Index column"
                    
                    # Set index
                    model_a_results.set_index("Index", inplace=True)
                    model_b_results.set_index("Index", inplace=True)
                    if model_c_results is not None:
                        model_c_results.set_index("Index", inplace=True)
                    
                    # Process original file
                    df = pd.read_csv(input_file.name)
                    if "Index" not in df.columns:
                        df["Index"] = df.index.astype(str)
                    
                    # Validate all required indices exist
                    df_indices = set(df["Index"].astype(str))
                    missing_a = df_indices - set(model_a_results.index.astype(str))
                    missing_b = df_indices - set(model_b_results.index.astype(str))
                    
                    if missing_a:
                        return None, f"Missing Model A results for indices: {', '.join(sorted(missing_a))}"
                    if missing_b:
                        return None, f"Missing Model B results for indices: {', '.join(sorted(missing_b))}"
                    
                    # Merge results
                    merged_df = df.copy()
                    merged_df.set_index("Index", inplace=True)
                    
                    # Add Model A results (Decision and Reasoning first)
                    merged_df['A_Decision'] = model_a_results['matches_criteria']
                    merged_df['A_Reasoning'] = model_a_results['reasoning']
                    merged_df['A_P'] = model_a_results['P']
                    merged_df['A_I'] = model_a_results['I']
                    merged_df['A_C'] = model_a_results['C']
                    merged_df['A_O'] = model_a_results['O']
                    merged_df['A_S'] = model_a_results['S']
                    
                    # Add Model B results (Decision and Reasoning first)
                    merged_df['B_Decision'] = model_b_results['inclusion_decision']
                    merged_df['B_Reasoning'] = model_b_results['review_comments']
                    merged_df['B_P'] = model_b_results['corrected_P']
                    merged_df['B_I'] = model_b_results['corrected_I']
                    merged_df['B_C'] = model_b_results['corrected_C']
                    merged_df['B_O'] = model_b_results['corrected_O']
                    merged_df['B_S'] = model_b_results['corrected_S']
                    
                    # Add Model C results (only for cases with disagreement)
                    if model_c_results is not None:
                        # Initialize C columns
                        merged_df['C_Decision'] = None
                        merged_df['C_Reasoning'] = None
                        
                        # Fill C results only for cases with disagreement
                        disagreement_mask = merged_df['A_Decision'] != merged_df['B_Decision']
                        for idx in merged_df[disagreement_mask].index:
                            if idx in model_c_results.index:
                                merged_df.loc[idx, 'C_Decision'] = model_c_results.loc[idx, 'final_decision']
                                merged_df.loc[idx, 'C_Reasoning'] = model_c_results.loc[idx, 'reasoning']
                    
                    # Calculate final decision
                    def get_final_decision(row):
                        # If A and B agree, use their decision
                        if row['A_Decision'] == row['B_Decision']:
                            return row['A_Decision']
                        # If there's disagreement and C has a decision, use C's decision
                        elif pd.notna(row.get('C_Decision')):
                            return row['C_Decision']
                        # If there's disagreement but no C decision, return None
                        return None
                    
                    # Apply final decision rule
                    merged_df['Final_Decision'] = merged_df.apply(get_final_decision, axis=1)
                    
                    # Save results
                    output_path = os.path.join(DATA_DIR, "final_results.csv")
                    merged_df.to_csv(output_path, index=True)
                    return output_path, "Results merged successfully"
                except Exception as e:
                    return None, f"Error merging results: {str(e)}"
            
            model_a_btn.click(
                process_model_a,
                inputs=[input_file],
                outputs=[model_a_output, status]
            )
            model_b_btn.click(
                process_model_b,
                inputs=[input_file, model_a_input],
                outputs=[model_b_output, status]
            )
            model_c_btn.click(
                process_model_c,
                inputs=[input_file, model_a_input, model_b_input],
                outputs=[model_c_output, status]
            )
            merge_btn.click(
                merge_results_with_files,
                inputs=[input_file, model_a_input, model_b_input, model_c_input],
                outputs=[final_output, status]
            )
            run_all_btn.click(
                run_all_models,
                inputs=[input_file],
                outputs=[model_a_output, model_b_output, model_c_output, final_output, status]
            )
        
        gr.Markdown("""
        ## Instructions
        1. Start by processing your NBIB file in the "NBIB Processing" tab
        2. Configure model settings in the "Model Settings" tab
        3. Test API connections before running analysis
        4. Upload the generated CSV file in the "Analysis" tab
        5. Run models in sequence: A -> B -> C
        6. Merge results to get the final analysis
        
        ## Input File Format
        The input CSV file should contain at least the following columns:
        - Index: Unique identifier for each abstract
        - Abstract: The text content to be analyzed
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860) 