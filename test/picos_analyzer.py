import os
import json
import pandas as pd
import requests
import logging
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm
import traceback
import threading
import concurrent.futures
import argparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 输入输出文件路径
INPUT_FILE = os.path.join(DATA_DIR, "extracted_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "picos_analysis.csv")

# 中间结果文件路径
MODEL_A_RESULTS = os.path.join(DATA_DIR, "model_a_results.csv")
MODEL_B_RESULTS = os.path.join(DATA_DIR, "model_b_results.csv")
MODEL_C_RESULTS = os.path.join(DATA_DIR, "model_c_results.csv")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件
log_file = os.path.join(LOG_DIR, f"picos_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 批处理配置
BATCH_SIZE = 10
NUM_THREADS = 8  # 线程数配置

# Set up logging
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"picos_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 优化日志配置
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 配置文件日志处理器
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# 配置控制台日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Predefined PICOS criteria
PICOS_CRITERIA = {
    "population": "patients with hepatocellular carcinoma",
    "intervention": "immunotherapy or targeted therapy",
    "comparison": "standard therapy or placebo",
    "outcome": "survival or response rate",
    "study_design": "randomized controlled trial"
}

# Predefined prompts
MODEL_A_PROMPT = """You are a medical research expert analyzing clinical trial abstracts.
Your task is to extract PICOS information from the following batch of article abstracts and determine if each matches specific criteria.

Target PICOS criteria:
- Population: {population}
- Intervention: {intervention}
- Comparison: {comparison}
- Outcome: {outcome}
- Study Design: {study_design}

For each abstract, analyze the PICOS elements and determine if they match the target criteria.
Each abstract is provided in the following format:
{{"index": "ARTICLE_INDEX", "abstract": "ABSTRACT_TEXT"}}

Input abstracts:
{abstracts_json}

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
Be strict in your evaluation and ensure the output is valid JSON format."""

MODEL_B_PROMPT = """You are a critical reviewer in a systematic review team.
Your role is to verify and challenge the initial PICOS analyses.

Articles and their analyses:
{analyses_json}

IMPORTANT: You must follow these strict JSON formatting rules:
1. Use double quotes for all strings
2. Ensure all strings are properly terminated
3. Use commas between array items and object properties
4. Do not use trailing commas
5. Keep the response concise and avoid unnecessary whitespace
6. Escape any special characters in strings
7. Use true/false for inclusion_decision (true means the article matches all PICOS criteria)
8. Use "-" for unchanged fields

Return your analysis in this exact JSON format:
[
    {{
        "index": "ARTICLE_INDEX",
        "inclusion_decision": true/false,
        "review_comments": "brief critical analysis without special characters",
        "corrected_P": "-" or "brief corrected population",
        "corrected_I": "-" or "brief corrected intervention",
        "corrected_C": "-" or "brief corrected comparison",
        "corrected_O": "-" or "brief corrected outcome",
        "corrected_S": "-" or "brief corrected study design"
    }},
    ...
]

Keep all descriptions brief and focused. Do not include line breaks or special characters in the text fields.
If a correction is not needed, use exactly "-" as the value.
Be thorough and objective in your review."""

MODEL_C_PROMPT = """You are the final arbitrator in a systematic review process.
Your task is to resolve disagreements between previous analyses.

Target PICOS criteria:
- Population: {population}
- Intervention: {intervention}
- Comparison: {comparison}
- Outcome: {outcome}
- Study Design: {study_design}

Articles with disagreements:
{disagreements_json}

IMPORTANT: You must follow these strict JSON formatting rules:
1. Use double quotes for all strings
2. Ensure all strings are properly terminated
3. Use commas between array items and object properties
4. Do not use trailing commas
5. Keep the response concise and avoid unnecessary whitespace
6. Escape any special characters in strings
7. Use true/false for final_decision (true means the article matches all PICOS criteria)

Return your decisions in this EXACT JSON format (no other text allowed):
{{
  "decisions": [
    {{
      "index": "ARTICLE_INDEX",
      "final_decision": true/false,
      "reasoning": "brief reasoning without special characters"
    }},
    ...
  ]
}}

Keep all descriptions brief and focused. Do not include line breaks or special characters in the text fields.
Be thorough and objective in your final judgment."""

# API Configuration
MODEL_A_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "api_url": "https://api.deepseek.com/v1/chat/completions",  # Deepseek API
    "model": "deepseek-chat",
    "name": "Model A (Primary Analyzer)",
    "temperature": 0.3,
    "max_tokens": 2000,
    "timeout": 60
}

MODEL_B_CONFIG = {
    "api_key": os.getenv("QWEN_API_KEY"),
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",  # 通义千问 API
    "model": "qwen-plus",
    "name": "Model B (Critical Reviewer)",
    "temperature": 0.3,
    "max_tokens": 2000,
    "timeout": 60
}

MODEL_C_CONFIG = {
    "api_key": os.getenv("GPTGE_API_KEY"),
    "api_url": "https://api.gpt.ge/v1/chat/completions",  # GPT.GE API
    "model": "gpt-4o-mini",
    "name": "Model C (Final Arbitrator)",
    "temperature": 0.3,
    "max_tokens": 2000,
    "timeout": 60
}

# Custom Exceptions
class PICOSAnalyzerError(Exception):
    """Base exception class for PICOS Analyzer"""
    pass

class APIConfigError(PICOSAnalyzerError):
    """Raised when there are issues with API configuration"""
    pass

class APICallError(PICOSAnalyzerError):
    """Raised when API calls fail"""
    pass

class ModelResponseError(PICOSAnalyzerError):
    """Raised when model responses are invalid"""
    pass

class DataProcessingError(PICOSAnalyzerError):
    """Raised when there are issues processing data"""
    pass

class BatchProcessingError(PICOSAnalyzerError):
    """Raised when batch processing fails"""
    pass

# LLM API Client
class LLMAPIClient:
    @staticmethod
    def call_api(config: Dict[str, str], messages: list) -> Dict[str, Any]:
        try:
            # 验证API配置
            if not config.get('api_key') or not config.get('api_url') or not config.get('model'):
                raise APIConfigError(f"Invalid API configuration for {config.get('name', 'Unknown Model')}")

            # 通用请求头和数据
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config['api_key']}"
            }
            
            data = {
                "model": config["model"],
                "messages": messages,
                "temperature": config.get("temperature", 0.3),
                "max_tokens": config.get("max_tokens", 2000),
                "response_format": {"type": "json_object"}
            }
            
            try:
                logging.debug(f"Sending request to {config['name']}: {json.dumps(data)}")
                response = requests.post(
                    config["api_url"],
                    headers=headers,
                    json=data,
                    timeout=config.get("timeout", 60)
                )
                logging.debug(f"Received response from {config['name']}: {response.text}")
            except requests.exceptions.Timeout:
                raise APICallError(f"API call timed out for {config['name']}")
            except requests.exceptions.RequestException as e:
                raise APICallError(f"API call failed for {config['name']}: {str(e)}")
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    # 根据不同API调整响应格式
                    if "deepseek" in config['api_url']:
                        if 'choices' not in response_json or not response_json['choices']:
                            raise ModelResponseError(f"Invalid Deepseek response format: {response_json}")
                        return {
                            "choices": [{
                                "message": {
                                    "content": response_json["choices"][0]["message"]["content"]
                                }
                            }]
                        }
                    elif "dashscope" in config['api_url']:
                        # 通义千问兼容模式已经返回标准格式
                        if 'choices' not in response_json or not response_json['choices']:
                            raise ModelResponseError(f"Invalid Dashscope response format: {response_json}")
                        return response_json
                    else:
                        if 'choices' not in response_json or not response_json['choices']:
                            raise ModelResponseError(f"Invalid GPT.GE response format: {response_json}")
                        return response_json
                except json.JSONDecodeError as e:
                    logging.error(f"Response text: {response.text}")
                    raise ModelResponseError(f"Failed to parse API response for {config['name']}: {str(e)}")
                except KeyError as e:
                    logging.error(f"Response JSON: {response_json}")
                    raise ModelResponseError(f"Unexpected response format from {config['name']}: {str(e)}")
            elif response.status_code == 401:
                raise APIConfigError(f"Invalid API key for {config['name']}")
            elif response.status_code == 429:
                raise APICallError(f"Rate limit exceeded for {config['name']}")
            else:
                raise APICallError(f"API call failed for {config['name']} with status {response.status_code}: {response.text}")
                
        except Exception as e:
            if not isinstance(e, PICOSAnalyzerError):
                e = APICallError(f"Unexpected error in API call for {config['name']}: {str(e)}")
            logging.error(str(e))
            raise e

    @staticmethod
    def test_api_connection(config: Dict[str, str]) -> bool:
        """测试API连接是否正常"""
        try:
            test_message = [{
                "role": "user",
                "content": "Return this exact JSON: {\"test\": \"success\"}"
            }]
            response = LLMAPIClient.call_api(config, test_message)
            content = response['choices'][0]['message']['content']
            parsed = json.loads(content)
            return True
        except Exception as e:
            logging.error(f"API test failed for {config['name']}: {str(e)}")
            return False

# Model Clients
class ModelAClient:
    @staticmethod
    def analyze_picos_batch(abstracts: List[Dict[str, str]]) -> List[Dict]:
        """批量分析文章的PICOS信息"""
        try:
            # 调试: 打印输入数据
            logging.debug("Model A Input Data:")
            logging.debug(f"Number of abstracts: {len(abstracts)}")
            for abstract in abstracts:
                logging.debug(f"Index: {abstract['index']}, Abstract length: {len(abstract['Abstract'])}")

            abstracts_json = json.dumps([
                {"index": item['index'], "abstract": item['Abstract']} 
                for item in abstracts
            ], indent=2)
            
            # 调试: 打印完整的请求内容
            prompt_content = MODEL_A_PROMPT.format(
                abstracts_json=abstracts_json,
                **PICOS_CRITERIA
            )
            logging.debug("Model A Prompt:")
            logging.debug(prompt_content)
            
            messages = [{
                "role": "user",
                "content": prompt_content
            }]
            
            # 调试: 打印API请求前的信息
            logging.debug("Sending request to Model A API...")
            response = LLMAPIClient.call_api(MODEL_A_CONFIG, messages)
            
            # 调试: 打印原始响应
            logging.debug("Model A Raw Response:")
            logging.debug(json.dumps(response, indent=2))
            
            content = response['choices'][0]['message']['content']
            logging.debug("Model A Response Content:")
            logging.debug(content)
            
            # 验证和清理响应内容
            content = content.strip()
            
            # 尝试解析 JSON
            try:
                parsed_response = json.loads(content)
            except json.JSONDecodeError as e:
                logging.warning(f"Initial JSON parse failed: {str(e)}, attempting to fix...")
                fixed_content = fix_json(content)
                try:
                    parsed_response = json.loads(fixed_content)
                    logging.info("Successfully fixed and parsed JSON response")
                except json.JSONDecodeError as e2:
                    logging.error(f"Failed to parse fixed JSON: {fixed_content}")
                    logging.error(f"Original error: {str(e)}")
                    logging.error(f"Fix attempt error: {str(e2)}")
                    raise ModelResponseError(f"Could not parse Model A response: {str(e2)}")
            
            if not isinstance(parsed_response, dict) or 'analysis' not in parsed_response:
                raise ModelResponseError("Invalid response format from Model A")
            
            # 验证每个分析结果
            for item in parsed_response['analysis']:
                validate_model_response(item, "Model A", item.get('index', 'unknown'))
            
            return parsed_response['analysis']
            
        except Exception as e:
            logging.error(f"Error in analyze_picos_batch: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            raise

class ModelBClient:
    @staticmethod
    def review_analysis_batch(abstracts: List[Dict[str, str]], model_a_results: List[Dict]) -> List[Dict]:
        """批量审查 Model A 的分析结果"""
        try:
            # 调试: 打印输入数据
            logging.debug("Model B Input Data:")
            logging.debug(f"Number of abstracts: {len(abstracts)}")
            logging.debug(f"Number of Model A results: {len(model_a_results)}")
            
            # 准备分析数据
            analyses = []
            for abstract, a_result in zip(abstracts, model_a_results):
                try:
                    analyses.append({
                        "index": a_result['index'],  # 使用 Model A 结果中的索引
                        "abstract": abstract['Abstract'],
                        "model_a_analysis": a_result
                    })
                except Exception as e:
                    logging.warning(f"Error preparing analysis for index {a_result.get('index', 'unknown')}: {str(e)}")
                    continue
            
            if not analyses:
                raise BatchProcessingError("No valid analyses to process")
            
            # 调试: 打印准备好的分析数据
            logging.debug("Prepared analyses for Model B:")
            logging.debug(json.dumps(analyses, indent=2))
            
            # 构建提示
            analyses_json = json.dumps(analyses, indent=2)
            prompt_content = MODEL_B_PROMPT.format(analyses_json=analyses_json)
            
            # 调试: 打印完整的请求内容
            logging.debug("Model B Prompt:")
            logging.debug(prompt_content)
            
            messages = [{
                "role": "user",
                "content": prompt_content
            }]
            
            # 调试: 打印API请求前的信息
            logging.debug("Sending request to Model B API...")
            response = LLMAPIClient.call_api(MODEL_B_CONFIG, messages)
            
            # 调试: 打印原始响应
            logging.debug("Model B Raw Response:")
            logging.debug(json.dumps(response, indent=2))
            
            content = response['choices'][0]['message']['content']
            logging.debug("Model B Response Content:")
            logging.debug(content)
            
            # 验证和清理响应内容
            content = content.strip()
            
            # 尝试解析 JSON
            try:
                results = json.loads(content)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Model B response: {str(e)}")
                logging.error(f"Raw content: {content}")
                raise ModelResponseError(f"Invalid JSON response from Model B: {str(e)}")
            
            if not isinstance(results, list):
                raise ModelResponseError("Model B response is not a list")
            
            # 验证每个结果
            for result in results:
                validate_model_response(result, "Model B", result.get('index', 'unknown'))
            
            return results
            
        except Exception as e:
            logging.error(f"Error in review_analysis_batch: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            raise

class ModelCClient:
    @staticmethod
    def resolve_disagreements_batch(disagreements: List[Dict]) -> List[Dict]:
        """批量解决Model A和Model B之间的分歧"""
        if not disagreements:
            return []
        
        try:
            # 调试: 打印输入数据
            logging.debug("Model C Input Data:")
            logging.debug(f"Number of disagreements: {len(disagreements)}")
            
            # 准备数据
            formatted_disagreements = []
            for d in disagreements:
                formatted_disagreements.append({
                    'index': d['index'],
                    'abstract': d['abstract'],
                    'model_a_decision': d['model_a_analysis']['decision'],
                    'model_a_reasoning': d['model_a_analysis']['reasoning'],
                    'model_a_picos': {
                        'P': d['model_a_analysis']['P'],
                        'I': d['model_a_analysis']['I'],
                        'C': d['model_a_analysis']['C'],
                        'O': d['model_a_analysis']['O'],
                        'S': d['model_a_analysis']['S']
                    },
                    'model_b_decision': d['model_b_analysis']['decision'],
                    'model_b_reasoning': d['model_b_analysis']['reasoning'],
                    'model_b_picos': {
                        'P': d['model_b_analysis']['P'],
                        'I': d['model_b_analysis']['I'],
                        'C': d['model_b_analysis']['C'],
                        'O': d['model_b_analysis']['O'],
                        'S': d['model_b_analysis']['S']
                    }
                })
            
            disagreements_json = json.dumps(formatted_disagreements, indent=2)
            
            # 调试: 打印完整的请求内容
            logging.debug("Model C Prompt:")
            logging.debug(MODEL_C_PROMPT.format(
                disagreements_json=disagreements_json,
                **PICOS_CRITERIA
            ))
            
            messages = [{
                "role": "user",
                "content": MODEL_C_PROMPT.format(
                    disagreements_json=disagreements_json,
                    **PICOS_CRITERIA
                )
            }]
            
            response = LLMAPIClient.call_api(MODEL_C_CONFIG, messages)
            content = response['choices'][0]['message']['content']
            
            # 调试: 打印原始响应
            logging.debug("Model C Raw Response:")
            logging.debug(content)
            
            # 确保内容是有效的JSON
            try:
                results = json.loads(content)
                
                # 验证结果格式
                if not isinstance(results, dict) or 'decisions' not in results:
                    logging.error(f"Invalid response format from Model C: missing 'decisions' field")
                    return []
                
                if not isinstance(results['decisions'], list):
                    logging.error(f"Invalid response format from Model C: 'decisions' is not a list")
                    return []
                
                validated_results = []
                for result in results['decisions']:
                    if not isinstance(result, dict):
                        logging.error(f"Invalid result format from Model C: not a dict - {result}")
                        continue
                        
                    if 'index' not in result or 'final_decision' not in result or 'reasoning' not in result:
                        logging.error(f"Missing required fields in Model C result: {result}")
                        continue
                        
                    # 确保数据类型正确
                    try:
                        validated_result = {
                            'index': str(result['index']),
                            'final_decision': bool(result['final_decision']),
                            'reasoning': str(result['reasoning'])
                        }
                        validated_results.append(validated_result)
                    except (ValueError, TypeError) as e:
                        logging.error(f"Error converting types in Model C result: {str(e)}")
                        continue
                
                if not validated_results:
                    logging.error("No valid results after validation")
                    return []
                    
                return validated_results
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Model C response: {content}")
                logging.error(f"JSON decode error: {str(e)}")
                return []
                
        except Exception as e:
            logging.error(f"Error in resolve_disagreements_batch: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            return []

def validate_model_response(response: Dict, model_name: str, index: str) -> None:
    """Validate model response format and content"""
    required_fields = {
        "Model A": ["index", "P", "I", "C", "O", "S", "matches_criteria", "reasoning"],
        "Model B": ["index", "inclusion_decision", "review_comments", "corrected_P", 
                   "corrected_I", "corrected_C", "corrected_O", "corrected_S"],
        "Model C": ["index", "final_decision", "reasoning"]
    }
    
    try:
        # Check required fields
        for field in required_fields[model_name]:
            if field not in response:
                raise ModelResponseError(
                    f"Missing required field '{field}' in {model_name} response for index {index}")
        
        # Validate specific fields based on model
        if model_name == "Model A":
            if not isinstance(response["matches_criteria"], bool):
                raise ModelResponseError(
                    f"Invalid 'matches_criteria' type in Model A response for index {index}")
        elif model_name == "Model B":
            if not isinstance(response["inclusion_decision"], bool):
                raise ModelResponseError(
                    f"Invalid 'inclusion_decision' type in Model B response for index {index}")
        elif model_name == "Model C":
            if not isinstance(response["final_decision"], bool):
                raise ModelResponseError(
                    f"Invalid 'final_decision' type in Model C response for index {index}")
    
    except KeyError as e:
        raise ModelResponseError(f"Missing field in {model_name} response: {str(e)}")
    except Exception as e:
        if not isinstance(e, ModelResponseError):
            raise ModelResponseError(f"Error validating {model_name} response: {str(e)}")
        raise e

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """预处理数据，移除无效条目并确保数据类型正确"""
    logging.info("Preprocessing data...")
    # 创建新的 DataFrame 而不是修改视图
    processed_df = df.copy()
    # 移除 Abstract 为空的行
    processed_df = processed_df.dropna(subset=['Abstract'])
    # 确保 Abstract 是字符串类型
    processed_df.loc[:, 'Abstract'] = processed_df['Abstract'].astype(str)
    # 移除 Abstract 内容过短的行
    processed_df = processed_df[processed_df['Abstract'].str.len() > 50]
    # 重置索引以确保连续性
    processed_df = processed_df.reset_index(drop=True)
    logging.info(f"Successfully loaded and preprocessed {len(processed_df)} valid articles")
    return processed_df

def handle_exception(e: Exception, context: str) -> None:
    """处理异常并记录错误信息"""
    if not isinstance(e, PICOSAnalyzerError):
        e = DataProcessingError(f"Unexpected error in {context}: {str(e)}")
    logging.error(str(e))
    logging.error(f"Full traceback: {traceback.format_exc()}")
    raise e

# 全局进度跟踪
progress = {
    'A': {'processed': 0, 'total': 0},
    'B': {'processed': 0, 'total': 0},
    'C': {'processed': 0, 'total': 0}
}

def update_progress(desc: str, current: int, total: int):
    """使用tqdm风格的进度条显示进度"""
    bar_width = 50
    filled = int(round(bar_width * current / float(total)))
    bar = '=' * filled + '-' * (bar_width - filled)
    percent = round(100.0 * current / float(total), 1)
    print(f'\r{desc}: [{bar}] {current}/{total} {percent}%', end='', flush=True)

def process_batches_with_threads(batches: List[tuple], process_func, num_threads: int, desc: str) -> List[Dict]:
    """使用多线程处理批次"""
    results = []
    results_lock = threading.Lock()
    total_items = sum(len(batch[0]) for batch in batches)
    processed_items = 0
    
    def process_batch(batch_data):
        nonlocal processed_items
        try:
            batch_results = process_func(batch_data)
            if batch_results:
                with results_lock:
                    results.extend(batch_results)
                    nonlocal processed_items
                    processed_items += len(batch_results)
                    update_progress(desc, processed_items, total_items)
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(process_batch, batch)
            futures.append(future)
        
        # 等待所有任务完成
        concurrent.futures.wait(futures)
    
    print()  # 换行
    return results

def run_model_a_analysis(df: pd.DataFrame) -> None:
    """运行 Model A 分析并保存结果"""
    logging.info("\nStarting Model A analysis...")
    
    # 准备批次
    batches = []
    for i in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[i:i+BATCH_SIZE].copy()
        abstracts = []
        for idx, row in batch_df.iterrows():
            if len(str(row['Abstract']).strip()) > 50:
                abstracts.append({
                    'index': str(idx),
                    'Abstract': str(row['Abstract']).strip()
                })
        if abstracts:
            batches.append((abstracts, i))
    
    # 运行分析
    model_a_results = process_batches_with_threads(
        batches,
        lambda batch: ModelAClient.analyze_picos_batch(batch),
        num_threads=NUM_THREADS,
        desc="Model A Progress"
    )
    
    if not model_a_results:
        raise DataProcessingError("No results from Model A")
    
    # 保存结果
    model_a_df = pd.DataFrame(model_a_results)
    model_a_df.to_csv(MODEL_A_RESULTS, index=False)
    logging.info(f"Model A results saved to {MODEL_A_RESULTS}")

def run_model_b_analysis(df: pd.DataFrame) -> None:
    """运行 Model B 分析并保存结果"""
    logging.info("\nStarting Model B analysis...")
    
    # 检查 Model A 结果文件
    if not os.path.exists(MODEL_A_RESULTS):
        raise DataProcessingError(f"Model A results not found at {MODEL_A_RESULTS}")
    
    # 加载 Model A 结果
    model_a_results = pd.read_csv(MODEL_A_RESULTS).to_dict('records')
    
    # 准备批次
    b_batches = []
    for i in range(0, len(model_a_results), BATCH_SIZE):
        batch_a_results = model_a_results[i:i+BATCH_SIZE]
        batch_abstracts = []
        for result in batch_a_results:
            abstract_idx = int(result['index'])
            abstract_data = {
                'index': result['index'],
                'Abstract': df.iloc[abstract_idx]['Abstract']
            }
            batch_abstracts.append(abstract_data)
        if batch_abstracts:
            b_batches.append((batch_abstracts, batch_a_results))
    
    # 运行分析
    model_b_results = process_batches_with_threads(
        b_batches,
        lambda batch: ModelBClient.review_analysis_batch(batch[0], batch[1]),  # 修复这里：正确传递两个参数
        num_threads=NUM_THREADS,
        desc="Model B Progress"
    )
    
    if not model_b_results:
        raise DataProcessingError("No results from Model B")
    
    # 保存结果
    model_b_df = pd.DataFrame(model_b_results)
    model_b_df.to_csv(MODEL_B_RESULTS, index=False)
    logging.info(f"Model B results saved to {MODEL_B_RESULTS}")

def run_model_c_analysis(df: pd.DataFrame) -> None:
    """运行 Model C 分析并保存结果"""
    logging.info("\nStarting Model C analysis...")
    
    # 检查必要的结果文件
    if not os.path.exists(MODEL_A_RESULTS):
        raise DataProcessingError(f"Model A results not found at {MODEL_A_RESULTS}")
    if not os.path.exists(MODEL_B_RESULTS):
        raise DataProcessingError(f"Model B results not found at {MODEL_B_RESULTS}")
    
    # 加载之前的结果
    model_a_df = pd.read_csv(MODEL_A_RESULTS)
    model_b_df = pd.read_csv(MODEL_B_RESULTS)
    
    # 确保索引列是字符串类型，布尔值是Python原生布尔类型
    model_a_df['index'] = model_a_df['index'].astype(str)
    model_b_df['index'] = model_b_df['index'].astype(str)
    model_a_df['matches_criteria'] = model_a_df['matches_criteria'].astype(bool)
    model_b_df['inclusion_decision'] = model_b_df['inclusion_decision'].astype(bool)
    
    # 收集分歧
    disagreements = []
    for idx in model_a_df['index'].unique():
        try:
            a_result = model_a_df[model_a_df['index'] == idx].iloc[0]
            b_result = model_b_df[model_b_df['index'] == idx].iloc[0]
            
            # 将Series转换为dict，并确保布尔值是Python原生类型
            a_dict = a_result.to_dict()
            b_dict = b_result.to_dict()
            a_dict['matches_criteria'] = bool(a_dict['matches_criteria'])
            b_dict['inclusion_decision'] = bool(b_dict['inclusion_decision'])
            
            if a_dict['matches_criteria'] != b_dict['inclusion_decision']:
                abstract = df.iloc[int(idx)]['Abstract']
                disagreements.append({
                    'index': idx,
                    'abstract': abstract,
                    'model_a_analysis': {
                        'decision': bool(a_dict['matches_criteria']),
                        'reasoning': str(a_dict['reasoning']),
                        'P': str(a_dict['P']),
                        'I': str(a_dict['I']),
                        'C': str(a_dict['C']),
                        'O': str(a_dict['O']),
                        'S': str(a_dict['S'])
                    },
                    'model_b_analysis': {
                        'decision': bool(b_dict['inclusion_decision']),
                        'reasoning': str(b_dict['review_comments']),
                        'P': str(b_dict['corrected_P']),
                        'I': str(b_dict['corrected_I']),
                        'C': str(b_dict['corrected_C']),
                        'O': str(b_dict['corrected_O']),
                        'S': str(b_dict['corrected_S'])
                    }
                })
        except Exception as e:
            logging.error(f"Error processing disagreement for index {idx}: {str(e)}")
            continue
    
    if not disagreements:
        logging.info("No disagreements found between Model A and B")
        # 创建一个空的结果文件，表示没有分歧需要解决
        pd.DataFrame(columns=['index', 'final_decision', 'reasoning']).to_csv(MODEL_C_RESULTS, index=False)
        return
    
    logging.info(f"Found {len(disagreements)} disagreements between Model A and B")
    
    # 准备批次
    c_batches = []
    for i in range(0, len(disagreements), BATCH_SIZE):
        batch_disagreements = disagreements[i:i+BATCH_SIZE]
        c_batches.append((batch_disagreements, i))
    
    # 运行分析
    all_results = []
    for batch, _ in c_batches:
        try:
            batch_results = ModelCClient.resolve_disagreements_batch(batch)
            if batch_results:
                # 验证每个结果的格式
                for result in batch_results:
                    if not isinstance(result, dict) or 'index' not in result or 'final_decision' not in result or 'reasoning' not in result:
                        logging.error(f"Invalid result format from Model C: {result}")
                        continue
                    # 确保final_decision是布尔值
                    result['final_decision'] = bool(result['final_decision'])
                    all_results.append(result)
        except Exception as e:
            logging.error(f"Error processing batch in Model C: {str(e)}")
            continue
    
    if all_results:
        # 保存结果
        model_c_df = pd.DataFrame(all_results)
        model_c_df.to_csv(MODEL_C_RESULTS, index=False)
        logging.info(f"Model C results saved to {MODEL_C_RESULTS}")
    else:
        logging.warning("No valid results from Model C, creating empty results file")
        pd.DataFrame(columns=['index', 'final_decision', 'reasoning']).to_csv(MODEL_C_RESULTS, index=False)

def merge_results(df: pd.DataFrame) -> pd.DataFrame:
    """合并所有模型的结果"""
    logging.info("\nMerging results...")
    
    # 检查必要的结果文件
    if not os.path.exists(MODEL_A_RESULTS):
        raise DataProcessingError(f"Model A results not found at {MODEL_A_RESULTS}")
    if not os.path.exists(MODEL_B_RESULTS):
        raise DataProcessingError(f"Model B results not found at {MODEL_B_RESULTS}")
    
    # 加载所有结果
    try:
        model_a_df = pd.read_csv(MODEL_A_RESULTS)
        model_b_df = pd.read_csv(MODEL_B_RESULTS)
        
        # 检查Model C结果文件
        model_c_df = None
        if os.path.exists(MODEL_C_RESULTS):
            try:
                model_c_df = pd.read_csv(MODEL_C_RESULTS)
                # 验证Model C结果文件的格式
                required_columns = ['index', 'final_decision', 'reasoning']
                if not all(col in model_c_df.columns for col in required_columns):
                    logging.warning("Model C results file has invalid format, proceeding without Model C results")
                    model_c_df = None
            except Exception as e:
                logging.warning(f"Failed to load Model C results: {str(e)}, proceeding without Model C results")
                model_c_df = None
    except Exception as e:
        logging.error(f"Error loading result files: {str(e)}")
        raise DataProcessingError(f"Failed to load result files: {str(e)}")

    # 确保索引列是字符串类型
    model_a_df['index'] = model_a_df['index'].astype(str)
    model_b_df['index'] = model_b_df['index'].astype(str)
    if model_c_df is not None and 'index' in model_c_df.columns:
        model_c_df['index'] = model_c_df['index'].astype(str)
    
    all_results = []
    processed_count = 0
    for idx, row in df.iterrows():
        try:
            str_idx = str(idx)
            
            # 在Model A结果中查找
            a_result = model_a_df[model_a_df['index'] == str_idx].to_dict('records')
            a_result = a_result[0] if a_result else None
            
            # 在Model B结果中查找
            b_result = model_b_df[model_b_df['index'] == str_idx].to_dict('records')
            b_result = b_result[0] if b_result else None

            if not a_result or not b_result:
                logging.warning(f"Missing Model A or B results for index {idx}")
                continue

            result_row = {
                'Index': str_idx,
                'Title': row['Title'],
                'Authors': row['Authors'],
                'Abstract': row['Abstract'],
                'A_Decision': a_result['matches_criteria'],
                'A_Reason': a_result['reasoning'],
                'A_P': a_result['P'],
                'A_I': a_result['I'],
                'A_C': a_result['C'],
                'A_O': a_result['O'],
                'A_S': a_result['S'],
                'B_Decision': b_result['inclusion_decision'],
                'B_Reason': b_result['review_comments'],
                'B_P': b_result['corrected_P'],
                'B_I': b_result['corrected_I'],
                'B_C': b_result['corrected_C'],
                'B_O': b_result['corrected_O'],
                'B_S': b_result['corrected_S']
            }

            # 如果有Model C的结果且格式正确
            if model_c_df is not None and 'index' in model_c_df.columns:
                c_result = model_c_df[model_c_df['index'] == str_idx].to_dict('records')
                c_result = c_result[0] if c_result else None
                
                if c_result:
                    result_row.update({
                        'C_Decision': c_result['final_decision'],
                        'C_Reason': c_result['reasoning'],
                        'Final_Decision': c_result['final_decision']
                    })
                else:
                    # 如果A和B一致，使用它们的共同决定
                    if a_result['matches_criteria'] == b_result['inclusion_decision']:
                        result_row.update({
                            'C_Decision': None,
                            'C_Reason': 'No disagreement between Model A and B',
                            'Final_Decision': a_result['matches_criteria']
                        })
                    else:
                        # 如果A和B不一致但没有C的结果，使用B的决定
                        result_row.update({
                            'C_Decision': None,
                            'C_Reason': 'Disagreement not resolved',
                            'Final_Decision': b_result['inclusion_decision']
                        })
            else:
                # 没有Model C结果时的处理
                if a_result['matches_criteria'] == b_result['inclusion_decision']:
                    result_row.update({
                        'C_Decision': None,
                        'C_Reason': 'No disagreement between Model A and B',
                        'Final_Decision': a_result['matches_criteria']
                    })
                else:
                    result_row.update({
                        'C_Decision': None,
                        'C_Reason': 'Disagreement not resolved',
                        'Final_Decision': b_result['inclusion_decision']
                    })

            all_results.append(result_row)
            processed_count += 1
            if processed_count % 50 == 0:
                logging.info(f"Processed {processed_count} articles...")

        except Exception as e:
            logging.error(f"Error processing article at index {idx}: {str(e)}")
            continue

    if not all_results:
        raise DataProcessingError("No articles were successfully processed")

    logging.info(f"Successfully processed {len(all_results)} articles")
    output_df = pd.DataFrame(all_results)
    output_df.set_index('Index', inplace=True)
    return output_df

def main():
    try:
        parser = argparse.ArgumentParser(description='PICOS Analysis Pipeline')
        parser.add_argument('--step', type=str, choices=['A', 'B', 'C', 'merge', 'all'],
                          default='all',  # 设置默认值为 'all'
                          help='Specify which step to run: A, B, C, merge, or all (default: all)')
        args = parser.parse_args()

        # 测试相关API连接
        if args.step in ['A', 'all']:
            print("\nTesting Model A API connection...")
            if not LLMAPIClient.test_api_connection(MODEL_A_CONFIG):
                raise APIConfigError("Model A API test failed")
            print("Model A (Deepseek): ✓")
            
        if args.step in ['B', 'all']:
            print("\nTesting Model B API connection...")
            if not LLMAPIClient.test_api_connection(MODEL_B_CONFIG):
                raise APIConfigError("Model B API test failed")
            print("Model B (通义千问): ✓")
            
        if args.step in ['C', 'all']:
            print("\nTesting Model C API connection...")
            if not LLMAPIClient.test_api_connection(MODEL_C_CONFIG):
                raise APIConfigError("Model C API test failed")
            print("Model C (GPT.GE): ✓")

        print("\nStarting PICOS analysis")
        print(f"Using PICOS criteria: {json.dumps(PICOS_CRITERIA, indent=2)}")
        
        # 加载和预处理数据
        df = pd.read_csv(INPUT_FILE)
        df = preprocess_data(df)
        
        # 根据指定的步骤运行相应的分析器
        if args.step == 'A' or args.step == 'all':
            run_model_a_analysis(df)
            
        if args.step == 'B' or args.step == 'all':
            # 检查是否存在Model A的结果
            if not os.path.exists(MODEL_A_RESULTS):
                raise DataProcessingError(f"Model A results not found at {MODEL_A_RESULTS}")
            run_model_b_analysis(df)
            
        if args.step == 'C' or args.step == 'all':
            # 检查是否存在Model A和B的结果
            if not os.path.exists(MODEL_A_RESULTS):
                raise DataProcessingError(f"Model A results not found at {MODEL_A_RESULTS}")
            if not os.path.exists(MODEL_B_RESULTS):
                raise DataProcessingError(f"Model B results not found at {MODEL_B_RESULTS}")
            run_model_c_analysis(df)
            
        if args.step == 'merge' or args.step == 'all':
            # 检查必要的结果文件
            if not os.path.exists(MODEL_A_RESULTS):
                raise DataProcessingError(f"Model A results not found at {MODEL_A_RESULTS}")
            if not os.path.exists(MODEL_B_RESULTS):
                raise DataProcessingError(f"Model B results not found at {MODEL_B_RESULTS}")
            # Model C的结果是可选的
            result_df = merge_results(df)
            result_df.to_csv(OUTPUT_FILE)
            print(f"\nResults merged and saved to {OUTPUT_FILE}")
        
        if args.step == 'all':
            print(f"\nComplete pipeline finished. Final results saved to {OUTPUT_FILE}")
        else:
            print(f"\nStep {args.step} completed successfully")
            
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        print(f"\nError: {str(e)}")
        print(f"Check the log file for details: {log_file}")
        raise

if __name__ == "__main__":
    main() 