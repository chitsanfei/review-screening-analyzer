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

# 加载环境变量
load_dotenv()

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 日志配置
log_file = os.path.join(LOG_DIR, f"picos_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

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
                "threads": 4
            },
            "model_b": {
                "api_key": os.getenv("QWEN_API_KEY", ""),
                "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                "model": "qwen-plus-2024-11-27",
                "name": "Model B (Critical Reviewer)",
                "temperature": 0.3,
                "max_tokens": 2000,
                "batch_size": 10,
                "threads": 4
            },
            "model_c": {
                "api_key": os.getenv("GPTGE_API_KEY", ""),
                "api_url": "https://api.gpt.ge/v1/chat/completions",
                "model": "gpt-4o",
                "name": "Model C (Final Arbitrator)",
                "temperature": 0.3,
                "max_tokens": 2000,
                "batch_size": 10,
                "threads": 4
            }
        }
        
        self.prompts = {
            "model_a": """You are a medical research expert analyzing clinical trial abstracts.
Your task is to extract PICOS information from the following batch of article abstracts and determine if each matches specific criteria.

Target PICOS criteria:
- Population: {population}
- Intervention: {intervention}
- Comparison: {comparison}
- Outcome: {outcome}
- Study Design: {study_design}

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
Be strict in your evaluation and ensure the output is valid JSON format.""",
            "model_b": """You are a critical reviewer in a systematic review team.
Your role is to verify and challenge the initial PICOS analyses.

Articles and their analyses:
{abstracts_json}

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
{{
  "reviews": [
    {{
      "index": "ARTICLE_INDEX",
      "inclusion_decision": true/false,
      "review_comments": "brief critical analysis",
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
If a correction is not needed, use exactly "-" as the value.
Be thorough and objective in your review.""",
            "model_c": """You are the final arbitrator in a systematic review process.
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
        }
        
        self.picos_criteria = {
            "population": "patients with hepatocellular carcinoma",
            "intervention": "immunotherapy or targeted therapy",
            "comparison": "standard therapy or placebo",
            "outcome": "survival or response rate",
            "study_design": "randomized controlled trial"
        }
    
    def update_model_config(self, model_key: str, config: Dict):
        """更新模型配置"""
        self.model_configs[model_key].update(config)
    
    def update_prompt(self, model_key: str, prompt: str):
        """更新模型提示词"""
        self.prompts[model_key] = prompt
    
    def test_api_connection(self, model_key: str) -> str:
        """测试API连接"""
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
        """处理一批数据"""
        config = self.model_configs[model_key]
        batch_size = config["batch_size"]
        threads = config["threads"]
        results = []
        total_rows = len(df)
        completed_rows = 0
        error_count = 0
        
        def update_progress():
            progress = (completed_rows / total_rows) * 100
            status_text = (f"Processing {model_key.upper()}: {completed_rows}/{total_rows} rows ({progress:.1f}%) "
                         f"- Errors: {error_count}")
            if hasattr(update_progress, 'last_status') and update_progress.last_status == status_text:
                return
            update_progress.last_status = status_text
            logging.info(status_text)
        
        def process_batch_data(batch_df: pd.DataFrame) -> List[Dict]:
            nonlocal error_count
            try:
                # 准备批次数据
                abstracts = []
                for idx, row in batch_df.iterrows():
                    if pd.isna(row["Abstract"]) or len(str(row["Abstract"]).strip()) < 50:
                        logging.warning(f"Skipping row {idx}: Abstract too short or empty")
                        error_count += 1
                        continue
                    abstracts.append({
                        "index": str(idx),
                        "abstract": str(row["Abstract"]).strip()
                    })
                
                if not abstracts:
                    logging.warning(f"No valid abstracts in batch")
                    return []
                
                if model_key == "model_a":
                    batch_results = self._call_model_a(abstracts)
                elif model_key == "model_b":
                    if not previous_results or "model_a" not in previous_results:
                        raise Exception("Model A results required for Model B")
                    model_a_results = []
                    for abstract in abstracts:
                        idx = int(abstract["index"])
                        if idx not in previous_results["model_a"].index:
                            raise Exception(f"Missing Model A result for index {idx}")
                        model_a_results.append(previous_results["model_a"].loc[idx].to_dict())
                    batch_results = self._call_model_b(abstracts, model_a_results)
                else:  # model_c
                    if not previous_results or "model_a" not in previous_results or "model_b" not in previous_results:
                        raise Exception("Model A and B results required for Model C")
                    model_a_results = []
                    model_b_results = []
                    for abstract in abstracts:
                        idx = int(abstract["index"])
                        if idx not in previous_results["model_a"].index:
                            raise Exception(f"Missing Model A result for index {idx}")
                        if idx not in previous_results["model_b"].index:
                            raise Exception(f"Missing Model B result for index {idx}")
                        model_a_results.append(previous_results["model_a"].loc[idx].to_dict())
                        model_b_results.append(previous_results["model_b"].loc[idx].to_dict())
                    batch_results = self._call_model_c(abstracts, model_a_results, model_b_results)
                
                # 验证结果
                if not batch_results:
                    raise Exception("Empty results from model")
                
                # 添加索引到结果中
                for result in batch_results:
                    if "index" in result:
                        result["Index"] = int(result.pop("index"))
                    elif "Index" not in result:
                        raise Exception("Missing index in result")
                
                return batch_results
            except Exception as e:
                error_count += len(batch_df)
                logging.error(f"Error processing batch: {str(e)}")
                return []
        
        # 创建批次
        batches = []
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size].copy()
            batches.append(batch_df)
        
        # 使用线程池处理批次
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for batch_df in batches:
                future = executor.submit(process_batch_data, batch_df)
                futures.append((batch_df.index, future))
            
            # 处理结果
            for batch_indices, future in futures:
                try:
                    batch_results = future.result()
                    if batch_results:
                        results.extend(batch_results)
                except Exception as e:
                    error_count += len(batch_indices)
                    logging.error(f"Error processing batch: {str(e)}")
                finally:
                    completed_rows += len(batch_indices)
                    update_progress()
        
        if not results:
            raise Exception("No results were successfully processed")
        
        # 转换为DataFrame并验证结果
        results_df = pd.DataFrame(results)
        if "Index" not in results_df.columns:
            raise Exception("Missing Index column in results")
        results_df.set_index("Index", inplace=True)
        
        # 验证必要的列
        required_columns = {
            "model_a": ["P", "I", "C", "O", "S", "matches_criteria", "reasoning"],
            "model_b": ["inclusion_decision", "review_comments", 
                       "corrected_P", "corrected_I", "corrected_C", "corrected_O", "corrected_S"],
            "model_c": ["final_decision", "reasoning"]
        }
        
        missing_columns = set(required_columns[model_key]) - set(results_df.columns)
        if missing_columns:
            raise Exception(f"Missing required columns in results: {missing_columns}")
        
        return results_df
    
    def merge_results(self, df: pd.DataFrame, model_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """合并所有模型的结果"""
        merged = df.copy()
        merged.index.name = "Index"
        
        # 添加Model A的结果
        if "model_a" in model_results:
            for col in model_results["model_a"].columns:
                merged[f"A_{col}"] = model_results["model_a"][col]
        
        # 添加Model B的结果
        if "model_b" in model_results:
            for col in model_results["model_b"].columns:
                merged[f"B_{col}"] = model_results["model_b"][col]
        
        # 添加Model C的结果
        if "model_c" in model_results:
            for col in model_results["model_c"].columns:
                merged[f"C_{col}"] = model_results["model_c"][col]
        
        return merged
    
    def _call_api(self, config: Dict, prompt: str, model_key: str) -> Dict:
        """调用API"""
        try:
            # 验证API配置
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
                logging.debug(f"Sending request to {config['name']}: {json.dumps(data)}")
                response = requests.post(
                    config["api_url"],
                    headers=headers,
                    json=data,
                    timeout=config.get("timeout", 60)
                )
                logging.debug(f"Received raw response from {config['name']}: {response.text}")
            except requests.exceptions.Timeout:
                raise Exception(f"API call timed out for {config['name']}")
            except requests.exceptions.RequestException as e:
                raise Exception(f"API call failed for {config['name']}: {str(e)}")

            if response.status_code != 200:
                raise Exception(f"API call failed: {response.status_code} - {response.text}")
            
            try:
                response_json = response.json()
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse response as JSON: {response.text}")
                raise Exception(f"Invalid JSON response from API: {str(e)}")
            
            if 'choices' not in response_json or not response_json['choices']:
                raise Exception(f"Invalid API response format: missing choices - {response_json}")
            
            if 'message' not in response_json['choices'][0] or 'content' not in response_json['choices'][0]['message']:
                raise Exception(f"Invalid API response structure: {response_json}")
            
            content = response_json['choices'][0]['message']['content']
            logging.debug(f"Extracted content: {content}")
            
            # 清理和验证JSON字符串
            content = content.strip()
            if not content.startswith('{'):
                # 尝试在内容中找到JSON对象的开始
                json_start = content.find('{')
                if json_start != -1:
                    content = content[json_start:]
                else:
                    raise Exception(f"Response content is not a JSON object: {content}")
            
            # 尝试解析JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                # 如果JSON解析失败，尝试修复常见问题
                logging.warning(f"Initial JSON parse failed, attempting to fix content: {content}")
                
                # 1. 移除所有换行符和多余的空格
                content = re.sub(r'\s+', ' ', content).strip()
                
                # 2. 确保引号正确配对
                content = re.sub(r'(?<!\\)"', '\\"', content)
                content = content.replace('\\"', '"')  # 重置所有引号
                content = re.sub(r'([^\\])"([^"]*?)"', r'\1"\2"', content)  # 修复引号配对
                
                # 3. 修复常见的JSON语法错误
                content = content.replace("'", '"')  # 将单引号替换为双引号
                content = re.sub(r',\s*}', '}', content)  # 移除对象末尾的逗号
                content = re.sub(r',\s*]', ']', content)  # 移除数组末尾的逗号
                
                logging.debug(f"Cleaned content: {content}")
                
                try:
                    result = json.loads(content)
                    logging.info(f"Successfully fixed and parsed JSON content")
                except json.JSONDecodeError as e2:
                    raise Exception(f"Failed to parse API response as JSON after cleanup: {str(e2)}\nContent: {content}")
            
            # 验证必要的字段
            if model_key == "model_a":
                if not isinstance(result, dict) or 'analysis' not in result or not isinstance(result['analysis'], list):
                    raise Exception("Invalid Model A response format: missing 'analysis' array")
                if not result['analysis']:
                    raise Exception("Empty analysis array in Model A response")
                return result  # 返回整个结果，让调用者处理
            elif model_key == "model_b":
                if not isinstance(result, dict) or "reviews" not in result:
                    raise Exception("Invalid Model B response format: missing 'reviews' field")
                if not isinstance(result["reviews"], list):
                    raise Exception("Invalid Model B response format: 'reviews' is not a list")
                if not result["reviews"]:
                    raise Exception("Empty reviews array in Model B response")
                # 验证每个结果
                for item in result["reviews"]:
                    required_fields = ["index", "inclusion_decision", "review_comments", 
                                     "corrected_P", "corrected_I", "corrected_C", 
                                     "corrected_O", "corrected_S"]
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        raise Exception(f"Missing fields in Model B result: {missing_fields}")
                return result
            else:  # model_c
                if not isinstance(result, dict) or "decisions" not in result:
                    raise Exception("Invalid Model C response format: missing 'decisions' field")
                if not isinstance(result["decisions"], list):
                    raise Exception("Invalid Model C response format: 'decisions' is not a list")
                if not result["decisions"]:
                    raise Exception("Empty decisions array in Model C response")
                # 验证每个决定
                for item in result["decisions"]:
                    required_fields = ["index", "final_decision", "reasoning"]
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        raise Exception(f"Missing fields in Model C decision: {missing_fields}")
                return result
                
        except Exception as e:
            logging.error(f"Error in API call: {str(e)}")
            raise
    
    def _call_model_a(self, abstracts: List[Dict]) -> List[Dict]:
        """批量调用Model A进行分析"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                config = self.model_configs["model_a"]
                # 准备批量数据
                abstracts_json = json.dumps([
                    {"index": item["index"], "abstract": item["abstract"]} 
                    for item in abstracts
                ], indent=2)
                
                prompt = self.prompts["model_a"].format(
                    abstracts_json=abstracts_json,
                    **self.picos_criteria
                )
                
                logging.debug(f"Model A attempt {attempt + 1}/{max_retries}")
                response = self._call_api(config, prompt, "model_a")
                
                # 验证响应格式
                if not isinstance(response, dict) or 'analysis' not in response:
                    raise Exception("Invalid Model A response format: missing 'analysis' field")
                if not isinstance(response['analysis'], list):
                    raise Exception("Invalid Model A response format: 'analysis' is not a list")
                if not response['analysis']:
                    raise Exception("Empty analysis array in Model A response")
                
                # 验证每个分析结果
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
        """批量调用Model B进行审查"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                config = self.model_configs["model_b"]
                # 准备批量数据
                analyses = []
                for abstract, a_result in zip(abstracts, model_a_results):
                    analyses.append({
                        "index": str(abstract["index"]),
                        "abstract": abstract["abstract"],
                        "model_a_analysis": a_result
                    })
                
                abstracts_json = json.dumps(analyses, indent=2)
                prompt = self.prompts["model_b"].format(abstracts_json=abstracts_json)
                
                logging.debug(f"Model B attempt {attempt + 1}/{max_retries}")
                response = self._call_api(config, prompt, "model_b")
                
                # 验证响应格式
                if not isinstance(response, dict) or "reviews" not in response:
                    raise Exception("Invalid Model B response format: missing 'reviews' field")
                if not isinstance(response["reviews"], list):
                    raise Exception("Invalid Model B response format: 'reviews' is not a list")
                if not response["reviews"]:
                    raise Exception("Empty reviews array in Model B response")
                
                # 验证每个结果
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
        """批量调用Model C进行仲裁"""
        max_retries = 3
        retry_delay = 2
        
        # 首先筛选出需要仲裁的案例
        disagreements = []
        for abstract, a_result, b_result in zip(abstracts, model_a_results, model_b_results):
            # 只有当A和B的决定不一致时才需要仲裁
            if a_result["matches_criteria"] != b_result["inclusion_decision"]:
                disagreements.append({
                    "index": str(abstract["index"]),
                    "abstract": abstract["abstract"],
                    "model_a_analysis": {
                        "decision": a_result["matches_criteria"],
                        "reasoning": a_result["reasoning"],
                        "P": a_result["P"],
                        "I": a_result["I"],
                        "C": a_result["C"],
                        "O": a_result["O"],
                        "S": a_result["S"]
                    },
                    "model_b_analysis": {
                        "decision": b_result["inclusion_decision"],
                        "reasoning": b_result["review_comments"],
                        "P": b_result["corrected_P"],
                        "I": b_result["corrected_I"],
                        "C": b_result["corrected_C"],
                        "O": b_result["corrected_O"],
                        "S": b_result["corrected_S"]
                    }
                })
        
        # 如果没有分歧，返回空列表
        if not disagreements:
            logging.info("No disagreements found between Model A and B")
            return []
        
        for attempt in range(max_retries):
            try:
                config = self.model_configs["model_c"]
                disagreements_json = json.dumps(disagreements, indent=2)
                prompt = self.prompts["model_c"].format(
                    disagreements_json=disagreements_json,
                    **self.picos_criteria
                )
                
                logging.debug(f"Model C attempt {attempt + 1}/{max_retries}")
                response = self._call_api(config, prompt, "model_c")
                
                # 验证响应格式
                if not isinstance(response, dict) or "decisions" not in response:
                    raise Exception("Invalid Model C response format")
                
                # 验证每个决定
                for item in response["decisions"]:
                    required_fields = ["index", "final_decision", "reasoning"]
                    missing_fields = [field for field in required_fields if field not in item]
                    if missing_fields:
                        raise Exception(f"Missing fields in Model C decision: {missing_fields}")
                
                return response["decisions"]
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Model C attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"All Model C attempts failed: {str(e)}")
                    raise

def update_picos_criteria(p, i, c, o, s):
    """更新PICOS标准"""
    try:
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
    analyzer = PICOSAnalyzer()
    model_results = {}
    
    def parse_nbib(file) -> tuple:
        """解析NBIB文件并返回结果"""
        try:
            records = []
            record = {}
            authors = []
            current_field = None

            with open(file.name, 'r', encoding='utf-8') as f:
                lines = f.readlines()

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

            # 创建DataFrame并添加索引
            df = pd.DataFrame(records)
            df.index.name = 'Index'
            
            # 保存到CSV
            output_path = os.path.join(DATA_DIR, "extracted_data.csv")
            df.to_csv(output_path)

            # 准备预览数据
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
        """更新模型设置"""
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
        """测试API连接"""
        return analyzer.test_api_connection(model_key)
    
    def process_model(input_file, model_key):
        """处理单个模型的分析"""
        try:
            # 读取CSV文件并确保索引正确
            df = pd.read_csv(input_file.name, index_col='Index')
            if df.index.name != 'Index':
                df.index.name = 'Index'
            
            if model_key in ["model_b", "model_c"] and not all(k in model_results for k in ["model_a", "model_b"][:{"model_b": 1, "model_c": 2}[model_key]]):
                return None, f"Previous model results required for {model_key.upper()}"
            
            # 开始处理
            logging.info(f"Starting {model_key.upper()} analysis...")
            results_df = analyzer.process_batch(df, model_key, model_results)
            model_results[model_key] = results_df
            
            # 保存结果时确保包含索引
            output_path = os.path.join(DATA_DIR, f"{model_key}_results.csv")
            results_df.to_csv(output_path)
            
            completion_msg = f"{model_key.upper()} analysis completed: processed {len(df)} rows"
            logging.info(completion_msg)
            return output_path, completion_msg
        except Exception as e:
            error_msg = f"Error in {model_key.upper()}: {str(e)}"
            logging.error(error_msg)
            return None, error_msg
    
    def merge_all_results(input_file):
        """合并所有结果"""
        try:
            if not all(k in model_results for k in ["model_a", "model_b"]):
                return None, "Model A and B results required"
            
            # 读取CSV文件并确保索引正确
            df = pd.read_csv(input_file.name, index_col='Index')
            if df.index.name != 'Index':
                df.index.name = 'Index'
            
            merged_df = analyzer.merge_results(df, model_results)
            
            # 保存结果时确保包含索引
            output_path = os.path.join(DATA_DIR, "final_results.csv")
            merged_df.to_csv(output_path)
            return output_path, "Results merged successfully"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def merge_results_with_files(input_file, model_a_file, model_b_file, model_c_file):
        """合并所有结果"""
        if not all([input_file, model_a_file, model_b_file]):
            return None, "Original file, Model A and B results are required"
        
        try:
            # 加载所有结果
            model_a_results = pd.read_csv(model_a_file.name)
            model_b_results = pd.read_csv(model_b_file.name)
            model_c_results = pd.read_csv(model_c_file.name) if model_c_file else None
            
            # 确保索引列存在
            if "Index" not in model_a_results.columns:
                return None, "Model A results missing Index column"
            if "Index" not in model_b_results.columns:
                return None, "Model B results missing Index column"
            if model_c_results is not None and "Index" not in model_c_results.columns:
                return None, "Model C results missing Index column"
            
            # 设置索引
            model_a_results.set_index("Index", inplace=True)
            model_b_results.set_index("Index", inplace=True)
            if model_c_results is not None:
                model_c_results.set_index("Index", inplace=True)
            
            # 处理原始文件
            df = pd.read_csv(input_file.name)
            if "Index" not in df.columns:
                df["Index"] = df.index.astype(str)
            
            # 验证所有必需的索引都存在
            df_indices = set(df["Index"].astype(str))
            missing_a = df_indices - set(model_a_results.index.astype(str))
            missing_b = df_indices - set(model_b_results.index.astype(str))
            
            if missing_a:
                return None, f"Missing Model A results for indices: {', '.join(sorted(missing_a))}"
            if missing_b:
                return None, f"Missing Model B results for indices: {', '.join(sorted(missing_b))}"
            
            # 合并结果
            merged_df = df.copy()
            merged_df.set_index("Index", inplace=True)
            
            # 添加Model A的结果（Decision和Reasoning在前）
            merged_df['A_Decision'] = model_a_results['matches_criteria']
            merged_df['A_Reasoning'] = model_a_results['reasoning']
            merged_df['A_P'] = model_a_results['P']
            merged_df['A_I'] = model_a_results['I']
            merged_df['A_C'] = model_a_results['C']
            merged_df['A_O'] = model_a_results['O']
            merged_df['A_S'] = model_a_results['S']
            
            # 添加Model B的结果（Decision和Reasoning在前）
            merged_df['B_Decision'] = model_b_results['inclusion_decision']
            merged_df['B_Reasoning'] = model_b_results['review_comments']
            merged_df['B_P'] = model_b_results['corrected_P']
            merged_df['B_I'] = model_b_results['corrected_I']
            merged_df['B_C'] = model_b_results['corrected_C']
            merged_df['B_O'] = model_b_results['corrected_O']
            merged_df['B_S'] = model_b_results['corrected_S']
            
            # 添加Model C的结果（只对有分歧的案例）
            if model_c_results is not None:
                # 初始化C的列
                merged_df['C_Decision'] = None
                merged_df['C_Reasoning'] = None
                
                # 只对有分歧的案例填充C的结果
                disagreement_mask = merged_df['A_Decision'] != merged_df['B_Decision']
                for idx in merged_df[disagreement_mask].index:
                    if idx in model_c_results.index:
                        merged_df.loc[idx, 'C_Decision'] = model_c_results.loc[idx, 'final_decision']
                        merged_df.loc[idx, 'C_Reasoning'] = model_c_results.loc[idx, 'reasoning']
            
            # 计算最终决策
            def get_final_decision(row):
                # 如果A和B一致，直接使用他们的决定
                if row['A_Decision'] == row['B_Decision']:
                    return row['A_Decision']
                # 如果有分歧且有C的决定，使用C的决定
                elif pd.notna(row.get('C_Decision')):
                    return row['C_Decision']
                # 如果有分歧但没有C的决定，返回None
                return None
            
            # 应用最终决策规则
            merged_df['Final_Decision'] = merged_df.apply(get_final_decision, axis=1)
            
            # 保存结果
            output_path = os.path.join(DATA_DIR, "final_results.csv")
            merged_df.to_csv(output_path)
            return output_path, "Results merged successfully"
        except Exception as e:
            return None, f"Error merging results: {str(e)}"
    
    def run_all_models(input_file):
        """运行所有模型的分析流程"""
        try:
            # 读取CSV文件并确保索引正确
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
            
            # 初始化输出变量
            model_a_path = None
            model_b_path = None
            model_c_path = None
            final_path = None
            
            # 运行Model A
            logging.info("Starting Model A analysis...")
            model_a_path, model_a_status = process_model(input_file, "model_a")
            if not model_a_path:
                yield model_a_path, model_b_path, model_c_path, final_path, f"Model A failed: {model_a_status}"
                return
            status = update_progress("Model A", "Completed")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            # 运行Model B
            logging.info("Starting Model B analysis...")
            model_b_path, model_b_status = process_model(input_file, "model_b")
            if not model_b_path:
                yield model_a_path, model_b_path, model_c_path, final_path, f"Model B failed: {model_b_status}"
                return
            status = update_progress("Model B", "Completed")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            # 运行Model C
            logging.info("Starting Model C analysis...")
            model_c_path, model_c_status = process_model(input_file, "model_c")
            if not model_c_path:
                yield model_a_path, model_b_path, model_c_path, final_path, f"Model C failed: {model_c_status}"
                return
            status = update_progress("Model C", "Completed")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            # 创建临时文件对象
            class TempFile:
                def __init__(self, path):
                    self.name = path
            
            # 合并结果
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
    
    # 创建Gradio界面
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
                """加载模型结果"""
                if not file_path:
                    return None
                try:
                    return pd.read_csv(file_path.name)
                except Exception as e:
                    logging.error(f"Error loading results: {str(e)}")
                    return None
            
            def process_model_a(input_file):
                """处理Model A"""
                if not input_file:
                    return None, "Please upload the original CSV file"
                return process_model(input_file, "model_a")
            
            def process_model_b(input_file, model_a_file):
                """处理Model B"""
                if not input_file:
                    return None, "Please upload the original CSV file"
                if not model_a_file:
                    return None, "Please upload Model A results"
                
                try:
                    # 加载Model A的结果
                    model_a_results = pd.read_csv(model_a_file.name)
                    # 确保索引列存在
                    if "Index" not in model_a_results.columns:
                        return None, "Model A results missing Index column"
                    
                    # 设置索引
                    model_a_results.set_index("Index", inplace=True)
                    model_results["model_a"] = model_a_results
                    
                    # 处理原始文件
                    df = pd.read_csv(input_file.name)
                    # 添加索引列
                    if "Index" not in df.columns:
                        df["Index"] = df.index.astype(str)
                    
                    # 验证所有必需的索引都存在
                    missing_indices = set(df["Index"].astype(str)) - set(model_a_results.index.astype(str))
                    if missing_indices:
                        return None, f"Missing Model A results for indices: {', '.join(sorted(missing_indices))}"
                    
                    return process_model(input_file, "model_b")
                except Exception as e:
                    return None, f"Error processing Model B: {str(e)}"
            
            def process_model_c(input_file, model_a_file, model_b_file):
                """处理Model C"""
                if not input_file:
                    return None, "Please upload the original CSV file"
                if not model_a_file:
                    return None, "Please upload Model A results"
                if not model_b_file:
                    return None, "Please upload Model B results"
                
                try:
                    # 加载Model A和B的结果
                    model_a_results = pd.read_csv(model_a_file.name)
                    model_b_results = pd.read_csv(model_b_file.name)
                    
                    # 确保索引列存在
                    if "Index" not in model_a_results.columns:
                        return None, "Model A results missing Index column"
                    if "Index" not in model_b_results.columns:
                        return None, "Model B results missing Index column"
                    
                    # 设置索引
                    model_a_results.set_index("Index", inplace=True)
                    model_b_results.set_index("Index", inplace=True)
                    
                    # 处理原始文件
                    df = pd.read_csv(input_file.name)
                    # 添加索引列
                    if "Index" not in df.columns:
                        df["Index"] = df.index.astype(str)
                    
                    # 验证所有必需的索引都存在
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
                """合并所有结果"""
                if not all([input_file, model_a_file, model_b_file]):
                    return None, "Original file, Model A and B results are required"
                
                try:
                    # 加载所有结果
                    model_a_results = pd.read_csv(model_a_file.name)
                    model_b_results = pd.read_csv(model_b_file.name)
                    model_c_results = pd.read_csv(model_c_file.name) if model_c_file else None
                    
                    # 确保索引列存在
                    if "Index" not in model_a_results.columns:
                        return None, "Model A results missing Index column"
                    if "Index" not in model_b_results.columns:
                        return None, "Model B results missing Index column"
                    if model_c_results is not None and "Index" not in model_c_results.columns:
                        return None, "Model C results missing Index column"
                    
                    # 设置索引
                    model_a_results.set_index("Index", inplace=True)
                    model_b_results.set_index("Index", inplace=True)
                    if model_c_results is not None:
                        model_c_results.set_index("Index", inplace=True)
                    
                    # 处理原始文件
                    df = pd.read_csv(input_file.name)
                    if "Index" not in df.columns:
                        df["Index"] = df.index.astype(str)
                    
                    # 验证所有必需的索引都存在
                    df_indices = set(df["Index"].astype(str))
                    missing_a = df_indices - set(model_a_results.index.astype(str))
                    missing_b = df_indices - set(model_b_results.index.astype(str))
                    
                    if missing_a:
                        return None, f"Missing Model A results for indices: {', '.join(sorted(missing_a))}"
                    if missing_b:
                        return None, f"Missing Model B results for indices: {', '.join(sorted(missing_b))}"
                    
                    # 合并结果
                    merged_df = df.copy()
                    merged_df.set_index("Index", inplace=True)
                    
                    # 添加Model A的结果（Decision和Reasoning在前）
                    merged_df['A_Decision'] = model_a_results['matches_criteria']
                    merged_df['A_Reasoning'] = model_a_results['reasoning']
                    merged_df['A_P'] = model_a_results['P']
                    merged_df['A_I'] = model_a_results['I']
                    merged_df['A_C'] = model_a_results['C']
                    merged_df['A_O'] = model_a_results['O']
                    merged_df['A_S'] = model_a_results['S']
                    
                    # 添加Model B的结果（Decision和Reasoning在前）
                    merged_df['B_Decision'] = model_b_results['inclusion_decision']
                    merged_df['B_Reasoning'] = model_b_results['review_comments']
                    merged_df['B_P'] = model_b_results['corrected_P']
                    merged_df['B_I'] = model_b_results['corrected_I']
                    merged_df['B_C'] = model_b_results['corrected_C']
                    merged_df['B_O'] = model_b_results['corrected_O']
                    merged_df['B_S'] = model_b_results['corrected_S']
                    
                    # 添加Model C的结果（只对有分歧的案例）
                    if model_c_results is not None:
                        # 初始化C的列
                        merged_df['C_Decision'] = None
                        merged_df['C_Reasoning'] = None
                        
                        # 只对有分歧的案例填充C的结果
                        disagreement_mask = merged_df['A_Decision'] != merged_df['B_Decision']
                        for idx in merged_df[disagreement_mask].index:
                            if idx in model_c_results.index:
                                merged_df.loc[idx, 'C_Decision'] = model_c_results.loc[idx, 'final_decision']
                                merged_df.loc[idx, 'C_Reasoning'] = model_c_results.loc[idx, 'reasoning']
                    
                    # 计算最终决策
                    def get_final_decision(row):
                        # 如果A和B一致，直接使用他们的决定
                        if row['A_Decision'] == row['B_Decision']:
                            return row['A_Decision']
                        # 如果有分歧且有C的决定，使用C的决定
                        elif pd.notna(row.get('C_Decision')):
                            return row['C_Decision']
                        # 如果有分歧但没有C的决定，返回None
                        return None
                    
                    # 应用最终决策规则
                    merged_df['Final_Decision'] = merged_df.apply(get_final_decision, axis=1)
                    
                    # 保存结果
                    output_path = os.path.join(DATA_DIR, "final_results.csv")
                    merged_df.to_csv(output_path)
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