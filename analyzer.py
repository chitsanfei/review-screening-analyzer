import pandas as pd
import logging
import json
import concurrent.futures
from typing import Dict, List, Optional
from model_manager import ModelManager
from prompt_manager import PromptManager
from result_processor import ResultProcessor
import re

class PICOSAnalyzer:
    def __init__(self):
        self.model_manager = ModelManager()
        self.prompt_manager = PromptManager()
        self.result_processor = ResultProcessor()
        self.picos_criteria = {
            "population": "patients with hepatocellular carcinoma",
            "intervention": "immune checkpoint inhibitors (ICIs)",
            "comparison": "treatment without the studied ICIs or placebo",
            "outcome": "survival rate or response rate",
            "study_design": "randomized controlled trial"
        }
    
    def update_picos_criteria(self, criteria: Dict[str, str]) -> None:
        """Update PICOS criteria"""
        self.picos_criteria.update(criteria)
    
    def update_model_config(self, model_key: str, config: Dict) -> None:
        """Update model configuration"""
        self.model_manager.update_model_config(model_key, config)
    
    def update_prompt(self, model_key: str, prompt: str) -> None:
        """Update model prompt"""
        self.prompt_manager.update_prompt(model_key, prompt)
    
    def test_api_connection(self, model_key: str) -> str:
        """Test API connection"""
        return self.model_manager.test_api_connection(model_key)
    
    def process_batch(self, df: pd.DataFrame, model_key: str, previous_results: Dict = None) -> pd.DataFrame:
        """Process a batch of data"""
        config = self.model_manager.get_config(model_key)
        batch_size = config["batch_size"]
        threads = config["threads"]
        results = []
        total_rows = len(df)
        completed_rows = 0
        error_count = 0
        failed_indices = set()
        
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
            empty_results = []
            
            for idx, row in batch_df.iterrows():
                try:
                    is_empty = pd.isna(row["Abstract"]) or len(str(row["Abstract"]).strip()) < 50
                    if is_empty:
                        logging.warning(f"Row {idx}: Abstract too short or empty")
                        empty_result = self._create_empty_result(idx, model_key)
                        empty_results.append(empty_result)
                        continue
                    
                    abstract = {
                        "Index": str(idx),
                        "abstract": str(row["Abstract"]).strip()
                    }
                    
                    if model_key in ["model_b", "model_c"] and previous_results:
                        if not self._validate_previous_results(idx, model_key, previous_results):
                            failed_indices.add(idx)
                            continue
                        
                        if model_key == "model_c":
                            if not self._check_disagreement(idx, previous_results):
                                empty_results.append(self._create_no_disagreement_result(idx, previous_results))
                                continue
                    
                    batch_results.append(abstract)
                    
                except Exception as e:
                    logging.error(f"Error processing row {idx}: {str(e)}")
                    error_count += 1
                    failed_indices.add(idx)
            
            try:
                api_results = []
                if batch_results:
                    api_results = self._call_model(model_key, batch_results, previous_results)
                
                all_results = []
                for result in api_results + empty_results:
                    if "Index" not in result:
                        logging.warning(f"Missing Index in result: {result}")
                        continue
                    all_results.append(result)
                return all_results
                
            except Exception as e:
                logging.error(f"Error in model call: {str(e)}")
                error_count += len(batch_results)
                for abstract in batch_results:
                    failed_indices.add(abstract["Index"])
                return empty_results
        
        # Create batches
        batches = []
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size].copy()
            batches.append(batch_df)
        
        # Process batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for batch_df in batches:
                future = executor.submit(process_batch_data, batch_df)
                futures.append((batch_df.index, future))
            
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
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        if "Index" in results_df.columns:
            results_df.set_index("Index", inplace=True)
        else:
            raise Exception("Missing Index column in results")
        
        # Validate results
        missing_columns = set(self.result_processor.required_columns[model_key]) - set(results_df.columns)
        if missing_columns:
            raise Exception(f"Missing required columns in results: {missing_columns}")
        
        if failed_indices:
            failed_list = sorted(list(failed_indices))
            logging.warning(f"Failed to process {len(failed_indices)} indices in {model_key}: {failed_list}")
        
        return results_df
    
    def merge_results(self, df: pd.DataFrame, model_results: Dict) -> pd.DataFrame:
        """Merge all model results"""
        return self.result_processor.merge_results(df, model_results)
    
    def _create_empty_result(self, idx: str, model_key: str) -> Dict:
        """Create empty result for invalid abstract"""
        result = {"Index": str(idx)}
        if model_key == "model_a":
            result.update({
                "A_P": "not applicable",
                "A_I": "not applicable",
                "A_C": "not applicable",
                "A_O": "not applicable",
                "A_S": "not applicable",
                "A_Decision": False,
                "A_Reason": "Abstract too short or empty"
            })
        elif model_key == "model_b":
            result.update({
                "B_P": "-",
                "B_I": "-",
                "B_C": "-",
                "B_O": "-",
                "B_S": "-",
                "B_Decision": False,
                "B_Reason": "Abstract too short or empty"
            })
        else:  # model_c
            result.update({
                "C_Decision": False,
                "C_Reason": "Abstract too short or empty"
            })
        return result
    
    def _create_no_disagreement_result(self, idx: str, previous_results: Dict) -> Dict:
        """Create result for no disagreement case"""
        str_idx = str(idx)
        a_result = previous_results["model_a"].loc[str_idx]
        return {
            "Index": str_idx,
            "C_Decision": a_result["A_Decision"],
            "C_Reason": "No disagreement between Model A and B"
        }
    
    def _validate_previous_results(self, idx: str, model_key: str, previous_results: Dict) -> bool:
        """Validate previous model results exist"""
        str_idx = str(idx)
        if "model_a" not in previous_results:
            raise Exception("Model A results required")
        
        model_a_data = previous_results["model_a"]
        if str_idx not in model_a_data.index.astype(str).values:
            logging.warning(f"Missing Model A result for index {idx}")
            return False
        
        if model_key == "model_c":
            if "model_b" not in previous_results:
                raise Exception("Model B results required")
            
            model_b_data = previous_results["model_b"]
            if str_idx not in model_b_data.index.astype(str).values:
                logging.warning(f"Missing Model B result for index {idx}")
                return False
        
        return True
    
    def _check_disagreement(self, idx: str, previous_results: Dict) -> bool:
        """Check if there is disagreement between Model A and B"""
        str_idx = str(idx)
        a_result = previous_results["model_a"].loc[str_idx]
        b_result = previous_results["model_b"].loc[str_idx]
        return a_result["A_Decision"] != b_result["B_Decision"]
    
    def _call_model(self, model_key: str, batch_results: List[Dict], previous_results: Dict = None) -> List[Dict]:
        """Call model API with appropriate data preparation"""
        if model_key == "model_a":
            return self._call_model_a(batch_results)
        elif model_key == "model_b":
            return self._call_model_b(batch_results, previous_results)
        else:  # model_c
            return self._call_model_c(batch_results, previous_results)
    
    def _call_model_a(self, abstracts: List[Dict]) -> List[Dict]:
        """Call Model A"""
        abstracts_json = json.dumps([
            {"index": str(item["Index"]), "abstract": item["abstract"]}
            for item in abstracts
        ], indent=2)
        
        prompt = self.prompt_manager.get_prompt("model_a").format(
            abstracts_json=abstracts_json,
            **self.picos_criteria
        )
        
        try:
            response = self.model_manager.call_api("model_a", prompt)
            self.result_processor.validate_model_response(response, "model_a")
            return response["analysis"]
        except Exception as e:
            logging.error(f"Error in Model A processing: {str(e)}")
            raise
    
    def _call_model_b(self, abstracts: List[Dict], previous_results: Dict) -> List[Dict]:
        """Call Model B"""
        if "model_a" not in previous_results:
            raise Exception("Model A results required")
            
        model_a_df = previous_results["model_a"]
        batch_data = []
        
        for abstract in abstracts:
            try:
                idx = str(abstract["Index"])
                
                if idx not in model_a_df.index:
                    logging.error(f"Index {idx} not found in Model A results")
                    continue
                
                a_result = model_a_df.loc[idx]
                if not isinstance(a_result, pd.Series):
                    logging.error(f"Unexpected result type for index {idx}")
                    continue
                
                a_result_dict = a_result.to_dict()
                
                # Check required fields
                required_fields = ["A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S"]
                missing_fields = [field for field in required_fields if field not in a_result_dict]
                if missing_fields:
                    logging.error(f"Missing required fields in Model A result: {missing_fields}")
                    continue
                
                # Prepare data
                batch_item = {
                    "Index": idx,
                    "abstract": abstract["abstract"],
                    "model_a_analysis": {
                        "A_Decision": a_result_dict["A_Decision"],
                        "A_Reason": a_result_dict["A_Reason"],
                        "A_P": a_result_dict["A_P"],
                        "A_I": a_result_dict["A_I"],
                        "A_C": a_result_dict["A_C"],
                        "A_O": a_result_dict["A_O"],
                        "A_S": a_result_dict["A_S"]
                    }
                }
                batch_data.append(batch_item)
            except Exception as e:
                logging.error(f"Error preparing batch data for index {idx}: {str(e)}")
                continue
        
        if not batch_data:
            logging.error("No valid batch data prepared for Model B")
            return []
        
        # Build prompt
        prompt = self.prompt_manager.get_prompt("model_b").format(
            abstracts_json=json.dumps(batch_data, indent=2),
            **self.picos_criteria
        )
        
        # Call API
        try:
            response = self.model_manager.call_api("model_b", prompt)
            
            # 检查响应格式
            if not isinstance(response, dict):
                logging.error(f"Invalid response type from Model B: {type(response)}")
                raise Exception("Invalid Model B response format: response is not a dictionary")
            
            if "choices" not in response:
                logging.error(f"Missing 'choices' in Model B response: {response}")
                raise Exception("Invalid Model B response format: missing 'choices' field")
            
            if not isinstance(response["choices"], list) or not response["choices"]:
                logging.error(f"Invalid 'choices' in Model B response: {response['choices']}")
                raise Exception("Invalid Model B response format: empty or invalid choices")
            
            content = response["choices"][0].get("message", {}).get("content", "")
            if not content:
                logging.error("Empty content in Model B response")
                raise Exception("Invalid Model B response format: empty content")
            
            # 检查是否包含 markdown 代码块
            if "```json" in content:
                pattern = r"```json\s*(.*?)\s*```"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    content = match.group(1)
            
            try:
                result = json.loads(content)
                self.result_processor.validate_model_response(result, "model_b")
                return result["reviews"]
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Model B response content: {content}")
                raise Exception(f"Failed to parse Model B response: {str(e)}")
            
        except Exception as e:
            logging.error(f"Error calling Model B API: {str(e)}")
            raise
    
    def _call_model_c(self, abstracts: List[Dict], previous_results: Dict) -> List[Dict]:
        """Call Model C"""
        batch_data = []
        for abstract in abstracts:
            idx = str(abstract["Index"])
            a_result = previous_results["model_a"].loc[idx]
            b_result = previous_results["model_b"].loc[idx]
            
            # 确保布尔值是 Python 原生类型
            a_decision = bool(a_result["A_Decision"])
            b_decision = bool(b_result["B_Decision"])
            
            # 使用与 prompt 中一致的字段名
            batch_data.append({
                "Index": idx,
                "Abstract": abstract["abstract"],
                "model_a_analysis": {
                    "A_Decision": a_decision,
                    "A_Reason": str(a_result["A_Reason"]),
                    "A_P": str(a_result["A_P"]),
                    "A_I": str(a_result["A_I"]),
                    "A_C": str(a_result["A_C"]),
                    "A_O": str(a_result["A_O"]),
                    "A_S": str(a_result["A_S"])
                },
                "model_b_analysis": {
                    "B_Decision": b_decision,
                    "B_Reason": str(b_result["B_Reason"]),
                    "B_P": str(b_result["B_P"]),
                    "B_I": str(b_result["B_I"]),
                    "B_C": str(b_result["B_C"]),
                    "B_O": str(b_result["B_O"]),
                    "B_S": str(b_result["B_S"])
                }
            })
        
        prompt = self.prompt_manager.get_prompt("model_c").format(
            disagreements_json=json.dumps(batch_data, indent=2),
            **self.picos_criteria
        )
        
        response = self.model_manager.call_api("model_c", prompt)
        
        # 检查响应格式
        if not isinstance(response, dict):
            logging.error(f"Invalid response type from Model C: {type(response)}")
            raise Exception("Invalid Model C response format: response is not a dictionary")
        
        if "choices" not in response:
            logging.error(f"Missing 'choices' in Model C response: {response}")
            raise Exception("Invalid Model C response format: missing 'choices' field")
        
        if not isinstance(response["choices"], list) or not response["choices"]:
            logging.error(f"Invalid 'choices' in Model C response: {response['choices']}")
            raise Exception("Invalid Model C response format: empty or invalid choices")
        
        content = response["choices"][0].get("message", {}).get("content", "")
        if not content:
            logging.error("Empty content in Model C response")
            raise Exception("Invalid Model C response format: empty content")
        
        # 检查是否包含 markdown 代码块
        if "```json" in content:
            pattern = r"```json\s*(.*?)\s*```"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = match.group(1)
        
        logging.debug(f"Model C response content: {content}")
        
        try:
            result = json.loads(content)
            self.result_processor.validate_model_response(result, "model_c")
            return result["decisions"]
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Model C response content: {content}")
            raise Exception(f"Failed to parse Model C response: {str(e)}")
        except Exception as e:
            logging.error(f"Error validating Model C response: {str(e)}")
            raise 