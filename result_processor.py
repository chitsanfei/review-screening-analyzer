import pandas as pd
import logging
from typing import Dict, List, Optional
import json
import re

class ResultProcessor:
    def __init__(self):
        self.required_columns = {
            "model_a": ["A_P", "A_I", "A_C", "A_O", "A_S", "A_Decision", "A_Reason"],
            "model_b": ["B_P", "B_I", "B_C", "B_O", "B_S", "B_Decision", "B_Reason"],
            "model_c": ["C_Decision", "C_Reason"]
        }
    
    def validate_model_response(self, result: Dict, model_key: str) -> None:
        """Validate model-specific response format"""
        if model_key == "model_a":
            # Check if response is in the completion format
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
                if content:
                    try:
                        # Remove markdown code block if present
                        json_content = content
                        if "```json" in content:
                            # Extract JSON content from markdown code block
                            pattern = r"```json\s*(.*?)\s*```"
                            match = re.search(pattern, content, re.DOTALL)
                            if match:
                                json_content = match.group(1)
                        
                        # Try to parse the content as JSON
                        parsed = json.loads(json_content)
                        if isinstance(parsed, dict) and "analysis" in parsed:
                            result.clear()
                            result.update(parsed)
                    except json.JSONDecodeError as e:
                        raise Exception(f"Invalid JSON in Model A response content: {content}. Error: {str(e)}")
            
            if not isinstance(result, dict) or 'analysis' not in result or not isinstance(result['analysis'], list):
                raise Exception("Invalid Model A response format: missing 'analysis' array")
            if not result['analysis']:
                raise Exception("Empty analysis array in Model A response")
            # Validate each analysis result
            for item in result['analysis']:
                if not isinstance(item, dict):
                    raise Exception(f"Invalid analysis item format: {item}")
                if 'Index' not in item:
                    raise Exception(f"Missing 'Index' in analysis item: {item}")
                missing_fields = [field for field in self.required_columns[model_key] if field not in item]
                if missing_fields:
                    raise Exception(f"Missing fields in analysis item: {missing_fields}")
                
        elif model_key == "model_b":
            # Check if response is in the completion format
            logging.debug(f"Validating Model B response: {result}")
            
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
                logging.debug(f"Model B raw content: {content}")
                
                if content:
                    try:
                        # Remove markdown code block if present
                        json_content = content
                        if "```json" in content:
                            # Extract JSON content from markdown code block
                            pattern = r"```json\s*(.*?)\s*```"
                            match = re.search(pattern, content, re.DOTALL)
                            if match:
                                json_content = match.group(1)
                                logging.debug(f"Extracted JSON content: {json_content}")
                        
                        # Try to parse the content as JSON
                        parsed = json.loads(json_content)
                        logging.debug(f"Parsed JSON: {parsed}")
                        
                        if isinstance(parsed, dict) and "reviews" in parsed:
                            result.clear()
                            result.update(parsed)
                            logging.debug(f"Updated result with parsed content: {result}")
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON parsing error: {str(e)}")
                        logging.error(f"Failed content: {json_content}")
                        raise Exception(f"Invalid JSON in Model B response content: {content}. Error: {str(e)}")

            if not isinstance(result, dict):
                logging.error(f"Result is not a dict: {type(result)}")
                raise Exception("Invalid Model B response format: result is not a dictionary")
                
            if "reviews" not in result:
                logging.error(f"Missing 'reviews' field in result: {result.keys()}")
                raise Exception("Invalid Model B response format: missing 'reviews' field")
                
            if not isinstance(result["reviews"], list):
                logging.error(f"'reviews' is not a list: {type(result['reviews'])}")
                raise Exception("Invalid Model B response format: 'reviews' is not a list")
                
            if not result["reviews"]:
                logging.error("Empty reviews array")
                raise Exception("Empty reviews array in Model B response")
            
            # Validate each review
            for i, item in enumerate(result["reviews"]):
                logging.debug(f"Validating review item {i}: {item}")
                
                if not isinstance(item, dict):
                    logging.error(f"Review item {i} is not a dict: {type(item)}")
                    raise Exception(f"Invalid review item format: {item}")
                    
                if "Index" not in item:
                    logging.error(f"Missing 'Index' in review item {i}: {item.keys()}")
                    raise Exception(f"Missing 'Index' in review item: {item}")
                
                # 检查所需字段（根据新的 Prompt 格式）
                required_fields = self.required_columns["model_b"]
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    logging.error(f"Review item {i} missing fields: {missing_fields}")
                    logging.error(f"Available fields: {item.keys()}")
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
                if "Index" not in item:
                    raise Exception(f"Missing 'Index' in decision item: {item}")
                missing_fields = [field for field in self.required_columns[model_key] if field not in item]
                if missing_fields:
                    raise Exception(f"Missing fields in Model C decision: {missing_fields}")
    
    def merge_results(self, df: pd.DataFrame, model_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all model results"""
        logging.info("Merging results...")
        
        if not all(k in model_results for k in ["model_a", "model_b"]):
            raise Exception("Model A and B results required")
            
        # Create a copy of input DataFrame
        merged_df = df.copy()
        
        # Add Model A results
        model_a_df = model_results["model_a"]
        for col in self.required_columns["model_a"]:
            merged_df[col] = model_a_df[col]
        
        # Add Model B results
        model_b_df = model_results["model_b"]
        for col in self.required_columns["model_b"]:
            merged_df[col] = model_b_df[col]
        
        # Add Model C results if available
        if "model_c" in model_results:
            model_c_df = model_results["model_c"]
            for col in self.required_columns["model_c"]:
                merged_df[col] = model_c_df[col]
        else:
            # If no Model C results, use logic to determine final decision
            merged_df["C_Decision"] = None
            merged_df["C_Reason"] = merged_df.apply(
                lambda row: "No disagreement between Model A and B" if row["A_Decision"] == row["B_Decision"]
                else "Disagreement not resolved",
                axis=1
            )
        
        # Determine final decision
        merged_df["Final_Decision"] = merged_df.apply(
            lambda row: row["C_Decision"] if pd.notnull(row["C_Decision"])
            else row["A_Decision"] if row["A_Decision"] == row["B_Decision"]
            else row["B_Decision"],
            axis=1
        )
        
        # Ensure all required columns are present and in correct order
        required_columns = [
            "Title", "DOI", "Abstract", "Authors",
            *self.required_columns["model_a"],
            *self.required_columns["model_b"],
            *self.required_columns["model_c"],
            "Final_Decision"
        ]
        
        # 确保索引名称正确
        merged_df.index.name = "Index"
        
        # 移除可能存在的 Index 列（如果它作为普通列存在）
        if "Index" in merged_df.columns:
            merged_df = merged_df.drop(columns=["Index"])
        
        # 只保留需要的列，并按指定顺序排列
        merged_df = merged_df[required_columns]
        
        return merged_df 