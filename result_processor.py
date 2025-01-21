import pandas as pd
import logging
from typing import Dict, List, Optional
import json
import re

class ResultProcessor:
    def __init__(self):
        """Initialize ResultProcessor"""
        self.required_columns = {
            "model_a": ["A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S"],
            "model_b": ["B_Decision", "B_Reason", "B_P", "B_I", "B_C", "B_O", "B_S"],
            "model_c": ["C_Decision", "C_Reason"]
        }
        
        self.output_columns = [
            "Index", 
            "A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S",
            "B_Decision", "B_Reason", "B_P", "B_I", "B_C", "B_O", "B_S",
            "C_Decision", "C_Reason"
        ]
    
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
                        if isinstance(parsed, dict) and "reviews" in parsed:
                            result.clear()
                            result.update(parsed)
                    except json.JSONDecodeError as e:
                        raise Exception(f"Invalid JSON in Model B response content: {content}. Error: {str(e)}")

            if not isinstance(result, dict):
                raise Exception("Invalid Model B response format: result is not a dictionary")
                
            if "reviews" not in result:
                raise Exception("Invalid Model B response format: missing 'reviews' field")
                
            if not isinstance(result["reviews"], list):
                raise Exception("Invalid Model B response format: 'reviews' is not a list")
                
            if not result["reviews"]:
                raise Exception("Empty reviews array in Model B response")
            
            # Validate each review
            for i, item in enumerate(result["reviews"]):
                if not isinstance(item, dict):
                    raise Exception(f"Invalid review item format: {item}")
                    
                if "Index" not in item:
                    raise Exception(f"Missing 'Index' in review item: {item}")
                
                # Check required fields according to the new prompt format
                required_fields = self.required_columns["model_b"]
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    raise Exception(f"Missing fields in Model B result: {missing_fields}")
                
        else:  # model_c
            if not isinstance(result, dict):
                raise Exception("Invalid Model C response format: result is not a dictionary")
            
            if "decisions" not in result:
                raise Exception("Invalid Model C response format: missing 'decisions' field")
            
            if not isinstance(result["decisions"], list):
                raise Exception("Invalid Model C response format: 'decisions' is not a list")
            
            if not result["decisions"]:
                raise Exception("Empty decisions array in Model C response")
            
            # Validate each decision
            for item in result["decisions"]:
                if not isinstance(item, dict):
                    raise Exception(f"Invalid decision item format: {item}")
                
                if "Index" not in item:
                    raise Exception(f"Missing 'Index' in decision item: {item}")
                
                if "C_Decision" not in item:
                    raise Exception(f"Missing 'C_Decision' in decision item: {item}")
                
                if "C_Reason" not in item:
                    raise Exception(f"Missing 'C_Reason' in decision item: {item}")
                
                # Ensure correct data types
                try:
                    str(item["Index"])
                    bool(item["C_Decision"])
                    str(item["C_Reason"])
                except (ValueError, TypeError) as e:
                    raise Exception(f"Invalid data type in decision item: {str(e)}")
    
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
        
        # Ensure index name is correct
        merged_df.index.name = "Index"
        
        # Remove possible Index column (if it exists as a regular column)
        if "Index" in merged_df.columns:
            merged_df = merged_df.drop(columns=["Index"])
        
        # Keep only needed columns and sort them in specified order
        merged_df = merged_df[required_columns]
        
        return merged_df 