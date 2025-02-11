import pandas as pd
import logging
from typing import Dict
import json
import re

class ResultProcessor:
    def __init__(self):
        """Initialize ResultProcessor with required column definitions for each model"""
        # Define required columns for each model's output
        self.required_columns = {
            "model_a": ["A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S"],
            "model_b": ["B_Decision", "B_Reason", "B_P", "B_I", "B_C", "B_O", "B_S"],
            "model_c": ["C_Decision", "C_Reason"]
        }
        
        # Define the order of columns in the final Excel output
        self.output_columns = [
            "Index", 
            "A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S",
            "B_Decision", "B_Reason", "B_P", "B_I", "B_C", "B_O", "B_S",
            "C_Decision", "C_Reason"
        ]
    
    def validate_model_response(self, result: Dict, model_key: str) -> None:
        """
        Validate the response format from each model
        
        Args:
            result: The model's response to validate
            model_key: The identifier of the model ('model_a', 'model_b', or 'model_c')
            
        Raises:
            Exception: If the response format is invalid
        """
        if model_key == "model_a":
            # Check if response is in completion format
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
                if content:
                    try:
                        # Handle markdown-wrapped JSON content
                        json_content = content
                        if "```json" in content:
                            pattern = r"```json\s*(.*?)\s*```"
                            match = re.search(pattern, content, re.DOTALL)
                            if match:
                                json_content = match.group(1)
                        parsed = json.loads(json_content)
                        if isinstance(parsed, dict) and "analysis" in parsed:
                            result.clear()
                            result.update(parsed)
                    except json.JSONDecodeError as e:
                        raise Exception(f"Invalid JSON in Model A response content: {content}. Error: {str(e)}")
            
            # Validate Model A specific format
            if not isinstance(result, dict) or 'analysis' not in result or not isinstance(result['analysis'], list):
                raise Exception("Invalid Model A response format: missing 'analysis' array")
            if not result['analysis']:
                raise Exception("Empty analysis array in Model A response")
            for item in result['analysis']:
                if not isinstance(item, dict):
                    raise Exception(f"Invalid analysis item format: {item}")
                if 'Index' not in item:
                    raise Exception(f"Missing 'Index' in analysis item: {item}")
                missing_fields = [field for field in self.required_columns[model_key] if field not in item]
                if missing_fields:
                    raise Exception(f"Missing fields in analysis item: {missing_fields}")
                
        elif model_key == "model_b":
            # Handle Model B's response format
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
                if content:
                    try:
                        json_content = content
                        if "```json" in content:
                            pattern = r"```json\s*(.*?)\s*```"
                            match = re.search(pattern, content, re.DOTALL)
                            if match:
                                json_content = match.group(1)
                        parsed = json.loads(json_content)
                        if isinstance(parsed, dict) and "reviews" in parsed:
                            result.clear()
                            result.update(parsed)
                    except json.JSONDecodeError as e:
                        raise Exception(f"Invalid JSON in Model B response content: {content}. Error: {str(e)}")
    
            # Validate Model B specific format
            if not isinstance(result, dict):
                raise Exception("Invalid Model B response format: result is not a dictionary")
            if "reviews" not in result:
                raise Exception("Invalid Model B response format: missing 'reviews' field")
            if not isinstance(result["reviews"], list):
                raise Exception("Invalid Model B response format: 'reviews' is not a list")
            if not result["reviews"]:
                raise Exception("Empty reviews array in Model B response")
            for i, item in enumerate(result["reviews"]):
                if not isinstance(item, dict):
                    raise Exception(f"Invalid review item format: {item}")
                if "Index" not in item:
                    raise Exception(f"Missing 'Index' in review item: {item}")
                required_fields = self.required_columns["model_b"]
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    raise Exception(f"Missing fields in Model B result: {missing_fields}")
                
        else:  # model_c
            # Validate Model C specific format
            if not isinstance(result, dict):
                raise Exception("Invalid Model C response format: result is not a dictionary")
            if "decisions" not in result:
                raise Exception("Invalid Model C response format: missing 'decisions' field")
            if not isinstance(result["decisions"], list):
                raise Exception("Invalid Model C response format: 'decisions' is not a list")
            if not result["decisions"]:
                raise Exception("Empty decisions array in Model C response")
            for item in result["decisions"]:
                if not isinstance(item, dict):
                    raise Exception(f"Invalid decision item format: {item}")
                if "Index" not in item:
                    raise Exception(f"Missing 'Index' in decision item: {item}")
                if "C_Decision" not in item:
                    raise Exception(f"Missing 'C_Decision' in decision item: {item}")
                if "C_Reason" not in item:
                    raise Exception(f"Missing 'C_Reason' in decision item: {item}")
                try:
                    str(item["Index"])
                    bool(item["C_Decision"])
                    str(item["C_Reason"])
                except (ValueError, TypeError) as e:
                    raise Exception(f"Invalid data type in decision item: {str(e)}")
    
    def merge_results(self, df: pd.DataFrame, model_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all model results with correct column alignment and compute final decision
        
        Args:
            df: Original DataFrame with abstracts
            model_results: Dictionary containing results from each model
            
        Returns:
            DataFrame with merged results from all models
        """
        try:
            # Copy and clean the original DataFrame's index (remove potential whitespace)
            df = df.copy()
            df.index = df.index.astype(str).str.strip()
            
            # Handle missing values and clean base columns
            for col in ["Abstract", "DOI", "Title", "Authors"]:
                if col in df.columns:
                    df[col] = df[col].fillna("").astype(str)
                    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else "")
                    df[col] = df[col].replace(r'^[\s-]*$', "", regex=True)
            
            # Create base DataFrame for merging model results
            merged_df = df.copy()
            
            def join_model_results(base_df: pd.DataFrame, model_key: str) -> pd.DataFrame:
                """
                Merge results from a specific model, ensuring data alignment and cleaning
                
                Args:
                    base_df: Base DataFrame to merge with
                    model_key: Identifier of the model
                    
                Returns:
                    DataFrame with merged model results
                """
                if model_key not in model_results:
                    logging.warning(f"{model_key} results not found")
                    # Create default values for all rows
                    for col in self.required_columns[model_key]:
                        if col.endswith('_Decision'):
                            base_df[col] = False
                        elif col.endswith('_Reason'):
                            base_df[col] = "Not applicable - No model result"
                        else:
                            base_df[col] = "not applicable"
                    return base_df
                
                try:
                    model_df = model_results[model_key].copy()
                    # Ensure model result indices and column names are strings without whitespace
                    model_df.index = model_df.index.astype(str).str.strip()
                    model_df.columns = model_df.columns.astype(str).str.strip()
                    
                    # Ensure all required columns exist
                    for col in self.required_columns[model_key]:
                        if col not in model_df.columns:
                            if col.endswith('_Decision'):
                                model_df[col] = False
                            elif col.endswith('_Reason'):
                                model_df[col] = "Not applicable - Missing column"
                            else:
                                model_df[col] = "not applicable"
                    
                    # Add default values for indices present in original data but missing in model results
                    missing_indices = set(base_df.index) - set(model_df.index)
                    if missing_indices:
                        logging.info(f"Found {len(missing_indices)} missing entries in {model_key}")
                        default_values = pd.DataFrame(
                            index=list(missing_indices),
                            columns=self.required_columns[model_key]
                        )
                        for col in self.required_columns[model_key]:
                            if col.endswith('_Decision'):
                                default_values[col] = False
                            elif col.endswith('_Reason'):
                                default_values[col] = "Not applicable - No result"
                            else:
                                default_values[col] = "not applicable"
                        model_df = pd.concat([model_df, default_values])
                    
                    # Select only required columns
                    model_df = model_df[self.required_columns[model_key]]
                    
                    # Use left join to preserve all original data indices
                    result = pd.merge(
                        base_df,
                        model_df,
                        left_index=True,
                        right_index=True,
                        how='left'
                    )
                    
                    # Fill potential NaN values
                    for col in self.required_columns[model_key]:
                        if col in result.columns:
                            if col.endswith('_Decision'):
                                result[col] = result[col].fillna(False)
                            elif col.endswith('_Reason'):
                                result[col] = result[col].fillna("Not applicable - Missing value")
                            else:
                                result[col] = result[col].fillna("not applicable")
                    
                    return result
                    
                except Exception as e:
                    logging.error(f"Error processing {model_key} results: {str(e)}")
                    # Return base DataFrame with default values
                    for col in self.required_columns[model_key]:
                        if col.endswith('_Decision'):
                            base_df[col] = False
                        elif col.endswith('_Reason'):
                            base_df[col] = f"Error processing {model_key} results: {str(e)}"
                        else:
                            base_df[col] = "not applicable"
                    return base_df
            
            # Merge results from each model in sequence
            merged_df = join_model_results(merged_df, "model_a")
            merged_df = join_model_results(merged_df, "model_b")
            
            # Merge Model C results or generate default values
            if "model_c" in model_results:
                merged_df = join_model_results(merged_df, "model_c")
            else:
                merged_df["C_Decision"] = False
                merged_df["C_Reason"] = merged_df.apply(
                    lambda row: "No disagreement between Model A and B" 
                        if pd.notna(row.get("A_Decision")) and pd.notna(row.get("B_Decision")) and row["A_Decision"] == row["B_Decision"]
                        else "Not applicable - No Model C result",
                    axis=1
                )
            
            # Compute final decision based on model results
            def compute_final_decision(row):
                """
                Compute final decision based on available model decisions
                Priority: Model C > Agreement between A&B > Model B > Model A > False
                """
                try:
                    if pd.notna(row.get("C_Decision")):
                        return bool(row["C_Decision"])
                    elif pd.notna(row.get("A_Decision")) and pd.notna(row.get("B_Decision")):
                        if bool(row["A_Decision"]) == bool(row["B_Decision"]):
                            return bool(row["A_Decision"])
                        else:
                            return bool(row["B_Decision"])  # Use Model B's result in case of disagreement
                    elif pd.notna(row.get("B_Decision")):
                        return bool(row["B_Decision"])
                    elif pd.notna(row.get("A_Decision")):
                        return bool(row["A_Decision"])
                except Exception as e:
                    logging.error(f"Error computing final decision: {str(e)}")
                return False
            
            merged_df["Final_Decision"] = merged_df.apply(compute_final_decision, axis=1)
            
            # Define final output columns and their order
            output_cols = [
                "Title", "DOI", "Abstract", "Authors",
                *self.required_columns.get("model_a", []),
                *self.required_columns.get("model_b", []),
                *self.required_columns.get("model_c", []),
                "Final_Decision"
            ]
            
            # Ensure all required columns exist (assign default values if missing)
            for col in output_cols:
                if col not in merged_df.columns:
                    if col.endswith('Decision'):
                        merged_df[col] = False
                    elif col.endswith('Reason'):
                        merged_df[col] = "Not applicable - Missing column"
                    else:
                        merged_df[col] = ""
            
            # Select existing columns in the specified order
            existing_cols = [col for col in output_cols if col in merged_df.columns]
            merged_df = merged_df[existing_cols]
            
            # Final cleaning of all column values
            for col in merged_df.columns:
                if col.endswith('Decision'):
                    merged_df[col] = merged_df[col].fillna(False).astype(bool)
                elif col.endswith('Reason'):
                    merged_df[col] = merged_df[col].fillna("Not applicable - Missing value")
                elif col in ["Title", "DOI", "Abstract", "Authors"]:
                    merged_df[col] = merged_df[col].fillna("").astype(str)
                else:
                    merged_df[col] = merged_df[col].fillna("not applicable")
            
            # Add index as a column in the final result
            merged_df.insert(0, "Index", merged_df.index)
            
            return merged_df
            
        except Exception as e:
            logging.error(f"Error merging results: {str(e)}")
            # Return a minimal DataFrame with error information
            error_df = pd.DataFrame(index=df.index)
            error_df["Error"] = f"Failed to merge results: {str(e)}"
            return error_df

    def export_to_excel(self, df: pd.DataFrame, filename: str) -> None:
        """
        Export DataFrame to Excel file
        
        Args:
            df: DataFrame to export
            filename: Target Excel file path
        """
        try:
            df.to_excel(filename, index=False)
            logging.info(f"Exported results to {filename} successfully.")
        except Exception as e:
            logging.error(f"Error exporting to Excel: {str(e)}")
