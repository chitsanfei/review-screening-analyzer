import pandas as pd
import logging
from typing import Dict
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
            # 检查是否为 completion 格式
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
                        if isinstance(parsed, dict) and "analysis" in parsed:
                            result.clear()
                            result.update(parsed)
                    except json.JSONDecodeError as e:
                        raise Exception(f"Invalid JSON in Model A response content: {content}. Error: {str(e)}")
            
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
        """Merge all model results with correct column alignment and final decision computation."""
        
        # 复制并清洗原始 DataFrame 的索引（去除可能的空格）
        df = df.copy()
        df.index = df.index.astype(str).str.strip()
        
        # 处理原始数据中的空值及清洗基础列
        for col in ["Abstract", "DOI", "Title", "Authors"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else "")
                df[col] = df[col].replace(r'^[\s-]*$', "", regex=True)
        
        # 创建基础 DataFrame（merged_df）用于合并模型结果
        merged_df = df.copy()
        
        def join_model_results(base_df: pd.DataFrame, model_key: str) -> pd.DataFrame:
            """合并特定模型的结果，确保数据对齐，同时清洗索引和列名"""
            if model_key not in model_results:
                logging.warning(f"{model_key} results not found")
                # 为所有行创建默认值
                for col in self.required_columns[model_key]:
                    if col.endswith('_Decision'):
                        base_df[col] = False
                    elif col.endswith('_Reason'):
                        base_df[col] = "Not applicable - No model result"
                    else:
                        base_df[col] = "not applicable"
                return base_df
            
            model_df = model_results[model_key].copy()
            # 确保模型结果的索引和列名为字符串且去除空格
            model_df.index = model_df.index.astype(str).str.strip()
            model_df.columns = model_df.columns.astype(str).str.strip()
            
            # 确保所有必需的列都存在
            for col in self.required_columns[model_key]:
                if col not in model_df.columns:
                    if col.endswith('_Decision'):
                        model_df[col] = False
                    elif col.endswith('_Reason'):
                        model_df[col] = "Not applicable - Missing column"
                    else:
                        model_df[col] = "not applicable"
            
            # 对于原始数据中存在但模型结果中缺失的索引，添加默认值
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
            
            # 只选择必需的列
            model_df = model_df[self.required_columns[model_key]]
            
            # 使用左连接确保保留所有原始数据的索引
            result = pd.merge(
                base_df,
                model_df,
                left_index=True,
                right_index=True,
                how='left'
            )
            
            # 填充可能的 NaN 值（使用赋值避免 chained assignment 的 inplace 操作）
            for col in self.required_columns[model_key]:
                if col in result.columns:
                    if col.endswith('_Decision'):
                        result[col] = result[col].fillna(False)
                    elif col.endswith('_Reason'):
                        result[col] = result[col].fillna("Not applicable - Missing value")
                    else:
                        result[col] = result[col].fillna("not applicable")
            
            return result
        
        # 按顺序合并各个模型的结果
        merged_df = join_model_results(merged_df, "model_a")
        merged_df = join_model_results(merged_df, "model_b")
        
        # 合并 Model C 结果，如果没有则生成默认值
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
        
        # 计算最终决策
        def compute_final_decision(row):
            if pd.notna(row.get("C_Decision")):
                return row["C_Decision"]
            elif pd.notna(row.get("A_Decision")) and pd.notna(row.get("B_Decision")):
                if row["A_Decision"] == row["B_Decision"]:
                    return row["A_Decision"]
                else:
                    return row["B_Decision"]  # 分歧时使用 Model B 的结果
            elif pd.notna(row.get("B_Decision")):
                return row["B_Decision"]
            elif pd.notna(row.get("A_Decision")):
                return row["A_Decision"]
            else:
                return False  # 默认返回 False
        
        merged_df["Final_Decision"] = merged_df.apply(compute_final_decision, axis=1)
        
        # 定义最终输出列及其顺序
        output_cols = [
            "Title", "DOI", "Abstract", "Authors",
            *self.required_columns.get("model_a", []),
            *self.required_columns.get("model_b", []),
            *self.required_columns.get("model_c", []),
            "Final_Decision"
        ]
        
        # 确保所有必需的列都存在（缺失时赋予默认值）
        for col in output_cols:
            if col not in merged_df.columns:
                if col.endswith('Decision'):
                    merged_df[col] = False
                elif col.endswith('Reason'):
                    merged_df[col] = "Not applicable - Missing column"
                else:
                    merged_df[col] = ""
        
        # 按顺序选择存在的列
        existing_cols = [col for col in output_cols if col in merged_df.columns]
        merged_df = merged_df[existing_cols]
        
        # 最后再次清洗所有列的空值（使用赋值避免 chained assignment 的 inplace 操作）
        for col in merged_df.columns:
            if col.endswith('Decision'):
                merged_df[col] = merged_df[col].fillna(False)
            elif col.endswith('Reason'):
                merged_df[col] = merged_df[col].fillna("Not applicable - Missing value")
            elif col in ["Title", "DOI", "Abstract", "Authors"]:
                merged_df[col] = merged_df[col].fillna("").astype(str)
            else:
                merged_df[col] = merged_df[col].fillna("not applicable")
        
        # 将索引作为一列添加到最终结果中
        merged_df.insert(0, "Index", merged_df.index)
        
        return merged_df

    def export_to_excel(self, df: pd.DataFrame, filename: str) -> None:
        """将 DataFrame 导出为 xlsx 文件"""
        try:
            df.to_excel(filename, index=False)
            logging.info(f"Exported results to {filename} successfully.")
        except Exception as e:
            logging.error(f"Error exporting to Excel: {str(e)}")
