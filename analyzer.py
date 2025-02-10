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
        # 示例的 PICOS 筛选标准
        self.picos_criteria = {
            "population": "patients with non-alcoholic fatty liver disease (NAFLD)",
            "intervention": "observation or management of NAFLD",
            "comparison": "patients without NAFLD or general population",
            "outcome": "incidence of various types of extra-hepatic cancers, such as colorectal cancer, stomach cancer, breast cancer, etc.",
            "study_design": "retrospective cohort studies"
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

    def process_batch(self, df: pd.DataFrame, model_key: str, previous_results: Dict = None, progress_callback=None) -> pd.DataFrame:
        """Process a batch of data, 保证即使空摘要或前置结果缺失，最终输出中也包含对应条目（理由为 Not applicable）"""
        config = self.model_manager.get_config(model_key)
        batch_size = config["batch_size"]
        threads = config["threads"]
        results = []
        failed_indices = set()
        total_rows = len(df)
        completed_rows = 0

        def process_batch_data(batch_df: pd.DataFrame) -> List[Dict]:
            nonlocal completed_rows, failed_indices
            batch_results = []     # 将用于调用 API 的条目（摘要有效且前置结果完整）
            empty_results = []     # 存储空摘要或不适合的条目

            for idx, row in batch_df.iterrows():
                try:
                    abstract_text = str(row["Abstract"]).strip() if pd.notna(row["Abstract"]) else ""
                    
                    # 如果摘要为空或长度不足，生成 Not applicable 记录
                    if not abstract_text or len(abstract_text) < 50:
                        logging.info(f"Row {idx}: Abstract too short or empty, marking as Not applicable")
                        empty_results.append(self._create_empty_result(idx, model_key, reason="Not applicable - Abstract insufficient"))
                        if progress_callback:
                            progress_callback(idx, False)  # 不将其标记为失败
                        continue

                    abstract = {"Index": str(idx), "abstract": abstract_text}

                    # 对于 Model B 和 C，检查前置模型结果
                    if model_key in ["model_b", "model_c"] and previous_results:
                        if not self._validate_previous_results(idx, model_key, previous_results):
                            empty_results.append(self._create_empty_result(idx, model_key, reason="Not applicable - Missing previous results"))
                            if progress_callback:
                                progress_callback(idx, False)  # 不将其标记为失败
                            continue

                        # 对于 Model C，若 Model A 与 Model B 结论一致，则不调用 API
                        if model_key == "model_c":
                            if not self._check_disagreement(idx, previous_results):
                                empty_results.append(self._create_no_disagreement_result(idx, previous_results))
                                if progress_callback:
                                    progress_callback(idx, False)
                                continue

                    batch_results.append(abstract)
                    if progress_callback:
                        progress_callback(idx, False)
                except Exception as e:
                    logging.error(f"Error processing row {idx}: {str(e)}")
                    failed_indices.add(idx)
                    if progress_callback:
                        progress_callback(idx, True)

            try:
                api_results = []
                if batch_results:
                    api_results = self._call_model(model_key, batch_results, previous_results)
                return api_results + empty_results
            except Exception as e:
                logging.error(f"Error in model call: {str(e)}")
                for abstract in batch_results:
                    failed_indices.add(abstract["Index"])
                return empty_results

        # 划分 batch
        batches = [df.iloc[i:i + batch_size].copy() for i in range(0, len(df), batch_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [(batch_df.index, executor.submit(process_batch_data, batch_df)) for batch_df in batches]
            for batch_indices, future in futures:
                try:
                    batch_results = future.result()
                    if batch_results:
                        results.extend(batch_results)
                except Exception as e:
                    for idx in batch_indices:
                        failed_indices.add(idx)
                    logging.error(f"Error processing batch: {str(e)}")
                finally:
                    completed_rows += len(batch_indices)

        if not results:
            raise Exception("No results were successfully processed")

        # 将结果转换为 DataFrame，确保包含 Index 列
        results_df = pd.DataFrame(results)
        results_df.set_index('Index', inplace=True)
        return results_df

    def merge_results(self, df: pd.DataFrame, model_results: Dict) -> pd.DataFrame:
        """Merge all model results"""
        return self.result_processor.merge_results(df, model_results)

    def _create_empty_result(self, idx: str, model_key: str, reason: Optional[str] = None) -> Dict:
        """创建空结果记录（用于摘要为空或缺失前置结果时），统一使用 reason 'Not applicable'"""
        default_reason = reason if reason is not None else "Not applicable"
        result = {"Index": str(idx)}
        if model_key == "model_a":
            result.update({
                "A_P": "not applicable",
                "A_I": "not applicable",
                "A_C": "not applicable",
                "A_O": "not applicable",
                "A_S": "not applicable",
                "A_Decision": False,
                "A_Reason": default_reason
            })
        elif model_key == "model_b":
            result.update({
                "B_P": "-",
                "B_I": "-",
                "B_C": "-",
                "B_O": "-",
                "B_S": "-",
                "B_Decision": False,
                "B_Reason": default_reason
            })
        else:  # model_c
            result.update({
                "C_Decision": False,
                "C_Reason": default_reason
            })
        return result

    def _create_no_disagreement_result(self, idx: str, previous_results: Dict) -> Dict:
        """当 Model A 与 Model B 结论一致时，直接返回 Model A 的结论，并注明 'No disagreement between Model A and B'"""
        str_idx = str(idx)
        a_result = previous_results["model_a"].loc[str_idx]
        return {
            "Index": str_idx,
            "C_Decision": a_result["A_Decision"],
            "C_Reason": "No disagreement between Model A and B"
        }

    def _validate_previous_results(self, idx: str, model_key: str, previous_results: Dict) -> bool:
        """验证前置模型结果是否存在；若缺失则返回 False，以便后续生成空记录"""
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
        """判断 Model A 与 Model B 是否存在分歧（返回布尔值）"""
        str_idx = str(idx)
        a_result = previous_results["model_a"].loc[str_idx]
        b_result = previous_results["model_b"].loc[str_idx]
        return a_result["A_Decision"] != b_result["B_Decision"]

    def _call_model(self, model_key: str, batch_results: List[Dict], previous_results: Dict = None) -> List[Dict]:
        """根据 model_key 分发调用相应的模型 API"""
        if model_key == "model_a":
            return self._call_model_a(batch_results)
        elif model_key == "model_b":
            return self._call_model_b(batch_results, previous_results)
        else:  # model_c
            return self._call_model_c(batch_results, previous_results)

    def _call_model_a(self, abstracts: List[Dict]) -> List[Dict]:
        """调用 Model A 的 API"""
        results = []  # 存储所有结果，包括空结果和API结果

        # 首先为每个索引创建一个默认的空结果
        for abstract in abstracts:
            idx = str(abstract["Index"])
            empty_result = self._create_empty_result(idx, "model_a", reason="Not applicable - Processing")
            results.append(empty_result)

        # 准备有效的批处理数据
        batch_data = []
        for abstract in abstracts:
            try:
                idx = str(abstract["Index"])
                abstract_text = abstract.get("abstract", "").strip()
                
                if not abstract_text:
                    logging.warning(f"Empty abstract for index {idx}")
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["A_Reason"] = "Not applicable - Empty abstract"
                    continue

                batch_data.append({
                    "index": str(idx),
                    "abstract": abstract_text
                })
            except Exception as e:
                logging.error(f"Error preparing batch data for index {abstract.get('Index')}: {str(e)}")
                results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["A_Reason"] = f"Not applicable - Error: {str(e)}"

        # 如果有有效的批处理数据，调用API
        if batch_data:
            try:
                abstracts_json = json.dumps(batch_data, indent=2)
                prompt = self.prompt_manager.get_prompt("model_a").format(
                    abstracts_json=abstracts_json,
                    **self.picos_criteria
                )
                response = self.model_manager.call_api("model_a", prompt)
                self.result_processor.validate_model_response(response, "model_a")
                api_results = response.get("analysis", [])

                # 更新结果列表中对应的条目
                for api_result in api_results:
                    idx = str(api_result["Index"])
                    result_idx = next(i for i, r in enumerate(results) if r["Index"] == idx)
                    results[result_idx].update(api_result)

            except Exception as e:
                logging.error(f"Error in Model A processing: {str(e)}")
                # 保持现有的空结果不变

        return results

    def _parse_api_response(self, response: dict) -> dict:
        """
        通用的 API 响应解析函数：
          - 检查响应格式是否正确；
          - 提取 message 中的 content（支持 markdown 格式的 ```json 代码块）；
          - 解析为 Python 字典
        """
        if not isinstance(response, dict):
            logging.error(f"Invalid response type: {type(response)}")
            raise Exception("Invalid response format: response is not a dictionary")
        if "choices" not in response:
            logging.error(f"Missing 'choices' in response: {response}")
            raise Exception("Invalid response format: missing 'choices' field")
        if not isinstance(response["choices"], list) or not response["choices"]:
            logging.error(f"Invalid 'choices' in response: {response['choices']}")
            raise Exception("Invalid response format: empty or invalid choices")
        content = response["choices"][0].get("message", {}).get("content", "")
        if not content:
            logging.error("Empty content in response")
            raise Exception("Invalid response format: empty content")
        if "```json" in content:
            pattern = r"```json\s*(.*?)\s*```"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = match.group(1)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse API response content: {content}")
            raise Exception(f"Failed to parse API response: {str(e)}")

    def _call_model_b(self, abstracts: List[Dict], previous_results: Dict) -> List[Dict]:
        """调用 Model B 的 API，基于 Model A 的结果准备数据"""
        if "model_a" not in previous_results:
            raise Exception("Model A results required")

        model_a_df = previous_results["model_a"]
        batch_data = []
        results = []  # 存储所有结果，包括空结果和API结果

        # 首先为每个索引创建一个默认的空结果
        for abstract in abstracts:
            idx = str(abstract["Index"])
            empty_result = self._create_empty_result(idx, "model_b", reason="Not applicable - Processing")
            results.append(empty_result)

        # 尝试处理每个摘要
        for abstract in abstracts:
            try:
                idx = str(abstract["Index"])
                # 确保 Model A 结果存在
                if idx not in model_a_df.index.astype(str).values:
                    logging.warning(f"Index {idx} not found in Model A results")
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["B_Reason"] = "Not applicable - Missing Model A result"
                    continue

                a_result = model_a_df.loc[idx]
                if not isinstance(a_result, pd.Series):
                    logging.warning(f"Unexpected result type for index {idx}")
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["B_Reason"] = "Not applicable - Invalid Model A result format"
                    continue

                # 检查 Model A 结果中必须的字段
                required_fields = ["A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S"]
                missing_fields = [field for field in required_fields if field not in a_result]
                if missing_fields:
                    logging.warning(f"Missing required fields in Model A result for index {idx}: {missing_fields}")
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["B_Reason"] = "Not applicable - Incomplete Model A result"
                    continue

                # 准备批处理数据
                batch_item = {
                    "Index": idx,
                    "abstract": abstract["abstract"],
                    "model_a_analysis": {
                        "A_Decision": bool(a_result["A_Decision"]),
                        "A_Reason": str(a_result["A_Reason"]),
                        "A_P": str(a_result["A_P"]),
                        "A_I": str(a_result["A_I"]),
                        "A_C": str(a_result["A_C"]),
                        "A_O": str(a_result["A_O"]),
                        "A_S": str(a_result["A_S"])
                    }
                }
                batch_data.append(batch_item)
            except Exception as e:
                logging.error(f"Error preparing batch data for index {abstract.get('Index')}: {str(e)}")
                results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["B_Reason"] = f"Not applicable - Error: {str(e)}"

        # 如果有有效的批处理数据，调用API
        if batch_data:
            try:
                prompt = self.prompt_manager.get_prompt("model_b").format(
                    abstracts_json=json.dumps(batch_data, indent=2),
                    **self.picos_criteria
                )
                response = self.model_manager.call_api("model_b", prompt)
                parsed_response = self._parse_api_response(response)
                self.result_processor.validate_model_response(parsed_response, "model_b")
                api_results = parsed_response.get("reviews", [])

                # 更新结果列表中对应的条目
                for api_result in api_results:
                    idx = str(api_result["Index"])
                    result_idx = next(i for i, r in enumerate(results) if r["Index"] == idx)
                    results[result_idx].update(api_result)

            except Exception as e:
                logging.error(f"Error calling Model B API: {str(e)}")
                # 保持现有的空结果不变

        return results

    def _call_model_c(self, abstracts: List[Dict], previous_results: Dict) -> List[Dict]:
        """调用 Model C 的 API，结合 Model A 和 Model B 的结果进行判断"""
        results = []  # 存储所有结果，包括空结果和API结果

        # 首先为每个索引创建一个默认的空结果
        for abstract in abstracts:
            idx = str(abstract["Index"])
            empty_result = self._create_empty_result(idx, "model_c", reason="Not applicable - Processing")
            results.append(empty_result)

        # 准备有效的批处理数据
        batch_data = []
        for abstract in abstracts:
            try:
                idx = str(abstract["Index"])
                
                # 检查前置结果是否存在
                if idx not in previous_results["model_a"].index.astype(str).values:
                    logging.warning(f"Missing Model A result for index {idx}")
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["C_Reason"] = "Not applicable - Missing Model A result"
                    continue
                    
                if idx not in previous_results["model_b"].index.astype(str).values:
                    logging.warning(f"Missing Model B result for index {idx}")
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["C_Reason"] = "Not applicable - Missing Model B result"
                    continue

                a_result = previous_results["model_a"].loc[idx]
                b_result = previous_results["model_b"].loc[idx]

                # 检查必需的字段
                a_required_fields = ["A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S"]
                b_required_fields = ["B_Decision", "B_Reason", "B_P", "B_I", "B_C", "B_O", "B_S"]
                
                missing_a_fields = [f for f in a_required_fields if f not in a_result]
                missing_b_fields = [f for f in b_required_fields if f not in b_result]
                
                if missing_a_fields or missing_b_fields:
                    logging.warning(f"Missing required fields for index {idx}: Model A: {missing_a_fields}, Model B: {missing_b_fields}")
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["C_Reason"] = "Not applicable - Incomplete previous results"
                    continue

                # 检查是否存在分歧
                a_decision = bool(a_result["A_Decision"])
                b_decision = bool(b_result["B_Decision"])
                
                if a_decision == b_decision:
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)].update({
                        "C_Decision": a_decision,
                        "C_Reason": "No disagreement between Model A and B"
                    })
                    continue

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
            except Exception as e:
                logging.error(f"Error preparing batch data for Model C index {idx}: {str(e)}")
                results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["C_Reason"] = f"Not applicable - Error: {str(e)}"

        # 如果有需要处理分歧的数据，调用API
        if batch_data:
            try:
                prompt = self.prompt_manager.get_prompt("model_c").format(
                    disagreements_json=json.dumps(batch_data, indent=2),
                    **self.picos_criteria
                )
                response = self.model_manager.call_api("model_c", prompt)
                parsed_response = self._parse_api_response(response)
                self.result_processor.validate_model_response(parsed_response, "model_c")
                api_results = parsed_response.get("decisions", [])

                # 更新结果列表中对应的条目
                for api_result in api_results:
                    idx = str(api_result["Index"])
                    result_idx = next(i for i, r in enumerate(results) if r["Index"] == idx)
                    results[result_idx].update(api_result)

            except Exception as e:
                logging.error(f"Error calling Model C API: {str(e)}")
                # 保持现有的空结果不变

        return results
