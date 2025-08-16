import pandas as pd
import logging
import json
from typing import Dict, List, Optional
from model_manager import ModelManager
from prompt_manager import PromptManager
from result_processor import ResultProcessor
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class PICOSAnalyzer:
    def __init__(self):
        # Initialize managers for models, prompts, and result processing
        self.model_manager = ModelManager()
        self.prompt_manager = PromptManager()
        self.result_processor = ResultProcessor()
        # Example PICOS filtering criteria
        self.picos_criteria = {
            "population": "patients with non-alcoholic fatty liver disease (NAFLD)",
            "intervention": "observation or management of NAFLD",
            "comparison": "patients without NAFLD or general population",
            "outcome": "incidence of various types of extra-hepatic cancers, such as colorectal cancer, stomach cancer, breast cancer, etc.",
            "study_design": "retrospective cohort studies"
        }

    def update_picos_criteria(self, criteria: Dict[str, str]) -> None:
        """Update the PICOS criteria with a given dictionary of criteria."""
        self.picos_criteria.update(criteria)

    def update_model_config(self, model_key: str, config: Dict) -> None:
        """Update configuration settings for a specific model."""
        self.model_manager.update_model_config(model_key, config)

    def update_prompt(self, model_key: str, prompt: str) -> None:
        """Update the prompt template for a specific model."""
        self.prompt_manager.update_prompt(model_key, prompt)

    def test_api_connection(self, model_key: str) -> str:
        """Test the API connection for the specified model."""
        return self.model_manager.test_api_connection(model_key)

    def _validate_data(self, idx: str, row: pd.Series, model_key: str, previous_results: Dict) -> bool:
        """
        Validate the completeness of a single data item.

        Returns:
            Tuple[bool, bool]: (is_valid, is_empty_abstract)
        """
        try:
            # Check if abstract exists and is not empty
            if not pd.notna(row.get("Abstract")):
                logging.warning(f"Empty abstract for index {idx}")
                return False, True  # Second value indicates empty abstract

            # For Model B and C, validate Model A results
            if model_key in ["model_b", "model_c"]:
                if not previous_results or "model_a" not in previous_results:
                    logging.warning(f"Missing Model A results for {model_key}")
                    return False, False
                if idx not in previous_results["model_a"].index:
                    logging.warning(f"Index {idx} not found in Model A results")
                    return False, False

            # For Model C, validate Model B results
            if model_key == "model_c":
                if "model_b" not in previous_results:
                    logging.warning("Missing Model B results")
                    return False, False
                if idx not in previous_results["model_b"].index:
                    logging.warning(f"Index {idx} not found in Model B results")
                    return False, False

            return True, False
        except Exception as e:
            logging.error(f"Validation error for index {idx}: {str(e)}")
            return False, False

    def _process_single_item(self, idx: str, row: pd.Series, model_key: str, previous_results: Dict) -> Optional[Dict]:
        """
        Process a single data item and prepare it for API call.
        """
        try:
            # Prepare base result with abstract
            result = {
                "Index": idx,
                "abstract": str(row["Abstract"]).strip()
            }

            # Add Model A results for Model B and C
            if model_key in ["model_b", "model_c"]:
                a_result = previous_results["model_a"].loc[idx]
                result["model_a_analysis"] = {
                    "A_Decision": bool(a_result["A_Decision"]),
                    "A_Reason": str(a_result["A_Reason"]),
                    "A_P": str(a_result["A_P"]),
                    "A_I": str(a_result["A_I"]),
                    "A_C": str(a_result["A_C"]),
                    "A_O": str(a_result["A_O"]),
                    "A_S": str(a_result["A_S"])
                }

            # Add Model B results for Model C
            if model_key == "model_c":
                b_result = previous_results["model_b"].loc[idx]
                result["model_b_analysis"] = {
                    "B_Decision": bool(b_result["B_Decision"]),
                    "B_Reason": str(b_result["B_Reason"]),
                    "B_P": str(b_result["B_P"]),
                    "B_I": str(b_result["B_I"]),
                    "B_C": str(b_result["B_C"]),
                    "B_O": str(b_result["B_O"]),
                    "B_S": str(b_result["B_S"])
                }

            return result
        except Exception as e:
            logging.error(f"Processing error for index {idx}: {str(e)}")
            return None

    def _process_api_response(self, response: Dict, model_key: str) -> List[Dict]:
        """
        Process API response and extract results.
        """
        try:
            if not response or not isinstance(response, dict):
                logging.error(f"Invalid response format from {model_key}")
                return []

            # Extract results from response
            if "results" not in response:
                # For inference mode, try to parse from content directly (model_c only)
                if model_key == "model_c" and self.model_manager.get_config(model_key).get("is_inference"):
                    try:
                        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                        if json_match:
                            content = json_match.group(1)
                        parsed_response = json.loads(content)
                        if "results" not in parsed_response:
                            logging.error(f"No results found in {model_key} inference response")
                            return []
                        response = parsed_response
                    except Exception as e:
                        logging.error(f"Failed to parse inference response from {model_key}: {str(e)}")
                        return []
                else:
                    logging.error(f"No results found in {model_key} response")
                    return []

            results = response["results"]
            if not isinstance(results, list):
                logging.error(f"Results from {model_key} is not a list")
                return []

            # Validate each result
            valid_results = []
            for result in results:
                if not isinstance(result, dict) or "Index" not in result:
                    logging.warning(f"Invalid result format in {model_key} response: {result}")
                    continue

                # Ensure all required fields are present based on model type
                if model_key == "model_a":
                    required_fields = ["A_P", "A_I", "A_C", "A_O", "A_S", "A_Decision", "A_Reason"]
                elif model_key == "model_b":
                    required_fields = ["B_P", "B_I", "B_C", "B_O", "B_S", "B_Decision", "B_Reason"]
                else:  # model_c
                    required_fields = ["C_Decision", "C_Reason"]

                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    logging.warning(f"Missing fields {missing_fields} in {model_key} result for Index {result['Index']}")
                    continue

                # Convert decision to boolean if it's a string
                if model_key == "model_c" and isinstance(result.get("C_Decision"), str):
                    result["C_Decision"] = result["C_Decision"].lower() == "true"

                valid_results.append(result)

            return valid_results

        except Exception as e:
            logging.error(f"Error processing {model_key} response: {str(e)}")
            return []

    def process_batch(self, df: pd.DataFrame, model_key: str, previous_results: Dict = None, progress_callback=None) -> pd.DataFrame:
        """
        Process a batch of data with improved data flow and validation.
        """
        # Get model configuration
        config = self.model_manager.get_config(model_key)
        batch_size = config["batch_size"]
        threads = config["threads"]
        results_dict = {}  # Use dictionary to prevent duplicate indices
        failed_indices = set()
        total_rows = len(df)
        start_time = time.time()
        processed_count = 0
        skipped_count = 0

        # Ensure consistent index type
        df.index = df.index.astype(str)
        if previous_results:
            for key in previous_results:
                previous_results[key].index = previous_results[key].index.astype(str)

        # For Model C, first identify indices where A and B disagree
        if model_key == "model_c":
            disagreement_indices = []
            for idx in df.index:
                try:
                    if not self._validate_previous_results(idx, model_key, previous_results):
                        empty_result = self._create_empty_result(idx, model_key, "Invalid or missing previous results")
                        results_dict[str(idx)] = empty_result
                        failed_indices.add(str(idx))
                        if progress_callback:
                            progress_callback(idx, True, False)
                        continue

                    if self._check_disagreement(idx, previous_results):
                        disagreement_indices.append(idx)
                    else:
                        # If no disagreement, use Model A's decision
                        no_disagreement_result = self._create_no_disagreement_result(idx, previous_results)
                        results_dict[str(idx)] = no_disagreement_result
                        skipped_count += 1
                        if progress_callback:
                            progress_callback(idx, False, False)
                except Exception as e:
                    logging.error(f"Error checking disagreement for index {idx}: {str(e)}")
                    empty_result = self._create_empty_result(idx, model_key, f"Error: {str(e)}")
                    results_dict[str(idx)] = empty_result
                    failed_indices.add(str(idx))
                    if progress_callback:
                        progress_callback(idx, True, False)

            # Update df to only include disagreement cases for Model C
            if disagreement_indices:
                df = df.loc[disagreement_indices]
            else:
                # If no disagreements, return results with default values
                results = list(results_dict.values())
                results_df = pd.DataFrame(results)
                results_df.set_index("Index", inplace=True)
                results_df.index = results_df.index.astype(str)
                return results_df

        def process_batch_data(batch_df: pd.DataFrame) -> List[Dict]:
            nonlocal processed_count, skipped_count
            batch_results = []
            empty_results = []

            # Process each item in the batch
            for idx, row in batch_df.iterrows():
                try:
                    # Skip if already processed (for Model C)
                    if str(idx) in results_dict:
                        skipped_count += 1
                        continue

                    # Validate data completeness
                    is_valid, is_empty = self._validate_data(idx, row, model_key, previous_results)
                    if not is_valid:
                        empty_result = self._create_empty_result(idx, model_key, "Not processed - Empty abstract" if is_empty else "Not processed - Invalid data")
                        empty_results.append(empty_result)
                        failed_indices.add(idx)
                        if progress_callback:
                            progress_callback(idx, True, is_empty)
                        continue

                    # Prepare data for API call
                    abstract_text = row.get("Abstract", "").strip()
                    if not abstract_text:
                        empty_result = self._create_empty_result(idx, model_key, "Not processed - Empty abstract")
                        empty_results.append(empty_result)
                        failed_indices.add(idx)
                        if progress_callback:
                            progress_callback(idx, True, True)
                        continue

                    # Add to batch for processing
                    batch_item = self._process_single_item(idx, row, model_key, previous_results)
                    if batch_item:
                        batch_results.append(batch_item)
                    else:
                        empty_result = self._create_empty_result(idx, model_key, "Error preparing batch data")
                        empty_results.append(empty_result)
                        failed_indices.add(idx)
                        if progress_callback:
                            progress_callback(idx, True, False)

                except Exception as e:
                    logging.error(f"Error preparing data for index {idx}: {str(e)}")
                    empty_result = self._create_empty_result(idx, model_key, f"Error: {str(e)}")
                    empty_results.append(empty_result)
                    failed_indices.add(idx)
                    if progress_callback:
                        progress_callback(idx, True, False)

            # Process batch with API if there are valid entries
            if batch_results:
                try:
                    # Prepare prompt with PICOS criteria and batch data
                    prompt = self.prompt_manager.get_prompt(model_key).format(
                        **{
                            **self.picos_criteria,
                            "abstracts_json": json.dumps(batch_results, ensure_ascii=False, indent=2)
                        }
                    )

                    # Call API and process response
                    response = self.model_manager.call_api(model_key, prompt)
                    api_results = self._process_api_response(response, model_key)

                    # If API call failed or returned no results, create empty results for all items
                    if not api_results:
                        for item in batch_results:
                            empty_result = self._create_empty_result(item["Index"], model_key, "API call failed or returned no results")
                            empty_results.append(empty_result)
                            if progress_callback:
                                progress_callback(item["Index"], True, False)
                    else:
                        # Update progress for successfully processed items
                        for result in api_results:
                            if progress_callback:
                                progress_callback(result["Index"], False, False)
                            # Add result to the batch results
                            results_dict[str(result["Index"])] = result
                            processed_count += 1

                        # Calculate time statistics
                        elapsed_time = time.time() - start_time
                        if processed_count > 0:
                            avg_time_per_item = elapsed_time / processed_count
                            remaining_items = total_rows - (processed_count + len(failed_indices) + skipped_count)
                            estimated_remaining_time = avg_time_per_item * remaining_items

                            # Log detailed progress information
                            logging.info(
                                f"{model_key.upper()} Progress: "
                                f"Processed: {processed_count} - "
                                f"Remaining: {remaining_items} - "
                                f"Skipped: {skipped_count} - "
                                f"Elapsed Time: {elapsed_time:.1f}s - "
                                f"Est. Remaining: {estimated_remaining_time:.1f}s"
                            )

                    return api_results + empty_results

                except Exception as e:
                    error_msg = f"Error processing batch: {str(e)}"
                    logging.error(error_msg)
                    for item in batch_results:
                        empty_result = self._create_empty_result(item["Index"], model_key, error_msg)
                        empty_results.append(empty_result)
                        failed_indices.add(item["Index"])
                        if progress_callback:
                            progress_callback(item["Index"], True, False)

            return empty_results

        # Process batches using thread pool
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                futures.append(executor.submit(process_batch_data, batch_df))

            # Collect results
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    # Store results in dictionary to handle potential duplicates
                    for result in batch_results:
                        idx = str(result["Index"])
                        results_dict[idx] = result
                except Exception as e:
                    error_msg = f"Error collecting batch results: {str(e)}"
                    logging.error(error_msg)

        # Convert results dictionary to DataFrame
        results = list(results_dict.values())
        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # Set index properly
            results_df.set_index("Index", inplace=True)
            results_df.index = results_df.index.astype(str)

            # Ensure all required columns exist with default values
            for col in self._get_model_columns(model_key):
                if col not in results_df.columns:
                    if col.endswith("_Decision"):
                        results_df[col] = False
                    elif col.endswith("_Reason"):
                        results_df[col] = "Not provided"
                    else:
                        results_df[col] = "not applicable"

            # Convert boolean columns
            decision_columns = [col for col in results_df.columns if col.endswith("_Decision")]
            for col in decision_columns:
                results_df[col] = results_df[col].astype(bool)
        else:
            # Create empty DataFrame with required columns
            results_df = pd.DataFrame(columns=self._get_model_columns(model_key))
            results_df.index.name = "Index"

        # Log final statistics
        total_time = time.time() - start_time
        success_rate = ((total_rows - len(failed_indices)) / total_rows) * 100
        logging.info(f"{model_key.upper()} completed in {total_time:.1f}s - "
                    f"Success rate: {success_rate:.1f}% ({total_rows - len(failed_indices)}/{total_rows})")

        return results_df

    def merge_results(self, df: pd.DataFrame, model_results: Dict) -> pd.DataFrame:
        """Merge results from all models into a single DataFrame."""
        return self.result_processor.merge_results(df, model_results)

    def _create_empty_result(self, idx: str, model_key: str, reason: Optional[str] = None) -> Dict:
        """
        Create a default empty result entry for cases where the abstract is empty
        or previous results are missing. The default reason is 'Not applicable' if not provided.
        """
        default_reason = reason if reason is not None else "Not applicable - Empty or invalid data"
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
                "B_P": "not applicable",
                "B_I": "not applicable",
                "B_C": "not applicable",
                "B_O": "not applicable",
                "B_S": "not applicable",
                "B_Decision": False,
                "B_Reason": default_reason
            })
        else:  # For model_c
            result.update({
                "C_Decision": False,
                "C_Reason": default_reason
            })
        return result

    def _create_no_disagreement_result(self, idx: str, previous_results: Dict) -> Dict:
        """
        When Model A and Model B agree on the decision,
        directly return Model A's result with a note indicating no disagreement.
        """
        str_idx = str(idx)
        a_result = previous_results["model_a"].loc[str_idx]
        return {
            "Index": str_idx,
            "C_Decision": a_result["A_Decision"],
            "C_Reason": "No disagreement between Model A and B"
        }

    def _validate_previous_results(self, idx: str, model_key: str, previous_results: Dict) -> bool:
        """
        Validate if previous model results exist for a given index.
        Returns False if any required result is missing.
        """
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
        """
        Check whether there is a disagreement between Model A and Model B for a given index.
        Returns True if the decisions differ, otherwise False.
        """
        str_idx = str(idx)
        a_result = previous_results["model_a"].loc[str_idx]
        b_result = previous_results["model_b"].loc[str_idx]
        return a_result["A_Decision"] != b_result["B_Decision"]

    def _get_model_columns(self, model_key: str) -> List[str]:
        """Get the expected columns for a specific model's output."""
        if model_key == "model_a":
            return ["A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S"]
        elif model_key == "model_b":
            return ["B_Decision", "B_Reason", "B_P", "B_I", "B_C", "B_O", "B_S"]
        else:  # model_c
            return ["C_Decision", "C_Reason"]
