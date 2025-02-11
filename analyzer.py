import pandas as pd
import logging
import json
import concurrent.futures
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
        
        Args:
            idx: Index of the data item
            row: Data row from DataFrame
            model_key: Identifier of the model
            previous_results: Results from previous models
            
        Returns:
            bool: True if data is valid, False otherwise
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
        
        Args:
            idx: Index of the data item
            row: Data row from DataFrame
            model_key: Identifier of the model
            previous_results: Results from previous models
            
        Returns:
            Dict containing processed item or None if processing fails
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
        
        Args:
            response: Raw API response.
            model_key: Identifier of the model.
            
        Returns:
            List of processed results.
        """
        try:
            if not response or not isinstance(response, dict):
                logging.error(f"Invalid response format from {model_key}")
                return []
            
            # Extract results from response
            if "results" not in response:
                # For inference mode, try to parse from content directly
                if model_key == "model_c" and self.model_manager.get_config(model_key).get("is_inference"):
                    try:
                        # Try to parse JSON from the response content
                        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                        # Extract JSON from markdown code block if present
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
                
                # Check for missing fields
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
        
        Args:
            df: Input DataFrame containing abstracts to process
            model_key: Identifier of the model to use
            previous_results: Dictionary containing results from previous models
            progress_callback: Optional callback function to report progress
            
        Returns:
            DataFrame containing processed results with consistent index
        """
        # Get model configuration
        config = self.model_manager.get_config(model_key)
        batch_size = config["batch_size"]
        threads = config["threads"]
        results_dict = {}  # Use dictionary to prevent duplicate indices
        failed_indices = set()
        total_rows = len(df)
        start_time = time.time()

        # Ensure consistent index type
        df.index = df.index.astype(str)
        if previous_results:
            for key in previous_results:
                previous_results[key].index = previous_results[key].index.astype(str)

        def process_batch_data(batch_df: pd.DataFrame) -> List[Dict]:
            nonlocal total_rows  # Add reference to outer variables
            batch_results = []
            empty_results = []
            processed_count = 0

            # Process each item in the batch
            for idx, row in batch_df.iterrows():
                try:
                    # Validate data completeness
                    is_valid, is_empty = self._validate_data(idx, row, model_key, previous_results)
                    if not is_valid:
                        empty_results.append(self._create_empty_result(idx, model_key, "Not processed - Empty abstract" if is_empty else "Not processed - Invalid data"))
                        failed_indices.add(idx)
                        if progress_callback:
                            progress_callback(idx, True, is_empty)
                        continue

                    # Prepare data for API call
                    abstract_text = row.get("Abstract", "").strip()
                    if not abstract_text:
                        empty_results.append(self._create_empty_result(idx, model_key, "Not processed - Empty abstract"))
                        failed_indices.add(idx)
                        if progress_callback:
                            progress_callback(idx, True, True)
                        continue

                    # Add to batch for processing
                    batch_results.append({
                        "Index": str(idx),
                        "abstract": abstract_text
                    })

                except Exception as e:
                    logging.error(f"Error preparing data for index {idx}: {str(e)}")
                    empty_results.append(self._create_empty_result(idx, model_key, f"Error: {str(e)}"))
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
                        # Update progress for processed items
                        if progress_callback:
                            for result in api_results:
                                progress_callback(result["Index"], False, False)
                        
                        return api_results + empty_results
                        
                except Exception as e:
                    error_msg = f"Error processing batch: {str(e)}"
                    logging.error(error_msg)
                    for item in batch_results:
                        empty_results.append(self._create_empty_result(item["Index"], model_key, error_msg))
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
                    # Don't store failed results here, they will be handled in the next step

        # Ensure all indices from original DataFrame have results
        for idx in df.index:
            str_idx = str(idx)
            if str_idx not in results_dict:
                error_msg = "Failed to process - API error or invalid response"
                results_dict[str_idx] = self._create_empty_result(str_idx, model_key, error_msg)
                failed_indices.add(str_idx)

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
        
        # Log the final DataFrame info for debugging
        logging.debug(f"Final DataFrame info: {results_df.info()}")
        logging.debug(f"Final DataFrame columns: {results_df.columns.tolist()}")
        logging.debug(f"Sample of results:\n{results_df.head()}")
        
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
        Returns False if any required result is missing, so that an empty entry can be generated.
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

    def _call_model(self, model_key: str, batch_results: List[Dict], previous_results: Dict = None) -> List[Dict]:
        """
        Dispatch the API call based on the model_key.
        """
        if model_key == "model_a":
            return self._call_model_a(batch_results)
        elif model_key == "model_b":
            return self._call_model_b(batch_results, previous_results)
        else:  # model_c
            return self._call_model_c(batch_results, previous_results)

    def _call_model_a(self, abstracts: List[Dict]) -> List[Dict]:
        """Call the API for Model A."""
        results = []  # List to store all results (both empty and API results)

        # Create a default empty result for each index
        for abstract in abstracts:
            idx = str(abstract["Index"])
            empty_result = self._create_empty_result(idx, "model_a", reason="Not applicable - Processing")
            results.append(empty_result)

        # Prepare batch data with valid abstracts only
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

        # If valid batch data exists, call the API
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

                # Update the corresponding entries in the results list with API results
                for api_result in api_results:
                    idx = str(api_result["Index"])
                    result_idx = next(i for i, r in enumerate(results) if r["Index"] == idx)
                    results[result_idx].update(api_result)

            except Exception as e:
                logging.error(f"Error in Model A processing: {str(e)}")
                # Retain the default empty results if an error occurs

        return results

    def _parse_api_response(self, response: dict) -> dict:
        """
        General function for parsing API responses:
          - Checks the response format.
          - Extracts the content from the message (supports markdown formatted ```json code blocks).
          - Parses the content into a Python dictionary.
        """
        logging.debug(f"Parsing API response: {json.dumps(response, indent=2, ensure_ascii=False)}")
        
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
            
        logging.debug(f"Extracted content from response: {content}")
        
        # Extract JSON content from markdown if present
        if "```json" in content:
            pattern = r"```json\s*(.*?)\s*```"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = match.group(1)
                logging.debug(f"Extracted JSON from markdown: {content}")
        try:
            parsed_content = json.loads(content)
            logging.debug(f"Successfully parsed content to JSON: {json.dumps(parsed_content, indent=2, ensure_ascii=False)}")
            return parsed_content
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse API response content: {content}")
            logging.error(f"JSON parse error details: {str(e)}")
            raise Exception(f"Failed to parse API response: {str(e)}")

    def _call_model_b(self, abstracts: List[Dict], previous_results: Dict) -> List[Dict]:
        """
        Call the API for Model B.
        This function prepares the batch data based on Model A's results.
        """
        if "model_a" not in previous_results:
            raise Exception("Model A results required")

        model_a_df = previous_results["model_a"]
        batch_data = []
        results = []  # List to store both empty and API results

        # Create a default empty result for each index
        for abstract in abstracts:
            idx = str(abstract["Index"])
            empty_result = self._create_empty_result(idx, "model_b", reason="Not applicable - Processing")
            results.append(empty_result)

        # Process each abstract for batching
        for abstract in abstracts:
            try:
                idx = str(abstract["Index"])
                # Ensure that Model A result exists for this index
                if idx not in model_a_df.index.astype(str).values:
                    logging.warning(f"Index {idx} not found in Model A results")
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["B_Reason"] = "Not applicable - Missing Model A result"
                    continue

                a_result = model_a_df.loc[idx]
                if not isinstance(a_result, pd.Series):
                    logging.warning(f"Unexpected result type for index {idx}")
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["B_Reason"] = "Not applicable - Invalid Model A result format"
                    continue

                # Verify required fields exist in Model A's result
                required_fields = ["A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S"]
                missing_fields = [field for field in required_fields if field not in a_result]
                if missing_fields:
                    logging.warning(f"Missing required fields in Model A result for index {idx}: {missing_fields}")
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["B_Reason"] = "Not applicable - Incomplete Model A result"
                    continue

                # Prepare batch data including Model A analysis
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

        # If there is valid batch data, call the API for Model B
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

                # Update corresponding results with API response
                for api_result in api_results:
                    idx = str(api_result["Index"])
                    result_idx = next(i for i, r in enumerate(results) if r["Index"] == idx)
                    results[result_idx].update(api_result)

            except Exception as e:
                logging.error(f"Error calling Model B API: {str(e)}")
                # Retain existing empty results in case of error

        return results

    def _call_model_c(self, abstracts: List[Dict], previous_results: Dict) -> List[Dict]:
        """
        Call the API for Model C.
        This function combines results from Model A and Model B to determine the final decision.
        """
        results = []  # List to store both empty and API results

        # Create a default empty result for each index
        for abstract in abstracts:
            idx = str(abstract["Index"])
            empty_result = self._create_empty_result(idx, "model_c", reason="Not applicable - Processing")
            results.append(empty_result)

        # Prepare valid batch data for processing disagreements
        batch_data = []
        for abstract in abstracts:
            try:
                idx = str(abstract["Index"])
                
                # Check if both Model A and Model B results exist for the current index
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

                # Check for the presence of required fields in both Model A and Model B results
                a_required_fields = ["A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S"]
                b_required_fields = ["B_Decision", "B_Reason", "B_P", "B_I", "B_C", "B_O", "B_S"]
                
                missing_a_fields = [f for f in a_required_fields if f not in a_result]
                missing_b_fields = [f for f in b_required_fields if f not in b_result]
                
                if missing_a_fields or missing_b_fields:
                    logging.warning(f"Missing required fields for index {idx}: Model A: {missing_a_fields}, Model B: {missing_b_fields}")
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)]["C_Reason"] = "Not applicable - Incomplete previous results"
                    continue

                # Check if there is a disagreement between Model A and B decisions
                a_decision = bool(a_result["A_Decision"])
                b_decision = bool(b_result["B_Decision"])
                
                if a_decision == b_decision:
                    results[next(i for i, r in enumerate(results) if r["Index"] == idx)].update({
                        "C_Decision": a_decision,
                        "C_Reason": "No disagreement between Model A and B"
                    })
                    continue

                # Add data for entries where there is a disagreement
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

        # If there is data with disagreements, call the API for Model C
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

                # Update corresponding entries with the API response
                for api_result in api_results:
                    idx = str(api_result["Index"])
                    result_idx = next(i for i, r in enumerate(results) if r["Index"] == idx)
                    results[result_idx].update(api_result)

            except Exception as e:
                logging.error(f"Error calling Model C API: {str(e)}")
                # Retain the default empty results if an error occurs

        return results

    def _get_model_columns(self, model_key: str) -> List[str]:
        """Get the expected columns for a specific model's output."""
        if model_key == "model_a":
            return ["A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S"]
        elif model_key == "model_b":
            return ["B_Decision", "B_Reason", "B_P", "B_I", "B_C", "B_O", "B_S"]
        else:  # model_c
            return ["C_Decision", "C_Reason"]
