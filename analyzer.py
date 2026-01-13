"""
PICOS Analyzer - Core analysis orchestrator for medical literature screening.
Optimized for performance with batch processing and concurrent execution.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from model_manager import ModelManager
from prompt_manager import PromptManager
from result_processor import ResultProcessor

# Type aliases
ProgressCallback = Optional[Callable[[str, bool, bool], None]]


@dataclass(frozen=True)
class ModelColumns:
    """Column definitions for each model's output."""
    MODEL_A: Tuple[str, ...] = ("A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S")
    MODEL_B: Tuple[str, ...] = ("B_Decision", "B_Reason", "B_P", "B_I", "B_C", "B_O", "B_S")
    MODEL_C: Tuple[str, ...] = ("C_Decision", "C_Reason")

    def get(self, model_key: str) -> Tuple[str, ...]:
        return getattr(self, model_key.upper(), ())


MODEL_COLUMNS = ModelColumns()


class PICOSAnalyzer:
    """Main analyzer class for PICOS-based literature screening."""

    __slots__ = ('model_manager', 'prompt_manager', 'result_processor', 'picos_criteria')

    def __init__(self):
        self.model_manager = ModelManager()
        self.prompt_manager = PromptManager()
        self.result_processor = ResultProcessor()
        self.picos_criteria = {
            "population": "patients with non-alcoholic fatty liver disease (NAFLD)",
            "intervention": "observation or management of NAFLD",
            "comparison": "patients without NAFLD or general population",
            "outcome": "incidence of various types of extra-hepatic cancers",
            "study_design": "retrospective cohort studies"
        }

    def update_picos_criteria(self, criteria: Dict[str, str]) -> None:
        """Update PICOS criteria."""
        self.picos_criteria.update(criteria)

    def update_model_config(self, model_key: str, config: Dict) -> None:
        """Update model configuration."""
        self.model_manager.update_model_config(model_key, config)

    def update_prompt(self, model_key: str, prompt: str) -> None:
        """Update model prompt template."""
        self.prompt_manager.update_prompt(model_key, prompt)

    def test_api_connection(self, model_key: str) -> str:
        """Test API connection for specified model."""
        return self.model_manager.test_api_connection(model_key)

    def merge_results(self, df: pd.DataFrame, model_results: Dict) -> pd.DataFrame:
        """Merge results from all models."""
        return self.result_processor.merge_results(df, model_results)

    def process_batch(
        self,
        df: pd.DataFrame,
        model_key: str,
        previous_results: Optional[Dict] = None,
        progress_callback: ProgressCallback = None
    ) -> Optional[pd.DataFrame]:
        """
        Process a batch of abstracts with the specified model.

        Args:
            df: DataFrame containing abstracts to analyze
            model_key: Model identifier ('model_a', 'model_b', 'model_c')
            previous_results: Results from previous models (required for model_b/c)
            progress_callback: Optional callback for progress updates

        Returns:
            DataFrame with analysis results or None on failure
        """
        config = self.model_manager.get_config(model_key)
        batch_size = config["batch_size"]
        threads = config["threads"]

        # Normalize indices
        df = df.copy()
        df.index = df.index.astype(str)
        if previous_results:
            previous_results = {
                k: v.set_axis(v.index.astype(str))
                for k, v in previous_results.items()
            }

        results_dict: Dict[str, Dict] = {}
        failed_indices: Set[str] = set()
        start_time = time.time()

        # Handle Model C special case: only process disagreements
        if model_key == "model_c":
            df, results_dict, failed_indices = self._filter_disagreements(
                df, previous_results
            )
            if df.empty:
                return self._build_results_df(results_dict, model_key)

        # Process in batches with thread pool
        processor = BatchProcessor(
            self, model_key, previous_results, results_dict, failed_indices, progress_callback, retry_count=2
        )

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [
                executor.submit(processor.process_batch, df.iloc[i:i + batch_size])
                for i in range(0, len(df), batch_size)
            ]

            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    for result in batch_results:
                        results_dict[str(result["Index"])] = result
                except Exception as e:
                    logging.error(f"Batch processing error: {e}")

        # Build results DataFrame
        results_df = self._build_results_df(results_dict, model_key)

        # Log completion stats
        elapsed = time.time() - start_time
        # Count actual failures from processed items (not from original df)
        total_processed = len([idx for idx in df.index if str(idx) in results_dict])
        failed_processed = len([idx for idx in df.index if str(idx) in failed_indices])
        success_rate = (total_processed - failed_processed) / max(total_processed, 1) * 100
        logging.info(
            f"{model_key.upper()} completed in {elapsed:.1f}s - "
            f"Success rate: {success_rate:.1f}%"
        )

        return results_df

    def _filter_disagreements(
        self,
        df: pd.DataFrame,
        previous_results: Dict
    ) -> Tuple[pd.DataFrame, Dict, Set]:
        """Filter to only rows where Model A and B disagree."""
        results_dict = {}
        failed_indices = set()
        disagreement_indices = []

        for idx in df.index:
            str_idx = str(idx)
            try:
                if not self._validate_previous_results(str_idx, "model_c", previous_results):
                    results_dict[str_idx] = self._create_empty_result(
                        str_idx, "model_c", "Invalid or missing previous results"
                    )
                    failed_indices.add(str_idx)
                    continue

                a_decision = previous_results["model_a"].loc[str_idx, "A_Decision"]
                b_decision = previous_results["model_b"].loc[str_idx, "B_Decision"]

                if a_decision != b_decision:
                    disagreement_indices.append(idx)
                else:
                    results_dict[str_idx] = {
                        "Index": str_idx,
                        "C_Decision": a_decision,
                        "C_Reason": "No disagreement between Model A and B"
                    }

            except Exception as e:
                logging.error(f"Disagreement check error for {idx}: {e}")
                results_dict[str_idx] = self._create_empty_result(str_idx, "model_c", str(e))
                failed_indices.add(str_idx)

        filtered_df = df.loc[disagreement_indices] if disagreement_indices else df.iloc[0:0]
        return filtered_df, results_dict, failed_indices

    def count_disagreements(
        self,
        df: pd.DataFrame,
        previous_results: Dict
    ) -> int:
        """Count the number of disagreements between Model A and B."""
        disagreement_count = 0

        for idx in df.index:
            str_idx = str(idx)
            try:
                if not self._validate_previous_results(str_idx, "model_c", previous_results):
                    continue

                a_decision = previous_results["model_a"].loc[str_idx, "A_Decision"]
                b_decision = previous_results["model_b"].loc[str_idx, "B_Decision"]

                if a_decision != b_decision:
                    disagreement_count += 1
            except Exception:
                continue

        return disagreement_count

    def _validate_previous_results(
        self, idx: str, model_key: str, previous_results: Dict
    ) -> bool:
        """Validate that required previous model results exist."""
        if "model_a" not in previous_results:
            return False
        if idx not in previous_results["model_a"].index:
            return False
        if model_key == "model_c":
            if "model_b" not in previous_results:
                return False
            if idx not in previous_results["model_b"].index:
                return False
        return True

    def _create_empty_result(
        self, idx: str, model_key: str, reason: str = "Not applicable"
    ) -> Dict:
        """Create default empty result for failed processing."""
        result = {"Index": str(idx)}

        if model_key == "model_a":
            result.update({
                "A_P": "not applicable", "A_I": "not applicable",
                "A_C": "not applicable", "A_O": "not applicable",
                "A_S": "not applicable", "A_Decision": False, "A_Reason": reason
            })
        elif model_key == "model_b":
            result.update({
                "B_P": "not applicable", "B_I": "not applicable",
                "B_C": "not applicable", "B_O": "not applicable",
                "B_S": "not applicable", "B_Decision": False, "B_Reason": reason
            })
        else:
            result.update({"C_Decision": False, "C_Reason": reason})

        return result

    def _build_results_df(self, results_dict: Dict, model_key: str) -> pd.DataFrame:
        """Build DataFrame from results dictionary."""
        if not results_dict:
            columns = list(MODEL_COLUMNS.get(model_key))
            return pd.DataFrame(columns=columns).set_index(pd.Index([], name="Index"))

        results_df = pd.DataFrame(list(results_dict.values()))
        results_df.set_index("Index", inplace=True)
        results_df.index = results_df.index.astype(str)

        # Ensure all columns exist with proper defaults
        for col in MODEL_COLUMNS.get(model_key):
            if col not in results_df.columns:
                if col.endswith("_Decision"):
                    results_df[col] = False
                elif col.endswith("_Reason"):
                    results_df[col] = "Not provided"
                else:
                    results_df[col] = "not applicable"

        # Convert decision columns to boolean
        for col in results_df.columns:
            if col.endswith("_Decision"):
                results_df[col] = results_df[col].astype(bool)

        return results_df


class BatchProcessor:
    """Handles batch processing of abstracts for a specific model."""

    __slots__ = (
        'analyzer', 'model_key', 'previous_results', 'results_dict',
        'failed_indices', 'progress_callback', 'retry_count'
    )

    def __init__(
        self,
        analyzer: PICOSAnalyzer,
        model_key: str,
        previous_results: Optional[Dict],
        results_dict: Dict,
        failed_indices: Set,
        progress_callback: ProgressCallback,
        retry_count: int = 2
    ):
        self.analyzer = analyzer
        self.model_key = model_key
        self.previous_results = previous_results
        self.results_dict = results_dict
        self.failed_indices = failed_indices
        self.progress_callback = progress_callback
        self.retry_count = retry_count

    def process_batch(self, batch_df: pd.DataFrame) -> List[Dict]:
        """Process a single batch of abstracts."""
        batch_items = []
        empty_results = []

        for idx, row in batch_df.iterrows():
            str_idx = str(idx)

            # Skip if already processed
            if str_idx in self.results_dict:
                continue

            # Validate abstract
            abstract = row.get("Abstract", "")
            if not pd.notna(abstract) or not str(abstract).strip():
                empty_results.append(
                    self.analyzer._create_empty_result(str_idx, self.model_key, "Empty abstract")
                )
                self.failed_indices.add(str_idx)
                if self.progress_callback:
                    self.progress_callback(str_idx, False, True)
                continue

            # Validate previous results for Model B/C
            if self.model_key in ("model_b", "model_c"):
                if not self.analyzer._validate_previous_results(
                    str_idx, self.model_key, self.previous_results
                ):
                    empty_results.append(
                        self.analyzer._create_empty_result(
                            str_idx, self.model_key, "Missing previous results"
                        )
                    )
                    self.failed_indices.add(str_idx)
                    if self.progress_callback:
                        self.progress_callback(str_idx, False, True)
                    continue

            # Prepare batch item
            item = self._prepare_item(str_idx, row)
            if item:
                batch_items.append(item)
            else:
                empty_results.append(
                    self.analyzer._create_empty_result(str_idx, self.model_key, "Preparation error")
                )
                self.failed_indices.add(str_idx)
                if self.progress_callback:
                    self.progress_callback(str_idx, False, True)

        # Process batch with API
        if batch_items:
            api_results = self._call_api(batch_items)
            if api_results:
                for result in api_results:
                    if self.progress_callback:
                        self.progress_callback(result["Index"], True, False)
                return api_results + empty_results
            else:
                # API failed - create empty results for all items
                for item in batch_items:
                    empty_results.append(
                        self.analyzer._create_empty_result(
                            item["Index"], self.model_key, "API call failed"
                        )
                    )
                    if self.progress_callback:
                        self.progress_callback(item["Index"], False, True)

        return empty_results

    def _prepare_item(self, idx: str, row: pd.Series) -> Optional[Dict]:
        """Prepare a single item for API call."""
        try:
            item = {"Index": idx, "abstract": str(row["Abstract"]).strip()}

            if self.model_key in ("model_b", "model_c"):
                a_result = self.previous_results["model_a"].loc[idx]
                item["model_a_analysis"] = {
                    "A_Decision": bool(a_result["A_Decision"]),
                    "A_Reason": str(a_result["A_Reason"]),
                    "A_P": str(a_result["A_P"]),
                    "A_I": str(a_result["A_I"]),
                    "A_C": str(a_result["A_C"]),
                    "A_O": str(a_result["A_O"]),
                    "A_S": str(a_result["A_S"])
                }

            if self.model_key == "model_c":
                b_result = self.previous_results["model_b"].loc[idx]
                item["model_b_analysis"] = {
                    "B_Decision": bool(b_result["B_Decision"]),
                    "B_Reason": str(b_result["B_Reason"]),
                    "B_P": str(b_result["B_P"]),
                    "B_I": str(b_result["B_I"]),
                    "B_C": str(b_result["B_C"]),
                    "B_O": str(b_result["B_O"]),
                    "B_S": str(b_result["B_S"])
                }

            return item
        except Exception as e:
            logging.error(f"Item preparation error for {idx}: {e}")
            return None

    def _call_api(self, batch_items: List[Dict]) -> List[Dict]:
        """Call API and process response with retry logic."""
        prompt = None
        for attempt in range(self.retry_count + 1):
            try:
                if prompt is None:
                    prompt = self.analyzer.prompt_manager.get_prompt(self.model_key).format(
                        **self.analyzer.picos_criteria,
                        abstracts_json=json.dumps(batch_items, ensure_ascii=False)
                    )

                response = self.analyzer.model_manager.call_api(self.model_key, prompt)
                results = self._process_response(response)

                if results:
                    return results

                if attempt < self.retry_count:
                    logging.warning(f"API call attempt {attempt + 1} failed, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logging.error(f"API call failed after {self.retry_count + 1} attempts")

            except Exception as e:
                logging.error(f"API call error (attempt {attempt + 1}): {e}")
                if attempt < self.retry_count:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logging.error(f"API call failed after {self.retry_count + 1} attempts")

        return []

    def _process_response(self, response: Dict) -> List[Dict]:
        """Process and validate API response."""
        if not response or not isinstance(response, dict):
            return []

        results = response.get("results", [])
        if not isinstance(results, list):
            return []

        # Define required fields per model
        required_fields = {
            "model_a": ("A_P", "A_I", "A_C", "A_O", "A_S", "A_Decision", "A_Reason"),
            "model_b": ("B_P", "B_I", "B_C", "B_O", "B_S", "B_Decision", "B_Reason"),
            "model_c": ("C_Decision", "C_Reason")
        }.get(self.model_key, ())

        valid_results = []
        for result in results:
            if not isinstance(result, dict) or "Index" not in result:
                continue

            missing = [f for f in required_fields if f not in result]
            if missing:
                logging.warning(f"Missing fields {missing} for Index {result.get('Index')}")
                continue

            # Normalize decision to boolean
            decision_key = f"{self.model_key[-1].upper()}_Decision"
            if decision_key in result and isinstance(result[decision_key], str):
                result[decision_key] = result[decision_key].lower() == "true"

            valid_results.append(result)

        return valid_results
