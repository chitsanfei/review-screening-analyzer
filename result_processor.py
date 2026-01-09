"""
Result Processor - Handles merging and validation of model results.
Optimized for efficient DataFrame operations.
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd

# Column definitions
MODEL_COLUMNS = {
    "model_a": ("A_Decision", "A_Reason", "A_P", "A_I", "A_C", "A_O", "A_S"),
    "model_b": ("B_Decision", "B_Reason", "B_P", "B_I", "B_C", "B_O", "B_S"),
    "model_c": ("C_Decision", "C_Reason")
}

BASE_COLUMNS = ("Title", "DOI", "Abstract", "Authors")


class ResultProcessor:
    """Processes and merges results from multiple analysis models."""

    __slots__ = ()

    def merge_results(
        self,
        df: pd.DataFrame,
        model_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merge results from all models into a single DataFrame.

        Args:
            df: Original DataFrame with abstracts
            model_results: Dictionary of model results DataFrames

        Returns:
            Merged DataFrame with all model results and final decision
        """
        try:
            # Prepare base DataFrame
            merged = df.copy()
            merged.index = merged.index.astype(str).str.strip()

            # Clean base columns
            for col in BASE_COLUMNS:
                if col in merged.columns:
                    merged[col] = (
                        merged[col]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        .replace(r'^[\s-]*$', "", regex=True)
                    )

            # Merge each model's results
            for model_key in ("model_a", "model_b", "model_c"):
                merged = self._merge_model_results(merged, model_key, model_results)

            # Compute final decision
            merged["Final_Decision"] = merged.apply(self._compute_final_decision, axis=1)

            # Organize output columns
            merged = self._organize_columns(merged)

            # Add Index as first column
            merged.insert(0, "Index", merged.index)

            return merged

        except Exception as e:
            logging.error(f"Merge error: {e}")
            error_df = pd.DataFrame(index=df.index)
            error_df["Error"] = f"Merge failed: {str(e)}"
            return error_df

    def _merge_model_results(
        self,
        base_df: pd.DataFrame,
        model_key: str,
        model_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge results from a specific model."""
        columns = MODEL_COLUMNS.get(model_key, ())

        if model_key not in model_results:
            # Add default values for missing model
            for col in columns:
                base_df[col] = self._get_default_value(col, "No model result")
            return base_df

        try:
            model_df = model_results[model_key].copy()
            model_df.index = model_df.index.astype(str).str.strip()

            # Ensure required columns exist
            for col in columns:
                if col not in model_df.columns:
                    model_df[col] = self._get_default_value(col, "Missing column")

            # Add defaults for missing indices
            missing_indices = set(base_df.index) - set(model_df.index)
            if missing_indices:
                logging.info(f"{len(missing_indices)} missing entries in {model_key}")
                defaults = pd.DataFrame(index=list(missing_indices))
                for col in columns:
                    defaults[col] = self._get_default_value(col, "No result")
                model_df = pd.concat([model_df, defaults])

            # Select only required columns and merge
            model_df = model_df[list(columns)]
            result = base_df.join(model_df, how='left')

            # Fill any remaining NaN values
            for col in columns:
                if col in result.columns:
                    result[col] = result[col].fillna(
                        self._get_default_value(col, "Missing value")
                    )

            return result

        except Exception as e:
            logging.error(f"Error merging {model_key}: {e}")
            for col in columns:
                base_df[col] = self._get_default_value(col, f"Error: {str(e)}")
            return base_df

    def _get_default_value(self, column: str, reason: str = "Not applicable"):
        """Get default value for a column based on its type."""
        if column.endswith("_Decision"):
            return False
        elif column.endswith("_Reason"):
            return f"Not applicable - {reason}"
        else:
            return "not applicable"

    def _compute_final_decision(self, row: pd.Series) -> bool:
        """
        Compute final decision based on model results.
        Priority: Model C > A&B Agreement > Model B > Model A > False
        """
        try:
            c_decision = row.get("C_Decision")
            a_decision = row.get("A_Decision")
            b_decision = row.get("B_Decision")

            # If Model C has a decision (and it's not just "no disagreement")
            if pd.notna(c_decision) and c_decision is not False:
                c_reason = str(row.get("C_Reason", ""))
                if "No disagreement" not in c_reason:
                    return bool(c_decision)

            # Check A and B agreement
            if pd.notna(a_decision) and pd.notna(b_decision):
                if bool(a_decision) == bool(b_decision):
                    return bool(a_decision)
                # On disagreement, prefer Model B
                return bool(b_decision)

            # Fallback to individual decisions
            if pd.notna(b_decision):
                return bool(b_decision)
            if pd.notna(a_decision):
                return bool(a_decision)

            return False

        except Exception as e:
            logging.error(f"Final decision computation error: {e}")
            return False

    def _organize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Organize DataFrame columns in proper order."""
        output_cols = [
            *BASE_COLUMNS,
            *MODEL_COLUMNS["model_a"],
            *MODEL_COLUMNS["model_b"],
            *MODEL_COLUMNS["model_c"],
            "Final_Decision"
        ]

        # Ensure all columns exist with defaults
        for col in output_cols:
            if col not in df.columns:
                df[col] = self._get_default_value(col, "Missing column")

        # Convert decision columns to boolean
        for col in df.columns:
            if col.endswith("_Decision"):
                df[col] = df[col].fillna(False).astype(bool)

        # Select existing columns in order
        existing_cols = [col for col in output_cols if col in df.columns]
        return df[existing_cols]

    def export_to_excel(self, df: pd.DataFrame, filename: str) -> None:
        """Export DataFrame to Excel file."""
        try:
            df.to_excel(filename, index=False)
            logging.info(f"Exported results to {filename}")
        except Exception as e:
            logging.error(f"Excel export error: {e}")
