"""
PICOS Literature Analysis System - Main Application
A modern Gradio-based web interface for medical literature screening.
"""

import os
import time
import logging
from datetime import datetime
from functools import lru_cache
from typing import Optional, Tuple, List, Generator

import gradio as gr
from dotenv import load_dotenv

from file_processor import FileProcessor
from analyzer import PICOSAnalyzer
from deduplicator import Deduplicator
from result_processor import ResultProcessor

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Load environment variables
dotenv_path = os.path.join(BASE_DIR, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

# Create required directories
for directory in [DATA_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
log_file = os.path.join(LOG_DIR, f"picos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Initialize components (singleton pattern)
analyzer = PICOSAnalyzer()
file_processor = FileProcessor(DATA_DIR)
deduplicator = Deduplicator()
result_processor = ResultProcessor()
model_results = {}

# Custom CSS for modern design with dark mode support
CUSTOM_CSS = """
/* Modern color scheme - Light mode */
:root {
    --primary-color: #2563eb;
    --primary-hover: #1d4ed8;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --card-bg: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    --card-border: #e2e8f0;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --workflow-bg: #f8fafc;
    --info-box-bg: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    --table-row-alt: #f8fafc;
    --table-border: #e2e8f0;
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    :root {
        --card-bg: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        --card-border: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --workflow-bg: #1e293b;
        --info-box-bg: linear-gradient(135deg, #1e3a5f 0%, #172554 100%);
        --table-row-alt: #1e293b;
        --table-border: #334155;
    }
}

/* Gradio dark mode class support */
.dark {
    --card-bg: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    --card-border: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --workflow-bg: #1e293b;
    --info-box-bg: linear-gradient(135deg, #1e3a5f 0%, #172554 100%);
    --table-row-alt: #1e293b;
    --table-border: #334155;
}

/* Header styling */
.header-container {
    background: var(--bg-gradient);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.header-container h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.header-container p {
    font-size: 1.1rem;
    opacity: 0.95;
    margin-top: 0.5rem;
}

/* Feature cards */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.feature-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    border-color: var(--primary-color);
}

.feature-icon {
    width: 48px;
    height: 48px;
    background: var(--bg-gradient);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 1rem;
}

.feature-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.feature-desc {
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* Workflow steps */
.workflow-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 2rem 0;
    padding: 1.5rem;
    background: var(--workflow-bg);
    border-radius: 12px;
    border: 1px solid var(--card-border);
}

.workflow-step {
    background: var(--primary-color);
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 25px;
    font-weight: 500;
    font-size: 0.9rem;
}

.workflow-arrow {
    color: var(--primary-color);
    font-size: 1.5rem;
    font-weight: bold;
}

/* Status badges */
.status-success { color: var(--success-color); }
.status-warning { color: var(--warning-color); }
.status-error { color: var(--error-color); }

/* Modern buttons */
.primary-btn {
    background: var(--bg-gradient) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important;
}

/* Tab styling */
.tab-nav button {
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected {
    background: var(--primary-color) !important;
    color: white !important;
}

/* Info boxes */
.info-box {
    background: var(--info-box-bg);
    border-left: 4px solid var(--primary-color);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    color: var(--text-primary);
}

/* Format table */
.format-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9rem;
}

.format-table th {
    background: var(--primary-color);
    color: white;
    padding: 0.75rem 1rem;
    text-align: left;
}

.format-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--table-border);
    color: var(--text-primary);
}

.format-table tr:nth-child(even) {
    background: var(--table-row-alt);
}

/* Section headings */
h3 {
    color: var(--text-primary) !important;
}

/* Accordion styling */
.accordion {
    border: 1px solid var(--card-border);
    border-radius: 8px;
    margin: 0.5rem 0;
    overflow: hidden;
}

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: var(--text-secondary);
    font-size: 0.85rem;
    border-top: 1px solid var(--card-border);
    margin-top: 2rem;
}
"""

# HTML content for the introduction page
INTRO_HTML = """
<div class="header-container">
    <h1>PICOS Literature Analysis System</h1>
    <p>AI-Powered Multi-Model Medical Literature Screening for Systematic Reviews</p>
</div>

<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">📚</div>
        <div class="feature-title">Multi-Source Support</div>
        <div class="feature-desc">Import citations from PubMed, Embase, Web of Science, and Scopus in their native formats.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">Smart Deduplication</div>
        <div class="feature-desc">Advanced TF-IDF and cosine similarity algorithms identify and remove duplicate entries across sources.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">Three-Model Consensus</div>
        <div class="feature-desc">Primary analyzer, critical reviewer, and final arbitrator ensure accurate PICOS classification.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Comprehensive Reports</div>
        <div class="feature-desc">Export detailed Excel reports with PICOS criteria matching, decisions, and reasoning for each article.</div>
    </div>
</div>

<div class="workflow-container">
    <span class="workflow-step">1. Import Citations</span>
    <span class="workflow-arrow">→</span>
    <span class="workflow-step">2. Deduplicate</span>
    <span class="workflow-arrow">→</span>
    <span class="workflow-step">3. Configure PICOS</span>
    <span class="workflow-arrow">→</span>
    <span class="workflow-step">4. Run Analysis</span>
    <span class="workflow-arrow">→</span>
    <span class="workflow-step">5. Export Results</span>
</div>

<div class="info-box">
    <strong>About This Project</strong><br>
    This project is a demo implementation of the paper <em>"Automated Literature Screening for Hepatocellular Carcinoma Treatment Through Integration of 3 Large Language Models: Methodological Study"</em> published in <strong>JMIR Medical Informatics</strong> (doi: <a href="https://medinform.jmir.org/2025/1/e76252" target="_blank" rel="noopener noreferrer">10.2196/76252</a>).
    <br><br>
    It is for learning purposes only. If it is helpful to you, please cite our article, thank you!
</div>

<h3 style="margin-top: 2rem; color: #1e293b;">Supported File Formats</h3>
<table class="format-table">
    <tr><th>Source</th><th>Format</th><th>Extension</th></tr>
    <tr><td>PubMed</td><td>NBIB</td><td>.nbib</td></tr>
    <tr><td>Embase</td><td>RIS</td><td>.ris</td></tr>
    <tr><td>Web of Science</td><td>RIS</td><td>.ris</td></tr>
    <tr><td>Scopus</td><td>RIS</td><td>.ris</td></tr>
</table>

<h3 style="margin-top: 2rem; color: #1e293b;">Analysis Output</h3>
<table class="format-table">
    <tr><th>Column</th><th>Description</th></tr>
    <tr><td>A_P, A_I, A_C, A_O, A_S</td><td>Model A's PICOS extraction</td></tr>
    <tr><td>A_Decision / A_Reason</td><td>Model A's inclusion decision and reasoning</td></tr>
    <tr><td>B_P, B_I, B_C, B_O, B_S</td><td>Model B's corrections (or "-" if agrees)</td></tr>
    <tr><td>B_Decision / B_Reason</td><td>Model B's critical review decision</td></tr>
    <tr><td>C_Decision / C_Reason</td><td>Model C's final arbitration (when A/B disagree)</td></tr>
    <tr><td>Final_Decision</td><td>Computed final inclusion decision</td></tr>
</table>

<div class="footer">
    PICOS Literature Analysis System | Built with Gradio
</div>
"""


def parse_citation_file(file, file_type: str) -> Tuple[Optional[str], str]:
    """Parse citation file based on its type."""
    if not file:
        return None, "No file uploaded"

    try:
        file_path = file.name
        ext = os.path.splitext(file_path)[1].lower()

        parsers = {
            'pubmed': lambda: file_processor.parse_nbib(file_path) if ext == '.nbib' else (None, "Expected .nbib file"),
            'embase': lambda: file_processor.parse_embase_ris(file_path) if ext == '.ris' else (None, "Expected .ris file"),
            'wos': lambda: file_processor.parse_wos_ris(file_path) if ext == '.ris' else (None, "Expected .ris file"),
            'scopus': lambda: file_processor.parse_scopus_ris(file_path) if ext == '.ris' else (None, "Expected .ris file"),
        }

        if file_type in parsers:
            return parsers[file_type]()

        # Auto-detect for generic parsing
        if ext == '.nbib':
            return file_processor.parse_nbib(file_path)
        elif ext == '.ris':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(2000)  # Read first 2000 chars for detection
            if 'T1  - ' in content:
                return file_processor.parse_embase_ris(file_path)
            return file_processor.parse_wos_ris(file_path)

        return None, "Unsupported file format"

    except Exception as e:
        logging.error(f"Error parsing file: {e}")
        return None, f"Error: {str(e)}"


def update_picos_criteria(p: str, i: str, c: str, o: str, s: str) -> str:
    """Update PICOS criteria for analysis."""
    try:
        analyzer.update_picos_criteria({
            "population": p.strip(),
            "intervention": i.strip(),
            "comparison": c.strip(),
            "outcome": o.strip(),
            "study_design": s.strip()
        })
        return "✓ PICOS criteria updated successfully"
    except Exception as e:
        return f"✗ Error: {str(e)}"


def update_model_settings(model_key: str, api_url: str, api_key: str, model_name: str,
                          temperature: float, max_tokens: int, batch_size: int,
                          threads: int, prompt: str, is_inference: bool, timeout: float) -> str:
    """Update settings for a specific model."""
    try:
        analyzer.update_model_config(model_key, {
            "api_url": api_url.strip(),
            "api_key": api_key.strip(),
            "model": model_name.strip(),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "batch_size": int(batch_size),
            "threads": int(threads),
            "is_inference": bool(is_inference),
            "timeout": float(timeout),
            "updated": True
        })
        analyzer.update_prompt(model_key, prompt.strip())
        return "✓ Settings updated successfully"
    except Exception as e:
        return f"✗ Error: {str(e)}"


def test_connection(model_key: str) -> str:
    """Test API connection for a model."""
    return analyzer.test_api_connection(model_key)


def process_single_model(input_file, model_key: str, model_a_input=None, model_b_input=None):
    """Process analysis for a single model."""
    if not input_file:
        return None, "No input file provided"

    try:
        df = file_processor.load_excel(input_file.name)
        if df is None:
            return None, "Failed to load input file"

        # Validate dependencies
        prev_results = {}
        if model_key == "model_b":
            if not model_a_input:
                return None, "Model A results required for Model B"
            prev_results["model_a"] = file_processor.load_excel(model_a_input.name)
        elif model_key == "model_c":
            if not model_a_input or not model_b_input:
                return None, "Model A and B results required for Model C"
            prev_results["model_a"] = file_processor.load_excel(model_a_input.name)
            prev_results["model_b"] = file_processor.load_excel(model_b_input.name)

        start_time = time.time()
        results_df = analyzer.process_batch(df, model_key, prev_results if prev_results else None)

        if results_df is None:
            return None, f"{model_key.upper()} analysis failed"

        # Save results
        output_file = os.path.join(DATA_DIR, f"{model_key}_results.xlsx")

        if model_key == "model_c":
            prev_results["model_c"] = results_df
            merged_df = analyzer.merge_results(df, prev_results)
            file_processor.save_excel(merged_df, output_file)
        else:
            file_processor.save_excel(results_df, output_file)

        elapsed = time.time() - start_time
        return gr.update(value=output_file), f"✓ {model_key.upper()} completed in {elapsed:.1f}s"

    except Exception as e:
        logging.error(f"Error in {model_key}: {e}")
        return None, f"✗ Error: {str(e)}"


def run_full_pipeline(input_file) -> Generator:
    """Run complete analysis pipeline with streaming updates."""
    if not input_file:
        yield [None, None, None, None, "No input file provided"]
        return

    try:
        df = file_processor.load_excel(input_file.name)
        if df is None:
            yield [None, None, None, None, "Failed to load input file"]
            return

        results = {}

        # Model A
        logging.info("Starting Model A analysis...")
        results["model_a"] = analyzer.process_batch(df, "model_a")
        if results["model_a"] is None:
            yield [None, None, None, None, "Model A failed"]
            return

        model_a_path = os.path.join(DATA_DIR, "model_a_results.xlsx")
        file_processor.save_excel(results["model_a"], model_a_path)
        yield [gr.update(value=model_a_path), None, None, None, "Model A completed"]

        # Model B
        logging.info("Starting Model B analysis...")
        results["model_b"] = analyzer.process_batch(df, "model_b", {"model_a": results["model_a"]})
        if results["model_b"] is None:
            yield [gr.update(value=model_a_path), None, None, None, "Model B failed"]
            return

        model_b_path = os.path.join(DATA_DIR, "model_b_results.xlsx")
        file_processor.save_excel(results["model_b"], model_b_path)
        yield [gr.update(value=model_a_path), gr.update(value=model_b_path), None, None, "Model B completed"]

        # Model C
        logging.info("Starting Model C analysis...")
        results["model_c"] = analyzer.process_batch(df, "model_c", {
            "model_a": results["model_a"],
            "model_b": results["model_b"]
        })

        model_c_path = None
        if results["model_c"] is not None:
            model_c_path = os.path.join(DATA_DIR, "model_c_results.xlsx")
            file_processor.save_excel(results["model_c"], model_c_path)

        yield [
            gr.update(value=model_a_path),
            gr.update(value=model_b_path),
            gr.update(value=model_c_path) if model_c_path else None,
            None,
            "Model C completed"
        ]

        # Merge results
        logging.info("Merging results...")
        merged_df = analyzer.merge_results(df, results)
        final_path = os.path.join(DATA_DIR, "final_results.xlsx")
        file_processor.save_excel(merged_df, final_path)

        yield [
            gr.update(value=model_a_path),
            gr.update(value=model_b_path),
            gr.update(value=model_c_path) if model_c_path else None,
            gr.update(value=final_path),
            "✓ All models completed successfully"
        ]

    except Exception as e:
        logging.error(f"Pipeline error: {e}")
        yield [None, None, None, None, f"✗ Error: {str(e)}"]


def merge_all_results(input_file, model_a_file, model_b_file, model_c_file):
    """Merge results from all model files."""
    if not all([input_file, model_a_file, model_b_file]):
        return None, "Original file and Model A/B results required"

    try:
        df = file_processor.load_excel(input_file.name)
        results = {
            "model_a": file_processor.load_excel(model_a_file.name),
            "model_b": file_processor.load_excel(model_b_file.name),
        }

        if model_c_file:
            results["model_c"] = file_processor.load_excel(model_c_file.name)

        if any(v is None for v in [df, results["model_a"], results["model_b"]]):
            return None, "Failed to load one or more files"

        merged_df = analyzer.merge_results(df, results)
        output_path = os.path.join(DATA_DIR, "final_results.xlsx")
        result_processor.export_to_excel(merged_df, output_path)

        return output_path, "✓ Results merged successfully"

    except Exception as e:
        return None, f"✗ Error: {str(e)}"


def process_deduplication(files: List, threshold: float):
    """Process deduplication for uploaded files."""
    if not files:
        return None, None, "No files uploaded"

    try:
        dataframes = []
        for file in files:
            if file:
                df = file_processor.load_excel(file.name)
                if df is not None:
                    dataframes.append(df)

        if not dataframes:
            return None, None, "No valid files to process"

        unique_df, clusters_df = deduplicator.process_dataframes(dataframes, threshold)

        unique_path = file_processor.save_excel(unique_df, "deduplicated_data.xlsx")
        clusters_path = file_processor.save_excel(clusters_df, "duplicate_clusters.xlsx")

        total_original = sum(len(df) for df in dataframes)
        num_clusters = len(clusters_df['Cluster_ID'].unique()) if len(clusters_df) > 0 else 0

        status = f"""✓ Deduplication completed
• Original entries: {total_original}
• Unique entries: {len(unique_df)}
• Duplicate clusters: {num_clusters}
• Duplicates removed: {total_original - len(unique_df)}"""

        return unique_path, clusters_path, status

    except Exception as e:
        logging.error(f"Deduplication error: {e}")
        return None, None, f"✗ Error: {str(e)}"


def create_model_settings_group(model_key: str):
    """Create settings UI for a model."""
    config = analyzer.model_manager.get_config(model_key)
    model_name_display = model_key.replace("_", " ").upper()

    with gr.Accordion(f"{model_name_display} Settings", open=False):
        with gr.Row():
            with gr.Column(scale=2):
                api_url = gr.Textbox(label="API URL", value=config["api_url"])
                api_key = gr.Textbox(label="API Key", value=config["api_key"], type="password")
                model_name = gr.Textbox(label="Model Name", value=config["model"])
            with gr.Column(scale=1):
                temperature = gr.Slider(
                    label="Temperature", minimum=0, maximum=2,
                    value=config["temperature"], step=0.1
                )
                max_tokens = gr.Number(label="Max Tokens", value=config["max_tokens"])
                batch_size = gr.Number(label="Batch Size", value=config["batch_size"])
                threads = gr.Slider(
                    label="Threads", minimum=1, maximum=32,
                    value=config["threads"], step=1
                )
                timeout = gr.Number(label="Timeout (s)", value=config.get("timeout", 180))
                is_inference = gr.Checkbox(
                    label="Inference Mode",
                    value=config.get("is_inference", False),
                    info="Enable for models with reasoning tags"
                )

        prompt = gr.Textbox(
            label="Prompt Template",
            value=analyzer.prompt_manager.get_prompt(model_key),
            lines=8
        )

        with gr.Row():
            update_btn = gr.Button(f"Update {model_name_display}", variant="secondary")
            test_btn = gr.Button(f"Test Connection", variant="secondary")

        status = gr.Textbox(label="Status", interactive=False)

        # Hidden textbox for model key
        model_key_box = gr.Textbox(value=model_key, visible=False)

        update_btn.click(
            update_model_settings,
            inputs=[model_key_box, api_url, api_key, model_name, temperature,
                    max_tokens, batch_size, threads, prompt, is_inference, timeout],
            outputs=status
        )
        test_btn.click(
            test_connection,
            inputs=[model_key_box],
            outputs=status
        )


def create_gradio_interface():
    """Create the main Gradio interface."""

    with gr.Blocks(
        title="PICOS Analysis System",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
        )
    ) as interface:

        # Instructions Tab
        with gr.Tab("🏠 Overview"):
            gr.HTML(INTRO_HTML)

        # Citation Processing Tab
        with gr.Tab("📁 Citation Processing"):
            gr.Markdown("### Import and Convert Citation Files")
            gr.Markdown("Upload citation files from different databases to convert them into a standardized Excel format for analysis.")

            with gr.Tabs():
                for source, file_type, ext, desc in [
                    ("PubMed", "pubmed", ".nbib", "NBIB format from PubMed database"),
                    ("Embase", "embase", ".ris", "RIS format from Embase database"),
                    ("Web of Science", "wos", ".ris", "RIS format from Web of Science"),
                    ("Scopus", "scopus", ".ris", "RIS format from Scopus database"),
                ]:
                    with gr.Tab(source):
                        gr.Markdown(f"**{desc}**")
                        with gr.Row():
                            file_input = gr.File(label=f"Upload {ext} File", file_types=[ext])
                            process_btn = gr.Button(f"Process {source} File", variant="primary")

                        with gr.Row():
                            preview = gr.Textbox(label="Preview", lines=15, interactive=False)
                            output_file = gr.File(label="Download Excel")

                        process_btn.click(
                            lambda f, ft=file_type: parse_citation_file(f, ft),
                            inputs=[file_input],
                            outputs=[output_file, preview]
                        )

        # Deduplication Tab
        with gr.Tab("🔄 Deduplication"):
            gr.Markdown("### Remove Duplicate Citations")
            gr.Markdown("Upload multiple Excel files to identify and remove duplicate entries across different citation sources using TF-IDF similarity matching.")

            with gr.Row():
                with gr.Column(scale=2):
                    dedup_files = gr.File(
                        label="Upload Excel Files",
                        file_types=[".xlsx", ".xls"],
                        file_count="multiple"
                    )
                with gr.Column(scale=1):
                    threshold = gr.Slider(
                        label="Similarity Threshold",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                        info="Higher = stricter matching (0.8 recommended)"
                    )
                    dedup_btn = gr.Button("Run Deduplication", variant="primary")

            dedup_status = gr.Textbox(label="Status", lines=5, interactive=False)

            with gr.Row():
                unique_output = gr.File(label="Deduplicated Data")
                clusters_output = gr.File(label="Duplicate Clusters")

            dedup_btn.click(
                process_deduplication,
                inputs=[dedup_files, threshold],
                outputs=[unique_output, clusters_output, dedup_status]
            )

        # LLM Analysis Tab
        with gr.Tab("🤖 LLM Analysis"):
            with gr.Tabs():
                # PICOS Criteria Tab
                with gr.Tab("PICOS Criteria"):
                    gr.Markdown("### Define PICOS Criteria")
                    gr.Markdown("Set the criteria that all three models will use to evaluate literature abstracts.")

                    with gr.Group():
                        population = gr.Textbox(
                            label="Population (P)",
                            value=analyzer.picos_criteria["population"],
                            placeholder="e.g., patients with hepatocellular carcinoma"
                        )
                        intervention = gr.Textbox(
                            label="Intervention (I)",
                            value=analyzer.picos_criteria["intervention"],
                            placeholder="e.g., immunotherapy or targeted therapy"
                        )
                        comparison = gr.Textbox(
                            label="Comparison (C)",
                            value=analyzer.picos_criteria["comparison"],
                            placeholder="e.g., standard therapy or placebo"
                        )
                        outcome = gr.Textbox(
                            label="Outcome (O)",
                            value=analyzer.picos_criteria["outcome"],
                            placeholder="e.g., survival rate or tumor response"
                        )
                        study_design = gr.Textbox(
                            label="Study Design (S)",
                            value=analyzer.picos_criteria["study_design"],
                            placeholder="e.g., randomized controlled trial"
                        )

                    picos_btn = gr.Button("Update PICOS Criteria", variant="primary")
                    picos_status = gr.Textbox(label="Status", interactive=False)

                    picos_btn.click(
                        update_picos_criteria,
                        inputs=[population, intervention, comparison, outcome, study_design],
                        outputs=picos_status
                    )

                # Model Settings Tab
                with gr.Tab("Model Settings"):
                    gr.Markdown("### Configure LLM Models")
                    gr.Markdown("Set up API endpoints and parameters for each analysis model.")

                    for model_key in ["model_a", "model_b", "model_c"]:
                        create_model_settings_group(model_key)

                # Analysis Tab
                with gr.Tab("Run Analysis"):
                    gr.Markdown("### Execute PICOS Analysis")
                    gr.Markdown("Upload your data and run the multi-model analysis pipeline.")

                    with gr.Row():
                        input_file = gr.File(label="Input Excel File")
                        model_a_input = gr.File(label="Model A Results (for B/C)")
                        model_b_input = gr.File(label="Model B Results (for C)")
                        model_c_input = gr.File(label="Model C Results (for merge)")

                    with gr.Row():
                        run_all_btn = gr.Button("▶ Run Full Pipeline", variant="primary", scale=2)
                        model_a_btn = gr.Button("Run Model A", variant="secondary")
                        model_b_btn = gr.Button("Run Model B", variant="secondary")
                        model_c_btn = gr.Button("Run Model C", variant="secondary")
                        merge_btn = gr.Button("Merge Results", variant="secondary")

                    analysis_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Row():
                        model_a_output = gr.File(label="Model A Results")
                        model_b_output = gr.File(label="Model B Results")
                        model_c_output = gr.File(label="Model C Results")
                        final_output = gr.File(label="Final Merged Results")

                    # Event handlers
                    run_all_btn.click(
                        run_full_pipeline,
                        inputs=[input_file],
                        outputs=[model_a_output, model_b_output, model_c_output, final_output, analysis_status]
                    )

                    model_a_btn.click(
                        lambda f: process_single_model(f, "model_a"),
                        inputs=[input_file],
                        outputs=[model_a_output, analysis_status]
                    )

                    model_b_btn.click(
                        lambda f, a: process_single_model(f, "model_b", a),
                        inputs=[input_file, model_a_input],
                        outputs=[model_b_output, analysis_status]
                    )

                    model_c_btn.click(
                        lambda f, a, b: process_single_model(f, "model_c", a, b),
                        inputs=[input_file, model_a_input, model_b_input],
                        outputs=[model_c_output, analysis_status]
                    )

                    merge_btn.click(
                        merge_all_results,
                        inputs=[input_file, model_a_input, model_b_input, model_c_input],
                        outputs=[final_output, analysis_status]
                    )

    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )
