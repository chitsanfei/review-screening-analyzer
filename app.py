import os
import logging
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv
from file_processor import FileProcessor
from analyzer import PICOSAnalyzer
from deduplicator import Deduplicator

# Load environment variables
load_dotenv()

# Load model configurations from environment variables
MODEL_CONFIGS = {
    "model_a": {
        "api_url": os.getenv("MODEL_A_API_URL", ""),
        "api_key": os.getenv("MODEL_A_API_KEY", ""),
        "model": os.getenv("MODEL_A_MODEL_NAME", ""),
        "temperature": 0.7,
        "max_tokens": 1024,
        "batch_size": 5,
        "threads": 4
    },
    "model_b": {
        "api_url": os.getenv("MODEL_B_API_URL", ""),
        "api_key": os.getenv("MODEL_B_API_KEY", ""),
        "model": os.getenv("MODEL_B_MODEL_NAME", ""),
        "temperature": 0.7,
        "max_tokens": 1024,
        "batch_size": 5,
        "threads": 4
    },
    "model_c": {
        "api_url": os.getenv("MODEL_C_API_URL", ""),
        "api_key": os.getenv("MODEL_C_API_KEY", ""),
        "model": os.getenv("MODEL_C_MODEL_NAME", ""),
        "temperature": 0.7,
        "max_tokens": 1024,
        "batch_size": 5,
        "threads": 4
    }
}

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Initialize components
analyzer = PICOSAnalyzer()
file_processor = FileProcessor(DATA_DIR)
model_results = {}
deduplicator = Deduplicator()

# Initialize model configurations
for model_key, config in MODEL_CONFIGS.items():
    analyzer.update_model_config(model_key, config)

# Ensure directories exist
for directory in [DATA_DIR, LOG_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {directory}: {str(e)}")

# Logging configuration
try:
    log_file = os.path.join(LOG_DIR, f"picos_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
except Exception as e:
    print(f"Failed to initialize logging: {str(e)}")
    raise

def create_gradio_interface():
    """Create Gradio interface"""
    def parse_nbib(file) -> tuple:
        """Parse NBIB or RIS file and return results"""
        try:
            if not file:
                return None, "No file uploaded"
            
            # Determine file type based on extension
            file_extension = os.path.splitext(file.name)[1].lower()
            
            if file_extension == '.nbib':
                # Parse NBIB file
                output_path, preview = file_processor.parse_nbib(file.name)
            elif file_extension == '.ris':
                # Parse RIS file
                output_path, preview = file_processor.parse_ris(file.name)
            else:
                return None, "Unsupported file format. Please upload a .nbib or .ris file"
            
            if not output_path:
                return None, "Failed to parse file"
            
            return output_path, preview
            
        except Exception as e:
            error_msg = f"Error parsing file: {str(e)}"
            logging.error(error_msg)
            return None, error_msg

    def update_picos_criteria(p, i, c, o, s):
        """Update PICOS criteria"""
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
            return f"❌ Error updating PICOS criteria: {str(e)}"
    
    def update_model_settings(model_key, api_url, api_key, model_name, temperature, max_tokens, batch_size, threads, prompt, is_inference):
        """Update model settings"""
        try:
            analyzer.update_model_config(model_key, {
                "api_url": api_url.strip(),
                "api_key": api_key.strip(),
                "model": model_name.strip(),
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "batch_size": int(batch_size),
                "threads": int(threads),
                "is_inference": bool(is_inference)
            })
            analyzer.update_prompt(model_key, prompt.strip())
            return "✓ Settings updated successfully"
        except Exception as e:
            return f"❌ Error updating settings: {str(e)}"
    
    def test_connection(model_key):
        """Test API connection with debug output"""
        try:
            result = analyzer.test_api_connection(model_key)
            
            # Debug: Test inference model response processing
            if analyzer.model_manager.get_config(model_key).get("is_inference", False):
                test_response = """
<think>
这是一个测试思考过程。
分析步骤如下：
1. 首先检查连接
2. 然后验证参数
</think>

这是实际的响应内容。
测试连接成功。

<html>
<body>
这些HTML标签应该被移除
</body>
</html>
"""
                processed_response = analyzer.model_manager.process_inference_response(test_response)
                return f"{result}\n\nDebug - Raw Response:\n{test_response}\n\nProcessed Response:\n{processed_response}"
            
            return result
        except Exception as e:
            return f"❌ Error testing connection: {str(e)}"
    
    def process_model(input_file, model_key, model_a_input=None, model_b_input=None):
        """Process analysis for a single model"""
        try:
            # Read CSV file using file processor
            df = file_processor.load_csv(input_file.name)
            if df is None:
                return None, "Failed to load CSV file"
            
            # Check and load previous model results
            if model_key == "model_b":
                if model_a_input is None or not os.path.exists(model_a_input.name):
                    return None, "Model A results file required for MODEL_B"
                model_results["model_a"] = file_processor.load_csv(model_a_input.name)
            elif model_key == "model_c":
                if model_a_input is None or not os.path.exists(model_a_input.name) or \
                   model_b_input is None or not os.path.exists(model_b_input.name):
                    return None, "Both Model A and B results files required for MODEL_C"
                model_results["model_a"] = file_processor.load_csv(model_a_input.name)
                model_results["model_b"] = file_processor.load_csv(model_b_input.name)
            
            # Start processing
            logging.info(f"Init Model: {model_key.upper()}...")
            results_df = analyzer.process_batch(df, model_key, model_results)
            model_results[model_key] = results_df
            
            # Save results
            output_path = file_processor.save_csv(results_df, f"{model_key}_results.csv")
            if not output_path:
                return None, f"Failed to save {model_key.upper()} results"
            
            completion_msg = f"{model_key.upper()} analysis completed: processed {len(df)} rows"
            logging.info(completion_msg)
            return output_path, completion_msg
        except Exception as e:
            error_msg = f"Error in {model_key.upper()}: {str(e)}"
            logging.error(error_msg)
            return None, error_msg
    
    def merge_results_with_files(input_file, model_a_file, model_b_file, model_c_file):
        """Merge all model results from files"""
        if not all([input_file, model_a_file, model_b_file]):
            return None, "Original file, Model A and B results are required"
        
        try:
            # Load all files using file processor
            df = file_processor.load_csv(input_file.name)
            model_a_results = file_processor.load_csv(model_a_file.name)
            model_b_results = file_processor.load_csv(model_b_file.name)
            model_c_results = file_processor.load_csv(model_c_file.name) if model_c_file else None
            
            if any(result is None for result in [df, model_a_results, model_b_results]):
                return None, "Failed to load one or more required files"
            
            # Store results in global variable
            model_results["model_a"] = model_a_results
            model_results["model_b"] = model_b_results
            if model_c_results is not None:
                model_results["model_c"] = model_c_results
            
            # Merge results
            merged_df = analyzer.merge_results(df, model_results)
            
            # Save results
            output_path = file_processor.save_csv(merged_df, "final_results.csv")
            if not output_path:
                return None, "Failed to save merged results"
                
            return output_path, "Results merged successfully"
        except Exception as e:
            return None, f"Error merging results: {str(e)}"
    
    def run_all_models(input_file):
        """Run analysis pipeline for all models"""
        try:
            # Read CSV file using file processor
            df = file_processor.load_csv(input_file.name)
            if df is None:
                yield None, None, None, None, "Failed to load input file"
                return
            
            total_steps = 4  # A, B, C, and merge
            current_step = 0
            
            def update_progress(step_name, status):
                nonlocal current_step
                current_step += 1
                progress = (current_step / total_steps) * 100
                return f"Step {current_step}/{total_steps} ({progress:.1f}%) - {step_name}: {status}"
            
            # Initialize output variables
            model_a_path = None
            model_b_path = None
            model_c_path = None
            final_path = None
            
            # Run Model A
            logging.info("Starting Model A analysis...")
            model_a_results = analyzer.process_batch(df, "model_a")
            if model_a_results is None:
                yield model_a_path, model_b_path, model_c_path, final_path, "Model A failed to process results"
                return
                
            # Save Model A results
            model_a_path = file_processor.save_csv(model_a_results, "model_a_results.csv")
            if not model_a_path:
                yield None, None, None, None, "Failed to save Model A results"
                return
                
            model_results["model_a"] = model_a_results
            
            status = update_progress("Model A", "Completed")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            # Run Model B
            logging.info("Starting Model B analysis...")
            model_b_results = analyzer.process_batch(df, "model_b", {"model_a": model_a_results})
            if model_b_results is None:
                yield model_a_path, model_b_path, model_c_path, final_path, "Model B failed to process results"
                return
                
            # Save Model B results
            model_b_path = file_processor.save_csv(model_b_results, "model_b_results.csv")
            if not model_b_path:
                yield model_a_path, None, None, None, "Failed to save Model B results"
                return
                
            model_results["model_b"] = model_b_results
            
            status = update_progress("Model B", "Completed")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            # Run Model C
            logging.info("Starting Model C analysis...")
            model_c_results = analyzer.process_batch(df, "model_c", {
                "model_a": model_a_results,
                "model_b": model_b_results
            })
            
            if model_c_results is not None:
                # Save Model C results
                model_c_path = file_processor.save_csv(model_c_results, "model_c_results.csv")
                if not model_c_path:
                    yield model_a_path, model_b_path, None, None, "Failed to save Model C results"
                    return
                    
                model_results["model_c"] = model_c_results
            
            status = update_progress("Model C", "Completed")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            # Merge results
            logging.info("Merging results...")
            merged_df = analyzer.merge_results(df, model_results)
            
            # Save final results
            final_path = file_processor.save_csv(merged_df, "final_results.csv")
            if not final_path:
                yield model_a_path, model_b_path, model_c_path, None, "Failed to save final results"
                return
            
            status = update_progress("Merge", "Completed")
            
            completion_msg = f"All models completed successfully - Processed {len(df)} rows"
            logging.info(completion_msg)
            yield model_a_path, model_b_path, model_c_path, final_path, completion_msg
            
        except Exception as e:
            error_msg = f"Error in pipeline: {str(e)}"
            logging.error(error_msg)
            yield model_a_path, model_b_path, model_c_path, final_path, error_msg

    def process_deduplication(files, threshold):
        """Process deduplication for multiple files"""
        try:
            if not files:
                return None, None, "No files uploaded"
            
            dataframes = []
            for file in files:
                if not file:
                    continue
                df = file_processor.load_csv(file.name)
                if df is None:
                    return None, None, f"Failed to load file: {file.name}"
                dataframes.append(df)
            
            if not dataframes:
                return None, None, "No valid files to process"
            
            # Process deduplication
            unique_df, clusters_df = deduplicator.process_dataframes(dataframes, threshold)
            
            # Save results
            unique_path = file_processor.save_csv(unique_df, "deduplicated_data.csv")
            clusters_path = file_processor.save_csv(clusters_df, "duplicate_clusters.csv")
            
            if not unique_path or not clusters_path:
                return None, None, "Failed to save results"
            
            status_msg = f"Deduplication completed successfully:\n"
            status_msg += f"Original entries: {sum(len(df) for df in dataframes)}\n"
            status_msg += f"Unique entries: {len(unique_df)}\n"
            status_msg += f"Duplicate clusters: {len(clusters_df['Cluster_ID'].unique()) if len(clusters_df) > 0 else 0}"
            
            return unique_path, clusters_path, status_msg
            
        except Exception as e:
            error_msg = f"Error in deduplication: {str(e)}"
            logging.error(error_msg)
            return None, None, error_msg

    # Create Gradio interface
    interface = gr.Blocks(title="PICOS Analysis System")

    with interface:
        gr.Markdown("# PICOS Literature Analysis System")
        gr.Markdown("This system uses a multi-model approach to analyze medical literature abstracts.")
        
        with gr.Tab("Citation File Processing"):
            with gr.Tab("Pubmed"):
                gr.Markdown("""
                ## Pubmed NBIB Processing
                Upload a .nbib file from Pubmed to extract and convert it to CSV format. The extracted data will include:
                - DOI
                - Title
                - Authors
                - Abstract
                """)
                
                with gr.Row():
                    nbib_file = gr.File(label="Upload NBIB File", file_types=[".nbib"])
                    process_nbib_btn = gr.Button("Process NBIB File")
                
                with gr.Row():
                    nbib_preview = gr.Textbox(label="Preview", lines=20)
                    nbib_output = gr.File(label="Download CSV")
                
                process_nbib_btn.click(
                    parse_nbib,
                    inputs=[nbib_file],
                    outputs=[nbib_output, nbib_preview]
                )
            
            with gr.Tab("Web of Science"):
                gr.Markdown("""
                ## Web of Science RIS Processing
                Upload a .ris file from Web of Science to extract and convert it to CSV format. The extracted data will include:
                - DOI
                - Title
                - Authors
                - Abstract
                """)
                
                with gr.Row():
                    ris_file = gr.File(label="Upload RIS File", file_types=[".ris"])
                    process_ris_btn = gr.Button("Process RIS File")
                
                with gr.Row():
                    ris_preview = gr.Textbox(label="Preview", lines=20)
                    ris_output = gr.File(label="Download CSV")
                
                process_ris_btn.click(
                    parse_nbib,  # 使用相同的处理函数，它会根据文件扩展名选择正确的解析器
                    inputs=[ris_file],
                    outputs=[ris_output, ris_preview]
                )

        with gr.Tab("Deduplication"):
            gr.Markdown("""
            ## Citation Deduplication
            Upload multiple CSV files to remove duplicate entries across different citation sources.
            The system will identify similar entries based on title and author information.
            
            ### Features:
            - Support for multiple CSV files
            - Adjustable similarity threshold
            - Detailed duplicate clusters report
            - Standardized output format
            """)
            
            with gr.Row():
                input_files = gr.File(
                    label="Upload CSV Files", 
                    file_types=[".csv"], 
                    file_count="multiple"
                )
                threshold = gr.Slider(
                    label="Similarity Threshold",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.8,
                    step=0.05,
                    info="Higher values mean stricter matching (0.8 recommended)"
                )
            
            with gr.Row():
                process_btn = gr.Button("Process Deduplication")
            
            with gr.Row():
                status = gr.Textbox(label="Status", lines=5)
            
            with gr.Row():
                unique_output = gr.File(label="Download Deduplicated Data")
                clusters_output = gr.File(label="Download Duplicate Clusters")
            
            process_btn.click(
                process_deduplication,
                inputs=[input_files, threshold],
                outputs=[unique_output, clusters_output, status]
            )

        with gr.Tab("LLM Analysis"):
            with gr.Tab("PICOS Criteria"):
                gr.Markdown("""
                ## PICOS Criteria Settings
                Define the standard PICOS criteria that will be used by all models.
                These criteria will be used to evaluate whether each article meets the requirements.
                """)
                
                with gr.Group("Standard PICOS Criteria"):
                    population = gr.Textbox(label="Population", value=analyzer.picos_criteria["population"],
                                          placeholder="e.g., patients with hepatocellular carcinoma")
                    intervention = gr.Textbox(label="Intervention", value=analyzer.picos_criteria["intervention"],
                                            placeholder="e.g., immunotherapy or targeted therapy")
                    comparison = gr.Textbox(label="Comparison", value=analyzer.picos_criteria["comparison"],
                                          placeholder="e.g., standard therapy or placebo")
                    outcome = gr.Textbox(label="Outcome", value=analyzer.picos_criteria["outcome"],
                                       placeholder="e.g., survival or response rate")
                    study_design = gr.Textbox(label="Study Design", value=analyzer.picos_criteria["study_design"],
                                            placeholder="e.g., randomized controlled trial")
                    
                    update_picos_btn = gr.Button("Update PICOS Criteria")
                    picos_status = gr.Textbox(label="Status")
                    
                    update_picos_btn.click(
                        update_picos_criteria,
                        inputs=[population, intervention, comparison, outcome, study_design],
                        outputs=picos_status
                    )
            
            with gr.Tab("Model Settings"):
                for model_key in ["model_a", "model_b", "model_c"]:
                    with gr.Group(f"{model_key.upper()} Settings"):
                        config = analyzer.model_manager.get_config(model_key)
                        api_url = gr.Textbox(label="API URL", value=config["api_url"])
                        api_key = gr.Textbox(label="API Key", value=config["api_key"])
                        model_name = gr.Textbox(label="Model", value=config["model"])
                        is_inference = gr.Checkbox(
                            label="Inference Model",
                            value=config.get("is_inference", False),
                            info="Enable inference compatibility mode for models that return reasoning process"
                        )
                        temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, value=config["temperature"])
                        max_tokens = gr.Number(label="Max Tokens", value=config["max_tokens"])
                        batch_size = gr.Number(label="Batch Size", value=config["batch_size"])
                        threads = gr.Slider(label="Threads", minimum=1, maximum=32, step=1, value=config["threads"])
                        prompt = gr.Textbox(label="Prompt Template", value=analyzer.prompt_manager.get_prompt(model_key), lines=10)
                        
                        update_btn = gr.Button(f"Update {model_key.upper().replace('_', ' ')} Settings")
                        test_btn = gr.Button(f"Test {model_key.upper().replace('_', ' ')} Connection")
                        status = gr.Textbox(label="Status", lines=10)  # Increased lines for debug output
                        
                        update_btn.click(
                            update_model_settings,
                            inputs=[
                                gr.Textbox(value=model_key, visible=False),
                                api_url,
                                api_key,
                                model_name,
                                temperature,
                                max_tokens,
                                batch_size,
                                threads,
                                prompt,
                                is_inference
                            ],
                            outputs=status
                        )
                        test_btn.click(
                            test_connection,
                            inputs=[gr.Textbox(value=model_key, visible=False)],
                            outputs=status
                        )
            
            with gr.Tab("Analysis"):
                with gr.Row():
                    input_file = gr.File(label="Original CSV File")
                    model_a_input = gr.File(label="Model A Results")
                    model_b_input = gr.File(label="Model B Results")
                    model_c_input = gr.File(label="Model C Results")
                
                with gr.Row():
                    model_a_btn = gr.Button("Run Model A")
                    model_b_btn = gr.Button("Run Model B")
                    model_c_btn = gr.Button("Run Model C")
                    merge_btn = gr.Button("Merge Results")
                    run_all_btn = gr.Button("Run All", variant="primary")
                
                status = gr.Textbox(label="Status")
                
                with gr.Row():
                    model_a_output = gr.File(label="Model A Results")
                    model_b_output = gr.File(label="Model B Results")
                    model_c_output = gr.File(label="Model C Results")
                    final_output = gr.File(label="Final Results")
                
                model_a_btn.click(
                    lambda x: process_model(x, "model_a"),
                    inputs=[input_file],
                    outputs=[model_a_output, status]
                )
                model_b_btn.click(
                    lambda x, y: process_model(x, "model_b", y),
                    inputs=[input_file, model_a_input],
                    outputs=[model_b_output, status]
                )
                model_c_btn.click(
                    lambda x, y, z: process_model(x, "model_c", y, z),
                    inputs=[input_file, model_a_input, model_b_input],
                    outputs=[model_c_output, status]
                )
                merge_btn.click(
                    merge_results_with_files,
                    inputs=[input_file, model_a_input, model_b_input, model_c_input],
                    outputs=[final_output, status]
                )
                run_all_btn.click(
                    run_all_models,
                    inputs=[input_file],
                    outputs=[model_a_output, model_b_output, model_c_output, final_output, status]
                )
        
        gr.Markdown("""
        ## Instructions
        1. Start by processing your citation file in the "Citation File Processing" tab
        2. Configure PICOS criteria and model settings in the "LLM Analysis" tab
        3. Test API connections before running analysis
        4. Upload the generated CSV file in the "Analysis" tab
        5. Run models in sequence: A -> B -> C
        6. Merge results to get the final analysis
        
        ## Input File Format
        The input CSV file should contain at least the following columns:
        - Index: Unique identifier for each abstract
        - Abstract: The text content to be analyzed
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    if interface:
        interface.launch(server_name="0.0.0.0", server_port=7860)
    else:
        print("Error: Failed to create Gradio interface") 