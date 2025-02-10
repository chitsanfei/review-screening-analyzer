import os
import logging
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv
from file_processor import FileProcessor
from analyzer import PICOSAnalyzer
from deduplicator import Deduplicator
from result_processor import ResultProcessor  # 新增：导入 ResultProcessor

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Initialize components
analyzer = PICOSAnalyzer()
file_processor = FileProcessor(DATA_DIR)
model_results = {}
deduplicator = Deduplicator()
result_processor = ResultProcessor()  # 实例化结果处理器

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
        """Parse citation file and return results"""
        try:
            if not file:
                return None, "No file uploaded"
            
            # Determine file type based on extension
            file_extension = os.path.splitext(file.name)[1].lower()
            
            if file_extension == '.nbib':
                # Parse NBIB file
                output_path, preview = file_processor.parse_nbib(file.name)
            elif file_extension == '.ris':
                # Check if it's Embase RIS or WoS RIS based on content
                with open(file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'T1  - ' in content:  # Embase RIS format
                    output_path, preview = file_processor.parse_embase_ris(file.name)
                else:  # WoS RIS format
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
    
    def update_model_settings(model_key, api_url, api_key, model_name, temperature, max_tokens, batch_size, threads, prompt, is_inference, timeout):
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
                "is_inference": bool(is_inference),
                "timeout": float(timeout)
            })
            analyzer.update_prompt(model_key, prompt.strip())
            return "✓ Settings updated successfully"
        except Exception as e:
            return f"❌ Error updating settings: {str(e)}"
    
    def test_connection(model_key):
        """Test API connection"""
        try:
            result = analyzer.test_api_connection(model_key)
            return result
        except Exception as e:
            return f"❌ Error testing connection: {str(e)}"
    
    def process_model(input_file, model_key, model_a_input=None, model_b_input=None):
        """Process analysis for a single model"""
        try:
            # Read CSV file using file processor
            logging.info(f"Loading input file for {model_key.upper()}...")
            df = file_processor.load_csv(input_file.name)
            if df is None:
                return None, "Failed to load CSV file"
            
            # Check and load previous model results
            if model_key == "model_b":
                logging.info("Loading Model A results for Model B analysis...")
                if model_a_input is None or not os.path.exists(model_a_input.name):
                    return None, "Model A results file required for MODEL_B"
                model_results["model_a"] = file_processor.load_csv(model_a_input.name)
            elif model_key == "model_c":
                logging.info("Loading Model A and B results for Model C analysis...")
                if model_a_input is None or not os.path.exists(model_a_input.name) or \
                   model_b_input is None or not os.path.exists(model_b_input.name):
                    return None, "Both Model A and B results files required for MODEL_C"
                model_results["model_a"] = file_processor.load_csv(model_a_input.name)
                model_results["model_b"] = file_processor.load_csv(model_b_input.name)
            
            # Start processing
            logging.info(f"Starting {model_key.upper()} analysis...")
            total_rows = len(df)
            processed_rows = 0
            errors = 0
            
            # 获取当前模型的批次大小
            batch_size = analyzer.model_manager.get_config(model_key).get("batch_size", 10)
            
            def progress_callback(row_index, error=False):
                nonlocal processed_rows, errors
                processed_rows += 1
                if error:
                    errors += 1
                # 根据批次大小或处理完成时显示进度
                if processed_rows % batch_size == 0 or processed_rows == total_rows:
                    progress = (processed_rows / total_rows) * 100
                    logging.info(f"Processing {model_key.upper()}: {processed_rows}/{total_rows} rows ({progress:.1f}%) - Errors: {errors}")
            
            # 根据是否有 model_results 来调用 process_batch
            if model_results:
                results_df = analyzer.process_batch(df, model_key, model_results, progress_callback)
            else:
                results_df = analyzer.process_batch(df, model_key, None, progress_callback)
            
            if results_df is None:
                return None, f"{model_key.upper()} failed to process results"
            
            model_results[model_key] = results_df
            
            # Save model-specific results as CSV (保留其它中间结果为 CSV)
            logging.info(f"Saving {model_key.upper()} results...")
            output_path = file_processor.save_csv(results_df, f"{model_key}_results.csv")
            if not output_path:
                return None, f"Failed to save {model_key.upper()} results"
            
            completion_msg = f"{model_key.upper()} analysis completed: processed {total_rows} rows"
            logging.info(completion_msg)
            return output_path, completion_msg
        except Exception as e:
            error_msg = f"Error in {model_key.upper()}: {str(e)}"
            logging.error(error_msg)
            return None, error_msg
    
    def merge_results_with_files(input_file, model_a_file, model_b_file, model_c_file):
        """Merge all model results from files and export as XLSX"""
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
            
            # Export merged results as XLSX instead of CSV
            final_filename = "final_results.xlsx"
            result_processor.export_to_excel(merged_df, final_filename)
            
            return final_filename, "Results merged successfully"
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
            status = update_progress("Model A", "Processing...")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            model_a_results = analyzer.process_batch(df, "model_a")
            if model_a_results is None:
                yield model_a_path, model_b_path, model_c_path, final_path, "Model A failed to process results"
                return
                
            # Save Model A results as CSV
            model_a_path = file_processor.save_csv(model_a_results, "model_a_results.csv")
            if not model_a_path:
                yield None, None, None, None, "Failed to save Model A results"
                return
                
            model_results["model_a"] = model_a_results
            status = update_progress("Model A", "Completed")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            # Run Model B
            logging.info("Starting Model B analysis...")
            status = update_progress("Model B", "Processing...")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            model_b_results = analyzer.process_batch(df, "model_b", {"model_a": model_a_results})
            if model_b_results is None:
                yield model_a_path, model_b_path, model_c_path, final_path, "Model B failed to process results"
                return
                
            # Save Model B results as CSV
            model_b_path = file_processor.save_csv(model_b_results, "model_b_results.csv")
            if not model_b_path:
                yield model_a_path, None, None, None, "Failed to save Model B results"
                return
                
            model_results["model_b"] = model_b_results
            status = update_progress("Model B", "Completed")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            # Run Model C
            logging.info("Starting Model C analysis...")
            status = update_progress("Model C", "Processing...")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            model_c_results = analyzer.process_batch(df, "model_c", {
                "model_a": model_a_results,
                "model_b": model_b_results
            })
            
            if model_c_results is not None:
                # Save Model C results as CSV
                model_c_path = file_processor.save_csv(model_c_results, "model_c_results.csv")
                if not model_c_path:
                    yield model_a_path, model_b_path, None, None, "Failed to save Model C results"
                    return
                    
                model_results["model_c"] = model_c_results
            
            status = update_progress("Model C", "Completed")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            # Merge results
            logging.info("Merging results...")
            status = update_progress("Merge", "Processing...")
            yield model_a_path, model_b_path, model_c_path, final_path, status
            
            merged_df = analyzer.merge_results(df, model_results)
            
            # Export final merged results as XLSX
            final_filename = "final_results.xlsx"
            result_processor.export_to_excel(merged_df, final_filename)
            final_path = final_filename
            
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
            
            # Save deduplication results as CSV (此处未改为 XLSX，可根据需要修改)
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
        gr.Markdown("""
        <div style="text-align: center;">
            <h1>PICOS Literature Analysis System</h1>
            <p>This system uses a multi-model approach to analyze medical literature abstracts.</p>
        </div>
        """)
        
        with gr.Tab("Instructions"):
            gr.Markdown("""
            ## System Overview
            This system helps researchers analyze medical literature by providing tools for citation management, 
            deduplication, and automated PICOS analysis using multiple language models.

            ## Workflow Steps
            1. **Citation Processing**
               - Import citations from different databases:
                 - Pubmed (.nbib files)
                 - Embase (.ris files)
                 - Web of Science (.ris files)
               - Each source will be converted to a standardized CSV format

            2. **Deduplication** (Optional)
               - Upload multiple CSV files from different sources
               - Adjust similarity threshold to control deduplication strictness
               - Get both deduplicated dataset and duplicate clusters report

            3. **PICOS Analysis Setup**
               - Configure PICOS criteria for your specific research question
               - Set up model parameters for each analysis stage:
                 - Model A: Initial screening
                 - Model B: Detailed review
                 - Model C: Arbitration for disagreements
               - Test API connections before proceeding

            4. **Analysis Execution**
               - Upload your processed citation file
               - Run models individually or use the "Run All" option
               - Review and merge results

            ## File Format Requirements
            ### Input Files
            - **Pubmed**: NBIB format (.nbib)
            - **Embase**: RIS format (.ris)
            - **Web of Science**: RIS format (.ris)

            ### Processed CSV Format
            The system will generate standardized CSV files with these columns:
            - **Index**: Unique identifier for each abstract
            - **Title**: Article title
            - **Authors**: Author list (semicolon-separated)
            - **Abstract**: Full abstract text
            - **DOI**: Digital Object Identifier (when available)

            ### Analysis Results
            Each model will generate a CSV file containing:
            - All original citation data
            - PICOS analysis results
            - Inclusion/exclusion decisions
            - Reasoning for decisions

            ## Tips
            - Always test API connections before running full analysis
            - Use deduplication when combining data from multiple sources
            - Consider using Model C only when Models A and B disagree
            - Regular model settings work well for most cases, but you can enable inference mode for models that provide reasoning steps
            """)
        
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
            
            with gr.Tab("Embase"):
                gr.Markdown("""
                ## Embase RIS Processing
                Upload a .ris file from Embase to extract and convert it to CSV format. The extracted data will include:
                - DOI
                - Title
                - Authors
                - Abstract
                """)
                
                with gr.Row():
                    embase_file = gr.File(label="Upload Embase RIS File", file_types=[".ris"])
                    process_embase_btn = gr.Button("Process Embase RIS File")
                
                with gr.Row():
                    embase_preview = gr.Textbox(label="Preview", lines=20)
                    embase_output = gr.File(label="Download CSV")
                
                process_embase_btn.click(
                    parse_nbib,  # 使用相同的处理函数，它会自动检测RIS文件类型
                    inputs=[embase_file],
                    outputs=[embase_output, embase_preview]
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
                        temperature = gr.Slider(label="Temperature", minimum=0, maximum=10, value=config["temperature"])
                        max_tokens = gr.Number(label="Max Tokens", value=config["max_tokens"])
                        batch_size = gr.Number(label="Batch Size", value=config["batch_size"])
                        threads = gr.Slider(label="Threads", minimum=1, maximum=32, step=1, value=config["threads"])
                        timeout = gr.Number(label="Timeout (seconds)", value=config.get("timeout", 180))
                        prompt = gr.Textbox(label="Prompt Template", value=analyzer.prompt_manager.get_prompt(model_key), lines=10)
                        
                        update_btn = gr.Button(f"Update {model_key.upper().replace('_', ' ')} Settings")
                        test_btn = gr.Button(f"Test {model_key.upper().replace('_', ' ')} Connection")
                        status = gr.Textbox(label="Status", lines=10)
                        
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
                                is_inference,
                                timeout
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
                    final_output = gr.File(label="Final Results (XLSX)")
                
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
    
    return interface

if __name__ == "__main__":
    interface = create_gradio_interface()
    if interface:
        interface.launch(server_name="0.0.0.0", server_port=7860)
    else:
        print("Error: Failed to create Gradio interface")
