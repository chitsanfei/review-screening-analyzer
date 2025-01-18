import os
import logging
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv
from file_processor import FileProcessor
from analyzer import PICOSAnalyzer

# Load environment variables
load_dotenv()

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Initialize components
file_processor = FileProcessor(DATA_DIR)
analyzer = PICOSAnalyzer()
model_results = {}

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
        """Parse NBIB file and return results"""
        if not file:
            return None, "Invalid file"
        return file_processor.parse_nbib(file.name)

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
    
    def update_model_settings(model_key, api_url, api_key, model_name, temperature, max_tokens, batch_size, threads, prompt):
        """Update model settings"""
        try:
            config = {
                "api_url": api_url,
                "api_key": api_key,
                "model": model_name,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "batch_size": int(batch_size),
                "threads": int(threads)
            }
            analyzer.update_model_config(model_key, config)
            analyzer.update_prompt(model_key, prompt)
            return f"✓ Model settings updated"
        except Exception as e:
            return f"❌ Error updating settings: {str(e)}"
    
    def test_connection(model_key):
        """Test API connection"""
        return analyzer.test_api_connection(model_key)
    
    def process_model(input_file, model_key, model_a_input=None, model_b_input=None):
        """Process analysis for a single model"""
        try:
            # Read CSV file using file processor
            df = file_processor.load_csv(input_file.name)
            if df is None:
                return None, "Failed to load CSV file"
            
            # 检查并加载之前的模型结果
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
    
    def merge_all_results(input_file):
        """Merge all model results"""
        try:
            if not all(k in model_results for k in ["model_a", "model_b"]):
                return None, "Model A and B results required"
            
            # Read CSV file using file processor
            df = file_processor.load_csv(input_file.name)
            if df is None:
                return None, "Failed to load CSV file"
            
            merged_df = analyzer.merge_results(df, model_results)
            
            # Save results
            output_path = file_processor.save_csv(merged_df, "final_results.csv")
            if not output_path:
                return None, "Failed to save merged results"
                
            return output_path, "Results merged successfully"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
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
    
    # Create Gradio interface
    interface = gr.Blocks(title="PICOS Analysis System")
    
    with interface:
        gr.Markdown("# PICOS Literature Analysis System")
        gr.Markdown("This system uses a multi-model approach to analyze medical literature abstracts.")
        
        with gr.Tab("NBIB Processing"):
            gr.Markdown("""
            ## NBIB File Processing
            Upload a .nbib file to extract and convert it to CSV format. The extracted data will include:
            - DOI
            - Title
            - Authors
            - Abstract
            """)
            
            with gr.Row():
                nbib_file = gr.File(label="Upload NBIB File", file_types=[".nbib"])
                process_btn = gr.Button("Process NBIB File")
            
            with gr.Row():
                preview = gr.Textbox(label="Preview", lines=20)
                csv_output = gr.File(label="Download CSV")
            
            process_btn.click(
                parse_nbib,
                inputs=[nbib_file],
                outputs=[csv_output, preview]
            )
        
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
                    temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, value=config["temperature"])
                    max_tokens = gr.Number(label="Max Tokens", value=config["max_tokens"])
                    batch_size = gr.Number(label="Batch Size", value=config["batch_size"])
                    threads = gr.Slider(label="Threads", minimum=1, maximum=32, step=1, value=config["threads"])
                    prompt = gr.Textbox(label="Prompt Template", value=analyzer.prompt_manager.get_prompt(model_key), lines=10)
                    
                    update_btn = gr.Button(f"Update {model_key.upper().replace('_', ' ')} Settings")
                    test_btn = gr.Button(f"Test {model_key.upper().replace('_', ' ')} Connection")
                    status = gr.Textbox(label="Status")
                    
                    update_btn.click(
                        update_model_settings,
                        inputs=[gr.Textbox(value=model_key, visible=False), api_url, api_key, model_name,
                               temperature, max_tokens, batch_size, threads, prompt],
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
        1. Start by processing your NBIB file in the "NBIB Processing" tab
        2. Configure model settings in the "Model Settings" tab
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