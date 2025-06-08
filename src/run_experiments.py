import argparse
import subprocess
import concurrent.futures
import time
import os
import sys
import re
from datetime import datetime
import ctypes

def parse_metrics(output_file):
    """
    Parse metrics from output file
    """
    metrics = {}
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Parse Accuracy
            acc_match = re.search(r'Accuracy:\s*([\d.]+)%', content)
            if acc_match:
                metrics['accuracy'] = float(acc_match.group(1)) / 100

            # Parse Entailment Ratio
            ratio_match = re.search(r'Average Entailment Ratio:\s*([\d.]+)%', content)
            if ratio_match:
                metrics['entailment_ratio'] = float(ratio_match.group(1)) / 100
    except Exception as e:
        print(f"Error parsing metrics: {str(e)}")
        metrics = {'accuracy': None, 'entailment_ratio': None}
    
    return metrics

def run_experiment(script_name, prompt_type, model_name, sample_size):
    """
    Run a single experiment script
    
    Parameters:
        script_name: script filename
        prompt_type: prompt type ('templated' or 'natural')
        model_name: model to use (e.g., 'mistral:7b', 'falcon3:7b')
        sample_size: number of samples to evaluate
    Returns:
        metrics: dictionary containing accuracy and entailment ratio
    """
    start_time = time.time()
    print(f"\n{'='*50}")
    print(f"Starting {script_name} - {prompt_type}")
    print(f"Model: {model_name}")
    print(f"Sample size: {sample_size}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create temporary output file
    output_file = f"temp_output_{script_name.replace('.py', '')}_{prompt_type}_{model_name.replace(':', '_')}.txt"
    
    try:
        # Set environment variables to pass parameters
        env = os.environ.copy()
        env["PROMPT_TYPE"] = prompt_type
        env["MODEL_NAME"] = model_name
        env["SAMPLE_SIZE"] = str(sample_size)
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set PYTHONUNBUFFERED=1 to ensure Python doesn't buffer output
        env["PYTHONUNBUFFERED"] = "1"
        
        # Run module using python -m
        module_name = f"src.{script_name[:-3]}"
        
        # On Windows, we need to set these flags to properly handle ANSI escape sequences
        if os.name == 'nt':
            import msvcrt
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(msvcrt.get_osfhandle(sys.stdout.fileno()), 7)
        
        # Run process directly and display output in real-time
        process = subprocess.Popen(
            [sys.executable, "-m", module_name],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        # Save output for later metric parsing
        output_lines = []
        
        # Read and display output in real-time
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                # Save output
                output_lines.append(line)
                # Display output in real-time
                print(line, end='', flush=True)
        
        # Wait for process to complete
        process.wait()
        
        # Save complete output to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
        
        if process.returncode == 0:
            print(f"\n{script_name} - {prompt_type} executed successfully")
        else:
            print(f"\n{script_name} - {prompt_type} execution failed (return code: {process.returncode})")
        
        # Parse metrics
        metrics = parse_metrics(output_file)
        
    except Exception as e:
        print(f"\nError executing {script_name}: {str(e)}")
        metrics = {'accuracy': None, 'entailment_ratio': None}
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution time: {duration:.2f} seconds")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*50)
    
    # Clean up temporary file
    try:
        os.remove(output_file)
    except:
        pass
    
    return metrics

def print_summary(all_results, model_name, sample_size):
    """
    Print summary of all experiment results
    """
    print("\n" + "="*80)
    print(f"Experiment Results Summary")
    print(f"Model: {model_name}")
    print(f"Sample size: {sample_size}")
    print("="*80)
    print(f"{'Dataset':<15} {'Prompt Type':<12} {'Accuracy':<12} {'Entailment Ratio':<15}")
    print("-"*80)
    
    for script in ['main.py', 'main_cose_entail.py']:
        dataset = "CommonsenseQA" if script == "main.py" else "CoS-E"
        for prompt_type in ['simple', 'templated', 'natural']:
            metrics = all_results[f"{script}_{prompt_type}"]
            acc = f"{metrics['accuracy']*100:.2f}%" if metrics['accuracy'] is not None else "N/A"
            ratio = f"{metrics['entailment_ratio']*100:.2f}%" if metrics['entailment_ratio'] is not None else "N/A"
            print(f"{dataset:<15} {prompt_type:<12} {acc:<12} {ratio:<15}")
    
    print("="*80)

def run_sequential(model_name, sample_size):
    """Execute all experiments sequentially"""
    experiments = [
        ("main.py", "simple"),
        ("main.py", "templated"),
        ("main.py", "natural"),
        ("main_cose_entail.py", "simple"),
        ("main_cose_entail.py", "templated"),
        ("main_cose_entail.py", "natural")
    ]
    
    print(f"\nStarting sequential execution of all experiments...")
    print(f"Model: {model_name}")
    print(f"Sample size: {sample_size}")
    total_start = time.time()
    
    all_results = {}
    for script, prompt_type in experiments:
        metrics = run_experiment(script, prompt_type, model_name, sample_size)
        all_results[f"{script}_{prompt_type}"] = metrics
    
    total_end = time.time()
    print(f"\nAll experiments completed! Total time: {(total_end - total_start):.2f} seconds")
    
    # Print summary results
    print_summary(all_results, model_name, sample_size)

def run_parallel(model_name, sample_size):
    """Execute all experiments in parallel"""
    experiments = [
        ("main.py", "simple"),
        ("main.py", "templated"),
        ("main.py", "natural"),
        ("main_cose_entail.py", "simple"),
        ("main_cose_entail.py", "templated"),
        ("main_cose_entail.py", "natural")
    ]
    
    print(f"\nStarting parallel execution of all experiments...")
    print(f"Model: {model_name}")
    print(f"Sample size: {sample_size}")
    total_start = time.time()
    
    all_results = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_exp = {
            executor.submit(run_experiment, script, prompt_type, model_name, sample_size): (script, prompt_type)
            for script, prompt_type in experiments
        }
        
        for future in concurrent.futures.as_completed(future_to_exp):
            script, prompt_type = future_to_exp[future]
            try:
                metrics = future.result()
                all_results[f"{script}_{prompt_type}"] = metrics
            except Exception as e:
                print(f"Error generating results for {script} - {prompt_type}: {str(e)}")
                all_results[f"{script}_{prompt_type}"] = {
                    'accuracy': None,
                    'entailment_ratio': None
                }
    
    total_end = time.time()
    print(f"\nAll experiments completed! Total time: {(total_end - total_start):.2f} seconds")
    
    # Print summary results
    print_summary(all_results, model_name, sample_size)

def get_script_by_dataset(dataset):
    """Get corresponding script filename based on dataset name"""
    if dataset.lower() in ['commonsenseqa', 'csqa']:
        return 'main.py'
    elif dataset.lower() in ['cose', 'cos-e']:
        return 'main_cose_entail.py'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def run_single_task(dataset, prompt_type, model_name, sample_size):
    """Run a single specified task"""
    try:
        script = get_script_by_dataset(dataset)
        print(f"\nStarting single task: {dataset} - {prompt_type}")
        print(f"Model: {model_name}")
        print(f"Sample size: {sample_size}")
        metrics = run_experiment(script, prompt_type, model_name, sample_size)
        
        # Print single task result summary
        print("\n" + "="*80)
        print("Experiment Result")
        print(f"Model: {model_name}")
        print(f"Sample size: {sample_size}")
        print("="*80)
        print(f"Dataset: {dataset}")
        print(f"Prompt Type: {prompt_type}")
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%" if metrics['accuracy'] is not None else "Accuracy: N/A")
        print(f"Entailment Ratio: {metrics['entailment_ratio']*100:.2f}%" if metrics['entailment_ratio'] is not None else "Entailment Ratio: N/A")
        print("="*80)
        
    except Exception as e:
        print(f"Error running task: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment scripts")
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel", "single"],
        default="sequential",
        help="Execution mode: sequential(sequential execution), parallel(parallel execution) or single(single task)"
    )
    parser.add_argument(
        "--dataset",
        choices=["commonsenseqa", "cose"],
        help="When mode is single, specify the dataset to run"
    )
    parser.add_argument(
        "--prompt_type",
        choices=["simple", "templated", "natural"],
        help="When mode is single, specify the prompt type to run"
    )
    parser.add_argument(
        "--model",
        choices=["mistral:7b", "falcon3:7b"],
        default="mistral:7b",
        help="Model to use for experiments"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=103,
        help="Number of samples to evaluate for each dataset"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.dataset or not args.prompt_type:
            parser.error("In single mode, both --dataset and --prompt_type parameters must be specified")
        run_single_task(args.dataset, args.prompt_type, args.model, args.sample_size)
    elif args.mode == "parallel":
        run_parallel(args.model, args.sample_size)
    else:  # sequential
        run_sequential(args.model, args.sample_size) 