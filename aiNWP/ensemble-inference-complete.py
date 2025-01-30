"""
Enhanced Ensemble Inference Script with Optimized Memory Management
This script runs ensemble inference using the Pangu weather model with carefully tuned
memory management settings. It includes optimizations for large models on high-memory GPUs,
with particular attention to preventing memory fragmentation and out-of-memory errors.
"""

import json
import os
import torch
import onnxruntime as ort
import xarray
import dotenv
from pynvml import (
    nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName, nvmlShutdown
)

def print_gpu_status(message=""):
    """
    Monitor GPU memory usage at different stages of execution.
    This helps track memory allocation and identify potential memory leaks.
    
    Args:
        message (str): Context message to identify the monitoring point
    """
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        name = nvmlDeviceGetName(handle)
        
        print(f"\n=== GPU Status {message} ===")
        print(f"GPU: {name.decode('utf-8') if isinstance(name, bytes) else name}")
        print(f"Memory Total: {info.total / (1024**3):.2f} GB")
        print(f"Memory Used: {info.used / (1024**3):.2f} GB")
        print(f"Memory Free: {info.free / (1024**3):.2f} GB")
        print(f"Memory Utilization: {(info.used / info.total) * 100:.2f}%")
        
        nvmlShutdown()
    except Exception as e:
        print(f"\nWarning: Could not get GPU status: {str(e)}")

def setup_gpu_memory_management():
    """
    Configure GPU memory management settings for optimal performance.
    Implements conservative memory allocation strategies to prevent OOM errors
    and memory fragmentation issues.
    
    Returns:
        tuple: ONNX Runtime providers and session options
    """
    # Start with a clean GPU memory state
    torch.cuda.empty_cache()
    print("\nCUDA cache cleared")

    # Ensure GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU to run.")

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Configure ONNX Runtime with conservative memory settings
    provider_options = [
        {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',  # More predictable memory allocation
            'gpu_mem_limit': 12 * 1024 * 1024 * 1024,    # 12GB limit to leave headroom
            'cudnn_conv_algo_search': 'DEFAULT',          # Conservative algorithm selection
            'cudnn_conv_use_max_workspace': '0',          # Minimize extra memory usage
            'do_copy_in_default_stream': '1',
        }
    ]

    # Set up providers with CPU fallback for reliability
    providers = [
        ('CUDAExecutionProvider', provider_options[0]),
        'CPUExecutionProvider'
    ]

    # Configure session options for memory efficiency
    session_options = ort.SessionOptions()
    session_options.enable_mem_pattern = True        # Enable memory optimization patterns
    session_options.enable_mem_reuse = True         # Allow memory reuse between operations
    session_options.intra_op_num_threads = 1        # Single thread per operation
    session_options.inter_op_num_threads = 1        # Single thread between operations
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Configure PyTorch memory settings
    torch.cuda.set_per_process_memory_fraction(0.6)  # Use only 60% of available memory
    torch.backends.cudnn.benchmark = False           # Disable auto-tuning to save memory
    torch.backends.cudnn.deterministic = True        # Ensure consistent memory usage

    # Set environment variables for memory management
    os.environ['ONNXRUNTIME_CUDA_MEMORY_LIMIT'] = str(12 * 1024 * 1024 * 1024)  # 12GB
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'

    print("\nMemory settings configured:")
    print(f"- GPU Memory Limit: 12 GB")
    print(f"- Memory Fraction: 60%")
    print(f"- Max Split Size: 256 MB")
    print(f"- Using expandable segments: True")

    return providers, session_options

def setup_cds_api():
    """
    Configure the Climate Data Store API if not already set up.
    Creates or verifies the existence of the .cdsapirc configuration file.
    """
    cds_api = os.path.join(os.path.expanduser("~"), ".cdsapirc")
    if not os.path.exists(cds_api):
        uid = input("Enter in CDS UID (e.g. 123456): ")
        key = input("Enter your CDS API key (e.g. 12345678-1234-1234-1234-123456123456): ")
        with open(cds_api, "w") as f:
            f.write("url: https://cds.climate.copernicus.eu/api/v2\n")
            f.write(f"key: {uid}:{key}\n")

def open_ensemble(f, domain, chunks={"time": 1}):
    """
    Open and process ensemble output data into an xarray Dataset.
    Uses chunked reading for memory efficiency with large datasets.
    
    Args:
        f (str): Path to the ensemble output file
        domain (str): Domain name to process
        chunks (dict): Chunk sizes for dask array creation
        
    Returns:
        xarray.Dataset: Processed ensemble data
    """
    time = xarray.open_dataset(f).time
    root = xarray.open_dataset(f, decode_times=False)
    ds = xarray.open_dataset(f, chunks=chunks, group=domain)
    ds.attrs = root.attrs
    return ds.assign_coords(time=time)

def main():
    """
    Main execution function that runs the ensemble inference pipeline.
    Implements memory-efficient processing and monitoring throughout the workflow.
    """
    # Monitor initial GPU state
    print_gpu_status("(Initial)")

    # Set up optimized GPU memory management
    providers, session_options = setup_gpu_memory_management()

    # Load environment variables
    dotenv.load_dotenv()

    # Import Earth-2 MIP components after environment setup
    from earth2mip import inference_ensemble, registry

    # Monitor GPU state before model loading
    print_gpu_status("(Before Model Loading)")

    print('Downloading Pangu model...')
    print("Fetching pangu_operational model package...")
    package = registry.get_model("e2mip://pangu_6")
    print("Successfully loaded model registry")

    # Set up CDS API if needed
    setup_cds_api()

    # Configure the ensemble run with memory-efficient settings
    config = {
        "ensemble_members": 12,
        "noise_amplitude": 0.05,
        "simulation_length": 10,
        "weather_event": {
            "properties": {
                "name": "Globe",
                "start_time": "2024-12-25 00:00:00",
                "initial_condition_source": "cds",
            },
            "domains": [
                {
                    "name": "global",
                    "type": "Window",
                    "diagnostics": [{"type": "raw", "channels": ["t2m", "u10m"]}],
                }
            ],
        },
        "output_path": "outputs/01_ensemble_notebook",
        "output_frequency": 1,
        "weather_model": "pangu_6",
        "seed": 12345,
        "use_cuda_graphs": False,
        "ensemble_batch_size": 1,
        "autocast_fp16": True,  # Enable mixed precision for memory efficiency
        "perturbation_strategy": "correlated",
        "noise_reddening": 2.0,
        "memory_efficient": True  # Enable memory-efficient mode
    }

    # Monitor GPU state before inference
    print_gpu_status("(Before Inference)")

    # Run the ensemble inference
    config_str = json.dumps(config)
    inference_ensemble.main(config_str)

    # Monitor final GPU state
    print_gpu_status("(After Inference)")

    # Process output
    output_path = config["output_path"]
    domains = config["weather_event"]["domains"][0]["name"]
    ensemble_members = config["ensemble_members"]
    ds = open_ensemble(os.path.join(output_path, "ensemble_out_0.nc"), domains)

    print("\nProcessing complete!")
    return ds

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Ensure we see GPU status even if an error occurs
        print("\nAn error occurred during execution:")
        print(str(e))
        print("\nFinal GPU status:")
        print_gpu_status("(After Error)")
        raise
