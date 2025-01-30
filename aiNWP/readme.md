This Python script performs ensemble inference using the Pangu weather model, focusing on optimized memory management. It's designed to handle large models and datasets on GPUs with limited memory, minimizing the risk of out-of-memory errors and memory fragmentation.

## Features

*   **GPU Memory Monitoring:** Tracks GPU memory usage at various stages of execution using `pynvml`.  This helps identify potential memory leaks or excessive memory consumption.
*   **Optimized Memory Management:** Implements several strategies to reduce memory footprint:
    *   Clears CUDA cache at the beginning.
    *   Sets a GPU memory limit (12GB in this example, adjustable).
    *   Uses conservative memory allocation strategies with ONNX Runtime (e.g., `arena_extend_strategy`, `cudnn_conv_algo_search`, `cudnn_conv_use_max_workspace`).
    *   Configures ONNX Runtime session options for memory efficiency (`enable_mem_pattern`, `enable_mem_reuse`, limited threads).
    *   Sets PyTorch memory fraction and disables cuDNN benchmarking.
    *   Uses environment variables (`ONNXRUNTIME_CUDA_MEMORY_LIMIT`, `PYTORCH_CUDA_ALLOC_CONF`) to control memory allocation.
    *   Enables mixed precision (FP16) for the model.
    *   Uses memory-efficient mode in the Earth-2 MIP configuration.
*   **Climate Data Store (CDS) API Setup:**  Automatically configures the CDS API by creating or verifying the `.cdsapirc` file.  Prompts the user for their UID and API key if the file doesn't exist.
*   **Chunked Data Loading:** Opens and processes large ensemble output data using `xarray` with chunking. This allows working with datasets that are larger than available memory.
*   **Error Handling:** Includes a `try...except` block to catch and report errors, while still displaying the GPU status in case of failure.
*   **Earth-2 MIP Integration:** Uses the Earth-2 MIP library for model inference.  The necessary components are imported *after* environment variables are set.
*   **Ensemble Configuration:** Allows customization of ensemble parameters like the number of members, simulation length, and output path.

## Requirements

*   Python 3.x
*   PyTorch
*   ONNX Runtime
*   xarray
*   pynvml
*   dotenv
*   earth2mip
*   cdsapi
