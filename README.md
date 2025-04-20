# comfyui-optimizations
A bit of optimizations for ComfyUI.

# Installation
Download the .py files in this repo, and replace the original files in the ComfyUI/comfy folder with these.

# About
This mainly changes three files in the ComfyUI/comfy folder. These are not exactly going to make images generate instantly; but they should shave a good few seconds off for users with weaker GPUs. I have tested these optimizations with SDXL and it works fully with everything I tested:
-txt2img
-ControlNet
-LoRA
-IPAdapter

And works for other models as well. This also adds a new scheduler to samplers.py, zipf_linear.

## samplers.py
1. Efficient Ramp Border ("Fuzz") Calculation

    The original implementation applies a ramping calculation directly and repeatedly for border fuzzing, involving multiple narrow operations in loops.

    The optimized version precomputes and caches ramping arrays based on their shape and border size. Subsequent calls reuse cached tensors, significantly reducing redundant computations.

2. Conditioning Cache

    The optimized file introduces _conditioning_cache, caching conditioning tensors associated with unique UUIDs.

    Redundant processing is avoided when conditions have been processed previously.

3. Efficient Grouping for Batching

    The optimized version explicitly groups conditions by (hooks_id, input_x.shape) to enable maximal batching through the model (U-Net). This reduces repeated GPU calls, optimizing GPU utilization.

4. Per-Pixel Normalization Optimization

    Rather than simply dividing condition outputs by counts, the optimized file explicitly computes an inverse-count mask, efficiently normalizing outputs without divisions by zero.

5. Pure-PyTorch GPU-based Schedulers

    The optimized file implements several schedulers (simple_scheduler, ddim_scheduler, normal_scheduler, etc.) entirely using GPU-accelerated PyTorch tensor operations, eliminating loops in Python space and substantially accelerating scheduler computation.

6. Vectorization and Device-local Computations

    Throughout, the optimized file heavily leverages vectorized tensor operations to minimize CPU-GPU transfers and Python loop overhead.

## sampler_helpers.py

1. Additional Models Computation is something that is always recomputed in the original file, but here, we simply cache it based on the cond and dtype arguments.
2. Memory Requirement Computation is now cached, based on model and shape.
3. Hook extraction clarity has been refined.

## model_sampling.py

Not much has been changed here, but the first function in the file, rescale_zero_terminal_snr_sigmas, is now a cached function, and makes models that use ZSNR (eg. NoobAI VPred) sample a bit quicker due to no longer needing to perform redundant calculations.
