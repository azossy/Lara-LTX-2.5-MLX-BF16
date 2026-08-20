# ComfyUI-LaraLTX

This is an optional thin ComfyUI adapter for `lara-ltx`. It contains no model,
scheduler, VAE, decoder, PyTorch/MPS, or Metal implementation. Every generation
delegates to the same in-process `LTXPipeline` used by the Python API and CLI.

Install `lara-ltx` in ComfyUI's Python environment, then place this directory
under `ComfyUI/custom_nodes/ComfyUI-LaraLTX` and restart ComfyUI.

The initial workflow is deliberately small:

1. `Lara LTX Model Loader`
2. `Lara LTX Text to Video`
3. `Lara LTX Save Video`

Import `examples/text_to_video_api.json` through ComfyUI's API workflow path,
or connect the same three nodes in the UI. Adjust the model ID to a local model
directory and enable `local_files_only` when running offline. The adapter and
example default to the measured 512x320/17-frame grid; larger values remain
editable but are subject to the core profile's preflight policy.

Set `LARA_LOCALE=ko` before starting ComfyUI for Korean node names and errors.
The save node confines generated MP4 files to ComfyUI's configured output
directory. Image conditioning remains unavailable until its core parity gate is
complete.
