# Experiment 6 — Direct Full-Logits Replication

This is a separate replication backend for the frozen 100-task / 600-trajectory Experiment 6 design.

## Why this exists

The Ollama confirmatory runner performs one HTTP request for every teacher-forced token.
The frozen run contains roughly 23k token-level queries, so HTTP/request overhead dominates.

This runner uses Hugging Face Transformers directly:

- one causal forward pass gives logits for all token positions in a sequence;
- trajectories are batched;
- no sampling is required;
- no HTTP request per token;
- full-vocabulary entropy is available exactly.

The runner now defaults to the immutable Qwen revision
`1cfa9a7208912126459214e8b04321603b3df60c`, verifies the resolved commit when
available and writes both the requested and resolved revisions to every new config.
The historical completed run predated this guard; its revision was recovered from
the unique local cache snapshot and documented in the publication provenance note.

## Scientific status

Do NOT merge these rows with the quantized Ollama confirmatory rows.

The Ollama run uses a quantized GGUF model.
This runner uses the official `Qwen/Qwen3-4B` checkpoint in BF16.

That makes this a useful backend/precision replication.

The directly comparable endpoint is still:

`mean_entropy_top20`

The direct runner additionally records exact:

`mean_entropy_full`

## Install on the current Windows / RTX 5090 machine

Install CUDA-enabled PyTorch:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then:

```powershell
pip install -U "transformers>=4.51" accelerate safetensors scipy pandas
```

Verify:

```powershell
python verify_direct_env.py
```

You want:

```text
cuda available: True
GPU 0: NVIDIA GeForce RTX 5090
```

## Important: run this on the SECOND RTX 5090

The current Ollama confirmatory run is using physical GPU 0.

From a NEW PowerShell window:

```powershell
cd .\experiment6_direct_logits
$env:CUDA_VISIBLE_DEVICES="1"
python verify_direct_env.py
```

Inside this process the physical second card will appear as `GPU 0`; that is normal because CUDA hides the first card.

## First do a 20-trajectory benchmark

```powershell
$env:CUDA_VISIBLE_DEVICES="1"
python run_direct_hf.py --limit 20 --results-dir results_direct_test
```

This will download the BF16 model on first use if it is not cached.

After the model is loaded, look at the printed batch timing and ETA.

## Full replication

Use a fresh output directory:

```powershell
$env:CUDA_VISIBLE_DEVICES="1"
python run_direct_hf.py --results-dir results_direct_hf
```

Default batch size is 8. On a 32 GiB RTX 5090, 8 should be conservative.
After a successful small benchmark you can try:

```powershell
python run_direct_hf.py --batch-size 16 --results-dir results_direct_hf
```

Do not change batch size in the middle of a completed scientific run unless you first verify the resulting logits are numerically equivalent enough for the intended analysis. Batch size should not change the mathematical model, but reproducibility is cleaner with one frozen configuration.

## Analyze after 600 trajectories

```powershell
python analyze_direct_hf.py --results-dir results_direct_hf
```

The analyzer reports task-level paired results for:

1. normalized top-20 entropy (directly comparable to Ollama), and
2. exact full-vocabulary entropy.

## Token alignment improvement

Unlike the original Ollama teacher-forcing implementation, this runner tokenizes:

`base_prompt + full_reference`

as one string and then identifies the reference-token span.

It verifies that the base prompt remains an exact token prefix. If not, it stops instead of silently using a possibly wrong BPE boundary.
