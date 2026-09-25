py -3.12 -m venv .venv_direct
.\.venv_direct\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

python -m pip install torch `
  --index-url https://download.pytorch.org/whl/cu128 `
  --timeout 1200 `
  --retries 10 `
  --no-cache-dir

python -m pip install `
  transformers==5.17.0 `
  accelerate==1.15.0 `
  safetensors==0.8.0 `
  numpy==2.5.3 `
  pandas==3.0.5

python verify_env.py
