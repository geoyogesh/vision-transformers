mise use python@3.12

uv venv
source .venv/bin/activate
uv pip install torch torchvision jupyter
uv pip install ipykernel
uv pip install matplotlib

jupyter notebook
jupyter notebook lab