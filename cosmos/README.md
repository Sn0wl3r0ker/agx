### jetson orin setup:
[jetson-orin-setup.md](https://github.com/Sn0wl3r0ker/cosmos-reason2/blob/feature/jetson-agx-orin-support/docs/jetson-orin-setup.md)
[main readme.md](https://github.com/Sn0wl3r0ker/cosmos-reason2/blob/feature/jetson-agx-orin-support/README.md)

#### installed CUDA12.6, CUDSS in native system.
#### fix CUDSS wrong CUDA version problem -> ImportError: libcublas.so.13: cannot open shared object file: No such file or directory

### cosmos-reason2 vllm server:
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/data/cosmos_cache/hf
export HF_HUB_CACHE=/data/cosmos_cache/hf/hub
export VLLM_CACHE_ROOT=/data/cosmos_cache/vllm
```

```bash
vllm serve /data/models/Cosmos-Reason2-2B --allowed-local-media-path "/" --max-model-len 16384 --gpu-memory-utilization 0.6 --reasoning-parser qwen3 --port 8000 --download-dir /data/cosmos_cache/vllm/ 

test: --max-model-len 8192
```
