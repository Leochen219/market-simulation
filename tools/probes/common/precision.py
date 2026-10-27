"""Enforce and audit the Laya FP32 inference path."""
from __future__ import annotations
from typing import Any

def configure_fp32(agent: Any, torch: Any) -> dict[str, Any]:
    """Disable Laya autocast, force FP32 weights, and reject reduced-precision forwards."""
    agent.amp_enabled = False
    agent.dtype = torch.float32
    agent.model.to(device=agent.device, dtype=torch.float32)
    agent.model.eval()
    matmul_precision = None
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
        matmul_precision = torch.get_float32_matmul_precision()
        if matmul_precision != "highest":
            raise RuntimeError(f"float32 matmul precision did not stay highest: {matmul_precision}")
    allow_tf32 = None
    cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        matmul_backend.allow_tf32 = False
        allow_tf32 = bool(matmul_backend.allow_tf32)
        if allow_tf32:
            raise RuntimeError("CUDA matmul TF32 remained enabled")
    cudnn_backend = getattr(getattr(torch, "backends", None), "cudnn", None)
    cudnn_allow_tf32 = None
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        cudnn_backend.allow_tf32 = False
        cudnn_allow_tf32 = bool(cudnn_backend.allow_tf32)
        if cudnn_allow_tf32:
            raise RuntimeError("cuDNN TF32 remained enabled")
    forward_dtype = agent.dtype_for(170)
    if forward_dtype != torch.float32:
        raise RuntimeError(f"Laya dtype_for(batch rows) is {forward_dtype}, expected torch.float32")
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in agent.model.parameters()})
    if parameter_dtypes and parameter_dtypes != ["torch.float32"]:
        raise RuntimeError(f"Laya parameters are not all float32: {parameter_dtypes}")
    return {
        "agent_amp_enabled": bool(agent.amp_enabled),
        "agent_dtype": str(agent.dtype),
        "dtype_for_large_batch": str(forward_dtype),
        "dtype_for_large_batch_is_fp32": forward_dtype == torch.float32,
        "parameter_dtypes": parameter_dtypes,
        "float32_matmul_precision": matmul_precision,
        "cuda_matmul_allow_tf32": allow_tf32,
        "cudnn_allow_tf32": cudnn_allow_tf32,
    }
