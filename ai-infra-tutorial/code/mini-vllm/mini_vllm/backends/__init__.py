"""Backend factory: default to TorchBackend; Triton is explicit opt-in."""
import os

from mini_vllm.backends.torch_backend import TorchBackend


def make_backend(device: str = "cpu", prefer_triton: bool | None = None):
    """Return an `AttentionBackend` instance suitable for `device`.

    `device` is the str form ("cpu", "mps", "cuda", "cuda:0", ...).
    TorchBackend is the default, including on CUDA. Triton is experimental
    and only selected when explicitly requested with
    `MINI_VLLM_BACKEND=triton` or `prefer_triton=True`.
    """
    if prefer_triton is None:
        prefer_triton = os.getenv("MINI_VLLM_BACKEND", "").lower() == "triton"
    if device.startswith("cuda") and prefer_triton:
        try:
            from mini_vllm.backends.triton_backend import TritonBackend, HAS_TRITON
            if HAS_TRITON:
                return TritonBackend()
        except Exception:
            pass
    return TorchBackend()
