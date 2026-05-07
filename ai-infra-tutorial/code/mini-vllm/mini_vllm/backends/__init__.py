"""Backend factory: pick TorchBackend (CPU/MPS) or TritonBackend (CUDA + Triton)."""
from mini_vllm.backends.torch_backend import TorchBackend


def make_backend(device: str = "cpu", prefer_triton: bool = True):
    """Return an `AttentionBackend` instance suitable for `device`.

    `device` is the str form ("cpu", "mps", "cuda", "cuda:0", ...).
    On CUDA we try Triton first; fall back to TorchBackend if `triton` isn't
    importable. Pass `prefer_triton=False` to force the Torch reference even
    on CUDA (useful for parity testing).
    """
    if device.startswith("cuda") and prefer_triton:
        try:
            from mini_vllm.backends.triton_backend import TritonBackend, HAS_TRITON
            if HAS_TRITON:
                return TritonBackend()
        except Exception:
            pass
    return TorchBackend()
