from types import SimpleNamespace
from contextlib import nullcontext
from capclaims.backends.laya import release_unused_cuda_cache


def test_release_only_unused_cuda_cache_above_budget():
    calls = []
    cuda = SimpleNamespace(memory_reserved=lambda device: 5 * 1024**3,
                           memory_allocated=lambda device: 2 * 1024**3,
                           device=lambda device: nullcontext(),
                           empty_cache=lambda: calls.append("released"))
    torch = SimpleNamespace(cuda=cuda)
    release_unused_cuda_cache(torch,"cpu")
    assert calls == []
    release_unused_cuda_cache(torch,"cuda")
    assert calls == ["released"]
    cuda.memory_reserved = lambda device: 2 * 1024**3
    release_unused_cuda_cache(torch,"cuda")
    assert calls == ["released"]
