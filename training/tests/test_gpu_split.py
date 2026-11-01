"""train.gpu_shares: one GPU shared by default, one each only when asked for and possible."""
import train


def test_the_default_is_one_shared_gpu(monkeypatch):
    monkeypatch.delenv("PSPPO_GPU_SPLIT", raising=False)
    for gpus in (0, 1, 2, 8):
        assert train.gpu_shares(gpus) == (0.35, 0.65)
    monkeypatch.setenv("PSPPO_GPU_SPLIT", "0")
    assert train.gpu_shares(8) == (0.35, 0.65)


def test_a_gpu_each_only_when_asked_for_and_there_are_two(monkeypatch):
    monkeypatch.setenv("PSPPO_GPU_SPLIT", "1")
    assert train.gpu_shares(8) == (1.0, 1.0) and train.gpu_shares(2) == (1.0, 1.0)
    assert train.gpu_shares(1) == (0.35, 0.65) and train.gpu_shares(0) == (0.35, 0.65)
