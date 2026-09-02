import torch
from torch import nn

from indextts.runtime.residency import ResidencyManager


class Wrapper:
    def __init__(self):
        self.llm = nn.Linear(3, 2)


def test_cpu_residency_is_a_noop_and_supports_wrappers():
    manager = ResidencyManager("cpu")
    resident = nn.Linear(2, 2)
    wrapper = Wrapper()
    manager.register("resident", resident, "gpu")
    manager.register("wrapped", wrapper, "on_demand")
    with manager.use("wrapped") as value:
        assert value is wrapper
        assert wrapper.llm.weight.device.type == "cpu"
    assert manager.resident_bytes() > 0
    summary = manager.summary()
    assert summary["models"]["wrapped"]["policy"] == "on_demand"
    manager.to_cpu_all()


def test_invalid_policy_is_rejected():
    manager = ResidencyManager(torch.device("cpu"))
    try:
        manager.register("bad", nn.Linear(1, 1), "sometimes")
    except ValueError as exc:
        assert "Invalid residency policy" in str(exc)
    else:
        raise AssertionError("invalid policy was accepted")


def test_explicit_cpu_policy_never_moves_to_managed_device():
    manager = ResidencyManager(torch.device("cpu"))
    module = nn.Linear(2, 2)
    manager.register("reference", module, "cpu")
    with manager.use("reference"):
        assert module.weight.device.type == "cpu"
    assert manager.summary()["models"]["reference"]["policy"] == "cpu"
