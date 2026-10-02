"""Dataset preparation and LoRA / DoRA training support for IndexTTS."""

__all__ = ["DatasetPrepConfig", "DatasetSummary", "run_dataset_prep"]


def __getattr__(name):
    # Reading checkpoint metadata must not import ASR, audio decoders and Torch.
    if name in __all__:
        from . import dataset_prep
        return getattr(dataset_prep, name)
    raise AttributeError(name)
