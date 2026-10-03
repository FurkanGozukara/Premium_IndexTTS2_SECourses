"""CUDA graphs whose cuBLAS workspaces have independent lifetimes.

PyTorch releases cuBLAS workspaces for an entire capture stream when a graph
on that stream is destroyed. Its default capture stream is shared, so evicting
one cached graph can invalidate another graph's workspace. See pytorch/pytorch
issue #193402. A dedicated CUDA stream per graph avoids that use-after-free.

Ported from Whisper-WebUI Premium (SECourses), modules/whisper/convrot at revision 613beba (September 2026).
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
import sys

import torch


@lru_cache(maxsize=1)
def _cuda_driver():
    driver = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
    driver.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    driver.cuStreamCreate.restype = ctypes.c_int
    driver.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
    driver.cuStreamDestroy_v2.restype = ctypes.c_int
    return driver


class CudaGraph:
    """Own a graph and a non-pooled stream until the graph has been reset.

    ``torch.cuda.Stream()`` draws from a finite round-robin pool; it cannot
    guarantee independent stream identities for a growing graph cache. The
    CUDA driver library is present on supported Windows and Linux systems.
    Eviction waits for pending device work before releasing captured storage.
    """

    def __init__(self):
        self._device = torch.cuda.current_device()
        self._handle = None
        self._graph = None
        # current_device() alone need not establish a primary driver context.
        torch.empty(1, device=torch.device("cuda", self._device))
        pointer = ctypes.c_void_p()
        # CU_STREAM_NON_BLOCKING prevents implicit synchronization with the
        # legacy default stream, which otherwise makes graph capture invalid.
        error = _cuda_driver().cuStreamCreate(ctypes.byref(pointer), 1)
        if int(error) != 0:
            raise RuntimeError(f"Unable to create a CUDA graph capture stream: {error}")
        self._handle = pointer.value
        self.stream = torch.cuda.ExternalStream(self._handle, device=self._device)
        self._graph = torch.cuda.CUDAGraph()

    def capture(self, **kwargs):
        return torch.cuda.graph(self._graph, stream=self.stream, **kwargs)

    def replay(self):
        self._graph.replay()

    def close(self):
        if self._handle is None:
            return
        with torch.cuda.device(self._device):
            # A max-token return can bypass the decoder's usual CPU completion
            # check. Graph destruction alone does not wait for queued replay.
            torch.cuda.synchronize(self._device)
            if self._graph is not None:
                self._graph.reset()
                self._graph = None
            error = _cuda_driver().cuStreamDestroy_v2(self._handle)
            self._handle = None
            if int(error) != 0:
                raise RuntimeError(f"Unable to destroy a CUDA graph capture stream: {error}")

    def __del__(self):
        try:
            self.close()
        except Exception:
            # The CUDA runtime may already be shutting down at interpreter exit.
            pass
