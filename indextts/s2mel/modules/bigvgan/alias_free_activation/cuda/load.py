# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import pathlib
import sys

import torch
from torch.utils import cpp_extension


_DLL_DIRECTORY_HANDLES = []


def _add_windows_dll_directories():
    if os.name != "nt":
        return
    candidates = [pathlib.Path(torch.__file__).parent / "lib", pathlib.Path(sys.base_prefix)]
    if cpp_extension.CUDA_HOME:
        candidates.append(pathlib.Path(cpp_extension.CUDA_HOME) / "bin")
    for path in candidates:
        if path.is_dir():
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(path)))


def load():
    # Python 3.8+ no longer searches PATH for extension-module dependencies on
    # Windows. Keep explicit handles alive while the JIT-built CUDA module is in use.
    _add_windows_dll_directories()

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        # PyTorch's generated -gencode flags honor TORCH_CUDA_ARCH_LIST, or
        # detect visible GPUs when the variable is not set.
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=[
                "-O3",
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
            ]
            + extra_cuda_flags,
            verbose=True,
        )

    extra_cuda_flags = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    sources = [
        srcpath / "anti_alias_activation.cpp",
        srcpath / "anti_alias_activation_cuda.cu",
    ]
    anti_alias_activation_cuda = _cpp_extention_load_helper(
        "anti_alias_activation_cuda", sources, extra_cuda_flags
    )

    return anti_alias_activation_cuda


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
