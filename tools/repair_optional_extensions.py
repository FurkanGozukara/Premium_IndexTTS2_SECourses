"""Repair binaries that no longer match the installed PyTorch after an update.

Removes custom CUDA wheels built for another PyTorch release, and reinstalls torchaudio and
torchvision from the installed PyTorch's own CUDA index when their CUDA build differs or they
fail to import. OmniVoice's pyproject points uv at its own CUDA 12.8 index for torch and
torchaudio ([tool.uv.sources]), which installers before 8.1 applied, so a fresh install could end
with a cu128 torchaudio beside a cu130 PyTorch that cannot load it.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import os
import re
import subprocess
import sys

COMPANIONS = ("torchaudio", "torchvision")
PYTORCH_INDEX = "https://download.pytorch.org/whl/{tag}"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def incompatible_extensions():
    torch_version = metadata.version("torch").split("+")[0]
    current = ".".join(torch_version.split(".")[:2])
    result = []
    for name in ("flash-attn", "sageattention", "mslk", "torchao", "xformers"):
        try:
            package = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        provenance = package.version + " " + (package.read_text("direct_url.json") or "")
        build = re.search(r"torch(\d+\.\d+)", provenance)
        if build and build.group(1) != current:
            result.append(name)
            print(f">> Optional {name} was built for PyTorch {build.group(1)}; installed PyTorch is {torch_version}.", flush=True)
    return result


def _imports(module: str) -> bool:
    """Import in a fresh interpreter: a failed native library load must not poison this process."""
    probe = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True,
                           creationflags=NO_WINDOW)
    return probe.returncode == 0


def mismatched_companions():
    """torchaudio / torchvision whose CUDA build differs from PyTorch's, or that cannot be imported."""
    torch_version = metadata.version("torch")
    tag = torch_version.partition("+")[2]
    result = []
    for name in COMPANIONS:
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        own_tag = version.partition("+")[2]
        if tag and own_tag != tag:
            print(f">> {name} {version} was built for {own_tag or 'another CUDA build'}; installed PyTorch is {torch_version}.",
                  flush=True)
            result.append(name)
        elif not _imports(name):
            print(f">> {name} {version} cannot be imported beside PyTorch {torch_version}.", flush=True)
            result.append(name)
    return result


def reinstall_companions(packages):
    """Reinstall the companions from PyTorch's CUDA index with the installed PyTorch pinned."""
    torch_version = metadata.version("torch")
    tag = torch_version.partition("+")[2]
    if not tag.startswith("cu"):
        print(f">> PyTorch {torch_version} names no CUDA build; reinstalling {', '.join(packages)} from PyPI.", flush=True)
        indexes = []
    else:
        indexes = ["--extra-index-url", PYTORCH_INDEX.format(tag=tag)]
    command = [sys.executable, "-m", "uv", "pip", "install", "--python", sys.executable, "--no-sources",
               "--index-strategy", "unsafe-best-match", *indexes, f"torch=={torch_version}", *packages,
               *[part for name in packages for part in ("--reinstall-package", name)]]
    print(">> " + " ".join(command[1:]), flush=True)
    subprocess.run(command, check=True, creationflags=NO_WINDOW)
    broken = [name for name in packages if not _imports(name)]
    if broken:
        raise SystemExit(f">> {', '.join(broken)} still cannot be imported after the reinstall.")
    print(f">> Reinstalled {', '.join(packages)} for PyTorch {torch_version}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove-incompatible", action="store_true")
    args = parser.parse_args()
    packages = incompatible_extensions()
    if packages and args.remove_incompatible:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *packages], check=True, creationflags=NO_WINDOW)
        print(">> Removed incompatible optional binaries. SDPA and ConvRot remain available; install matching FlashAttention wheels to enable its optional engine.", flush=True)
    companions = mismatched_companions()
    if companions and args.remove_incompatible:
        reinstall_companions(companions)
    return int(bool(packages or companions) and not args.remove_incompatible)


if __name__ == "__main__":
    raise SystemExit(main())
