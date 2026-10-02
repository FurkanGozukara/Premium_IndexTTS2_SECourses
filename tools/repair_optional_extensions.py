"""Remove custom CUDA wheels whose recorded PyTorch ABI differs after an update."""
from __future__ import annotations

import argparse
from importlib import metadata
import os
import re
import subprocess
import sys


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove-incompatible", action="store_true")
    args = parser.parse_args()
    packages = incompatible_extensions()
    if packages and args.remove_incompatible:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *packages], check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        print(">> Removed incompatible optional binaries. SDPA and ConvRot remain available; install matching FlashAttention wheels to enable its optional engine.", flush=True)
    return int(bool(packages) and not args.remove_incompatible)


if __name__ == "__main__":
    raise SystemExit(main())
