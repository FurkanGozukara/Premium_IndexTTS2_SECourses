"""IndexTTS 2.5 Premium package.

When ``INDEXTTS_VRAM_EMULATE_GB`` names a card size, every process that imports
this package behaves like a card of that size (allocator cap and reported memory);
see ``indextts.runtime.gpu``. The check is a plain environment lookup, so the
import stays free of any framework cost when the variable is absent.
"""

import os as _os

if _os.environ.get("INDEXTTS_VRAM_EMULATE_GB", "").strip():
    from indextts.runtime.gpu import apply_emulated_vram_cap as _apply_emulated_vram_cap

    _apply_emulated_vram_cap()
