"""Universal, typed presets shared by every UI tab.

The registry is deliberately independent from Gradio.  Unit tests and command-line
tools can therefore validate defaults and migrations without constructing a UI.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

from indextts.utils.atomic_json import read_json_retry, replace_with_retry


PRESET_FORMAT = "indextts2_premium_universal"
PRESET_VERSION = 2
SYSTEM_PREFIX = "★ "


@dataclass(slots=True)
class ControlSpec:
    key: str
    default: Any
    kind: str = "auto"
    component: Any = None
    preset: bool = True
    choices: tuple[Any, ...] = ()
    minimum: float | int | None = None
    maximum: float | int | None = None
    nullable: bool = False


# Gradio can dispatch preset events concurrently during startup and from multiple
# browser sessions. Serialize these tiny transactions so their files stay coherent.
_PRESET_IO_LOCK = threading.RLock()


class PresetRegistry:
    """Ordered registry of controls and their coercion contract."""

    def __init__(self) -> None:
        self._specs: OrderedDict[str, ControlSpec] = OrderedDict()

    def register(
        self,
        key: str,
        component: Any = None,
        default: Any = None,
        *,
        kind: str = "auto",
        preset: bool = True,
        choices: Sequence[Any] | None = None,
        minimum: float | int | None = None,
        maximum: float | int | None = None,
        min: float | int | None = None,
        max: float | int | None = None,
        nullable: bool = False,
    ) -> Any:
        normalized = str(key or "").strip()
        if not normalized:
            raise ValueError("Preset registry keys cannot be blank")
        if normalized in self._specs:
            raise ValueError(f"Duplicate preset registry key: {normalized}")
        if minimum is None:
            minimum = min
        if maximum is None:
            maximum = max
        if default is None and component is not None and hasattr(component, "value"):
            default = component.value
        self._specs[normalized] = ControlSpec(
            key=normalized,
            default=_json_value(default),
            kind=str(kind or "auto"),
            component=component,
            preset=bool(preset),
            choices=tuple(choices or ()),
            minimum=minimum,
            maximum=maximum,
            nullable=bool(nullable),
        )
        return component

    @property
    def specs(self) -> tuple[ControlSpec, ...]:
        return tuple(self._specs.values())

    @property
    def entries(self) -> tuple[ControlSpec, ...]:
        return self.specs

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._specs)

    @property
    def components(self) -> list[Any]:
        return [spec.component for spec in self._specs.values() if spec.preset and spec.component is not None]

    @property
    def component_specs(self) -> list[ControlSpec]:
        return [spec for spec in self._specs.values() if spec.preset and spec.component is not None]

    def __contains__(self, key: str) -> bool:
        return key in self._specs

    def __getitem__(self, key: str) -> ControlSpec:
        return self._specs[key]

    def __len__(self) -> int:
        return len(self._specs)

    def defaults(self, *, preset_only: bool = True) -> dict[str, Any]:
        return {
            key: _json_value(spec.default)
            for key, spec in self._specs.items()
            if not preset_only or spec.preset
        }

    def values_from_sequence(self, values: Sequence[Any]) -> dict[str, Any]:
        specs = self.component_specs
        return {spec.key: value for spec, value in zip(specs, values)}

    def coerce(self, values: Mapping[str, Any] | None) -> dict[str, Any]:
        source = dict(values or {})
        result: dict[str, Any] = {}
        for key, spec in self._specs.items():
            if not spec.preset:
                continue
            result[key] = coerce_value(source.get(key, spec.default), spec)
        return result


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "y"}:
            return True
        if normalized in {"0", "false", "no", "off", "n", ""}:
            return False
        return default
    if value is None:
        return default
    return bool(value)


def _number(value: Any, default: float, integer: bool) -> float | int:
    try:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        result = float(default)
    return int(result) if integer else result


def coerce_value(value: Any, spec: ControlSpec) -> Any:
    if spec.nullable and (value is None or value == ""):
        return None
    kind = spec.kind
    default = spec.default
    if kind == "auto":
        if isinstance(default, bool):
            kind = "bool"
        elif isinstance(default, int) and not isinstance(default, bool):
            kind = "int"
        elif isinstance(default, float):
            kind = "float"
        elif isinstance(default, (list, tuple)):
            kind = "list"
        elif isinstance(default, dict):
            kind = "dict"
        else:
            kind = "str"
    if kind == "bool":
        return _bool(value, bool(default))
    if kind in {"int", "float"}:
        normalized = _number(value, default or 0, kind == "int")
        if spec.minimum is not None:
            normalized = max(normalized, spec.minimum)
        if spec.maximum is not None:
            normalized = min(normalized, spec.maximum)
        return int(normalized) if kind == "int" else float(normalized)
    if kind == "choice":
        normalized = value
        allowed = list(spec.choices)
        if normalized in allowed:
            return normalized
        text_matches = [choice for choice in allowed if str(choice) == str(normalized)]
        return text_matches[0] if text_matches else _json_value(default)
    if kind in {"list", "multiselect"}:
        if isinstance(value, str):
            values = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            values = list(default or [])
        if spec.choices:
            values = [item for item in values if item in spec.choices]
        return [_json_value(item) for item in values]
    if kind == "dict":
        return _json_value(value) if isinstance(value, Mapping) else _json_value(default or {})
    if value is None:
        return str(default or "")
    return str(value)


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_. -]+")


def sanitize_preset_name(name: str) -> str:
    value = str(name or "").strip()
    if value.startswith(SYSTEM_PREFIX):
        value = value[len(SYSTEM_PREFIX):]
    value = _SAFE_NAME_RE.sub("_", value).strip(" ._")
    if not value:
        raise ValueError("Enter a preset name")
    return value[:80]


LEGACY_KEY_MAP = {
    "language": "generation.language",
    "autoregressive_batch_size": "generation.section_batch_size",
    "output_filename": "generation.output_filename",
    "save_used_audio": "generation.save_used_audio",
    "use_subprocess_system": "generation.use_subprocess",
    "emo_control_method": "generation.emotion_mode",
    "emo_random": "generation.emotion_random",
    "emo_text": "generation.emotion_text",
    "emo_weight": "generation.emotion_weight",
    "diffusion_steps": "generation.diffusion_steps",
    "inference_cfg_rate": "generation.inference_cfg_rate",
    "max_speaker_audio_length": "generation.max_speaker_audio_length",
    "max_emotion_audio_length": "generation.max_emotion_audio_length",
    "do_sample": "generation.do_sample",
    "temperature": "generation.temperature",
    "num_beams": "generation.num_beams",
    "max_text_tokens_per_segment": "generation.max_text_tokens_per_segment",
    "save_as_mp3": "generation.save_as_mp3",
    "low_memory_mode": "generation.low_memory_mode",
    "prevent_vram_accumulation": "generation.prevent_vram_accumulation",
    "mp3_bitrate": "generation.mp3_bitrate",
    "latent_multiplier": "generation.latent_multiplier",
    "top_p": "generation.top_p",
    "top_k": "generation.top_k",
    "repetition_penalty": "generation.repetition_penalty",
    "length_penalty": "generation.length_penalty",
    "max_consecutive_silence": "generation.max_consecutive_silence",
    "interval_silence": "generation.interval_silence",
    "apply_emo_bias": "generation.apply_emotion_bias",
    "max_emotion_sum": "generation.max_emotion_sum",
    "max_mel_tokens": "generation.max_mel_tokens",
    "semantic_layer": "generation.semantic_layer",
    "cfm_cache_length": "generation.cfm_cache_length",
}
for _index, _name in enumerate(("joy", "anger", "sad", "fear", "disgust", "depression", "surprise", "calm"), start=1):
    LEGACY_KEY_MAP[f"vec{_index}"] = f"generation.emotion_{_name}"
    LEGACY_KEY_MAP[f"emo_bias_{_name}"] = f"generation.emotion_bias_{_name}"


class PresetStore:
    """Filesystem-backed preset store with read-only system presets."""

    def __init__(
        self,
        registry: PresetRegistry | str | os.PathLike[str],
        root: str | os.PathLike[str] | PresetRegistry = "presets",
    ) -> None:
        # Accept both PresetStore(registry, root) and PresetStore(root, registry).
        if isinstance(registry, PresetRegistry):
            self.registry = registry
            self.root = Path(root) if not isinstance(root, PresetRegistry) else Path("presets")
        elif isinstance(root, PresetRegistry):
            self.registry = root
            self.root = Path(registry)
        else:
            raise TypeError("PresetStore requires a PresetRegistry")
        self.root = self.root.expanduser().resolve()
        self.system_dir = self.root / "system"
        self.user_dir = self.root / "user"
        self.last_used_path = self.user_dir / ".last_used_preset.txt"
        self._last_used_memory: str | None = None
        self.system_dir.mkdir(parents=True, exist_ok=True)
        self.user_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str, *, system: bool) -> Path:
        clean = sanitize_preset_name(name)
        return (self.system_dir if system else self.user_dir) / f"{clean}.json"

    def _system_names(self) -> list[str]:
        names = sorted(path.stem for path in self.system_dir.glob("*.json") if path.is_file())
        return (["default"] if "default" in names else []) + [name for name in names if name != "default"]

    def _user_names(self) -> list[str]:
        protected = set(self._system_names())
        return sorted(
            path.stem
            for path in self.user_dir.glob("*.json")
            if path.is_file()
            and not path.name.startswith(".")
            and path.stem not in protected
        )

    def list_presets(self) -> list[str]:
        return [SYSTEM_PREFIX + name for name in self._system_names()] + self._user_names()

    list = list_presets

    def is_system(self, name: str) -> bool:
        clean = sanitize_preset_name(name)
        return self._path(clean, system=True).is_file()

    def _payload(self, name: str, values: Mapping[str, Any], *, system: bool) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "format": PRESET_FORMAT,
            "version": PRESET_VERSION,
            "name": sanitize_preset_name(name),
            "scope": "system" if system else "user",
            "read_only": bool(system),
        }
        if not system:
            meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        return {"_meta": meta, "values": self.registry.coerce(values)}

    @staticmethod
    def _serialize(payload: Mapping[str, Any]) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        """Durably replace a preset file without sharing a temporary filename."""

        with _PRESET_IO_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                replace_with_retry(temporary, path)
            finally:
                try:
                    temporary.unlink()
                except OSError:
                    pass

    @staticmethod
    def _read_text_retry(path: Path, *, attempts: int = 8) -> str:
        wait = 0.01
        for attempt in range(attempts):
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                if attempt >= attempts - 1:
                    raise
                time.sleep(wait)
                wait = min(0.15, wait * 1.75)
        raise OSError(f"Could not read {path}")

    def write_system(self, name: str, values: Mapping[str, Any]) -> Path:
        path = self._path(name, system=True)
        serialized = self._serialize(self._payload(name, values, system=True))
        try:
            current = path.read_text(encoding="utf-8")
        except OSError:
            current = None
        if current != serialized:
            self._write_atomic(path, serialized)
        return path

    def ensure_system_presets(self) -> None:
        defaults = self.registry.defaults()
        self.write_system("default", defaults)

        quality = dict(defaults)
        quality.update(
            {
                "generation.diffusion_steps": 40,
                "generation.num_beams": 4,
                "generation.section_batch_size": 1,
                "generation.cfm_temperature": 0.9,
                "generation.audio_tuning_preset": "bypass",
            }
        )
        self.write_system("quality", quality)

        fast = dict(defaults)
        fast.update(
            {
                "generation.diffusion_steps": 12,
                "generation.num_beams": 1,
                "generation.section_batch_size": 4,
                "generation.max_mel_tokens": 1200,
            }
        )
        self.write_system("fast", fast)

        low = dict(defaults)
        low.update(
            {
                "runtime.vram_tier": "8",
                "runtime.model_variant": "int8_convrot",
                "runtime.blocks_to_swap": 8,
                "runtime.swap_ring_size": 2,
                "runtime.aux_residency.semantic_model": "on_demand",
                "runtime.aux_residency.qwen_emo": "on_demand",
                "runtime.cfm_cache_length": 4096,
                "runtime.max_section_batch_size_hint": 2,
                "generation.section_batch_size": 2,
                "generation.num_beams": 2,
                "generation.max_text_tokens_per_segment": 80,
                "generation.cfm_cache_length": 4096,
                "generation.low_memory_mode": True,
            }
        )
        self.write_system("low_vram_8gb", low)

    def save(self, name: str, values: Mapping[str, Any] | Sequence[Any]) -> str:
        with _PRESET_IO_LOCK:
            clean = sanitize_preset_name(name)
            if self.is_system(clean):
                raise PermissionError(f"System preset '{clean}' is read-only")
            if not isinstance(values, Mapping):
                values = self.registry.values_from_sequence(list(values))
            payload = self._payload(clean, values, system=False)
            self._write_atomic(self._path(clean, system=False), self._serialize(payload))
            self.set_last_used(clean)
            return clean

    def _read_payload(self, name: str) -> dict[str, Any] | None:
        clean = sanitize_preset_name(name)
        system_path = self._path(clean, system=True)
        path = system_path if system_path.is_file() else self._path(clean, system=False)
        missing = object()
        payload = read_json_retry(path, missing)
        if payload is missing:
            return None
        return payload if isinstance(payload, dict) else None

    def _migrate_legacy(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for section in ("audio_generation", "advanced_parameters"):
            section_values = payload.get(section)
            if not isinstance(section_values, Mapping):
                continue
            for old_key, value in section_values.items():
                target = LEGACY_KEY_MAP.get(str(old_key), str(old_key))
                candidates = (target, str(old_key), f"generation.{old_key}")
                match = next((candidate for candidate in candidates if candidate in self.registry), None)
                if match:
                    if old_key == "emo_control_method" and isinstance(value, (int, float)):
                        modes = ("Same as speaker voice", "Emotion reference audio", "Emotion vector", "Emotion text")
                        index = max(0, min(len(modes) - 1, int(value)))
                        value = modes[index]
                    values[match] = value
        return values

    def load(self, name: str | None) -> dict[str, Any]:
        with _PRESET_IO_LOCK:
            requested = sanitize_preset_name(name or "default")
            payload = self._read_payload(requested)
            if payload is None:
                return self.registry.defaults()
            if isinstance(payload.get("values"), Mapping):
                values = dict(payload["values"])
            elif payload.get("_meta", {}).get("format") == "indextts2_premium_ui" or any(
                key in payload for key in ("audio_generation", "advanced_parameters")
            ):
                values = self._migrate_legacy(payload)
            else:
                values = {key: value for key, value in payload.items() if key != "_meta"}
            result = self.registry.coerce(values)
            self.set_last_used(requested)
            return result

    load_values = load

    def component_values(self, name: str | None) -> list[Any]:
        values = self.load(name)
        return [values[spec.key] for spec in self.registry.component_specs]

    def reset(self) -> dict[str, Any]:
        with _PRESET_IO_LOCK:
            self.set_last_used("default")
            return self.registry.defaults()

    def delete(self, name: str) -> bool:
        with _PRESET_IO_LOCK:
            clean = sanitize_preset_name(name)
            if self.is_system(clean):
                raise PermissionError(f"System preset '{clean}' is read-only")
            path = self._path(clean, system=False)
            if not path.is_file():
                return False
            path.unlink()
            self.set_last_used("default")
            return True

    def set_last_used(self, name: str) -> bool:
        clean = sanitize_preset_name(name)
        with _PRESET_IO_LOCK:
            self._last_used_memory = clean
            try:
                current = self._read_text_retry(self.last_used_path).strip()
            except OSError:
                current = ""
            if current == clean:
                return True
            try:
                self._write_atomic(self.last_used_path, clean + "\n")
            except OSError as exc:
                # Loading a valid preset must not fail just because its small
                # last-used bookmark is temporarily unavailable.
                print(f">> Warning: could not persist last-used preset '{clean}': {exc}", flush=True)
                return False
            return True

    def get_last_used(self) -> str:
        with _PRESET_IO_LOCK:
            candidates = (self.last_used_path, self.user_dir / ".last_used_ui_preset.txt")
            for path in candidates:
                try:
                    value = sanitize_preset_name(self._read_text_retry(path).strip())
                except (OSError, ValueError):
                    continue
                if self._read_payload(value) is not None:
                    self._last_used_memory = value
                    return value
            if self._last_used_memory and self._read_payload(self._last_used_memory) is not None:
                return self._last_used_memory
            return "default"


__all__ = [
    "ControlSpec",
    "LEGACY_KEY_MAP",
    "PRESET_FORMAT",
    "PRESET_VERSION",
    "PresetRegistry",
    "PresetStore",
    "SYSTEM_PREFIX",
    "coerce_value",
    "sanitize_preset_name",
]
