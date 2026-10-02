"""AuK editing, enhancement and separation tasks: instructions and target durations.

Instructions are upstream's templates verbatim (``pe.config.yaml`` and ``docs/COOKBOOK.md``);
the model was trained on these phrasings, and rewording them makes it read the
instruction aloud. Durations follow the Prompt Enhancer's rules. Plain Python: the
interface imports this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SPEEDS = (0.5, 0.75, 1.25, 1.5, 2.0)
SEMITONES = (1, 2, 3)
GAINS_DB = (5, 10, 15)
EMOTIONS = ("happy", "angry", "sad", "fearful", "surprised", "disgusted", "calm", "excited")
EMOTION_DURATION = {"sad": 1.22, "fearful": 1.16}
EMOTION_DURATION_DEFAULT = 1.06
# The 13 nonverbal events with English names (the other 26 exist only in Chinese).
NONVERBAL_EVENTS = ("breath", "laugh", "sigh", "throat clearing", "cough", "sniff", "crying", "gasp", "yawn",
                    "sneeze", "moan", "snore", "grunt")
# (keywords, seconds added, seconds removed) by sound family, first match wins.
NONVERBAL_DURATION = (
    (("breath", "breathing", "pant", "inhale", "exhale"), 0.35, 0.6),
    (("tsk", "smack", "sniff", "gasp"), 0.5, 1.0),
    (("laugh", "laughter", "chuckle", "sigh", "cough", "throat", "clearing"), 0.75, 1.05),
)
NONVERBAL_DURATION_DEFAULT = (0.55, 0.9)
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth")
EFFECTS = {
    "telephone": "This speech carries telephone-band coloration. Please restore it to a natural wideband voice.",
    "megaphone": "This audio has a megaphone-like coloration. Please restore it to a natural-sounding voice.",
    "muffled": "This audio sounds underwater / muffled. Please restore it to a normal, clear-sounding voice.",
    "clipping": "This audio is clipped. Please declip it and restore the speech to a natural-looking waveform.",
    "dropouts": "This audio has audible packet dropouts or short cut-offs. Please fill in the missing segments, "
                "outputting a continuous, intact voice.",
    "dc offset": "This audio has a DC offset. Please remove the DC component and output a properly-centered clean voice.",
    "other coloration": "Please restore the audio quality by removing the recording/coloration artifacts, and output "
                        "a natural, clear-sounding wideband voice.",
}
ENHANCE = {
    "noise and reverberation": "Please clean up this input speech and keep all speakers intact, while removing the "
                               "background noise and room reverberation. Output the restored clean speech with the same "
                               "length as the input.",
    "noise only": "Please remove only the background noise from this audio while preserving the original room "
                  "reverberation and any colorations. Output a denoised speech of the same length as the input.",
    "reverberation only": "Please remove only the room reverberation from this audio while preserving the original "
                          "background noise and other colorations. Output a dereverberated speech of the same length "
                          "as the input.",
    "noise, reverberation and distortion": "Please clean up this input speech and keep all speakers intact, while "
                                           "removing the background noise, room reverberation, and various distortions. "
                                           "Output the restored clean speech with the same length as the input.",
    "limited bandwidth": "This audio suffers from limited bandwidth. Please restore it to a wideband, clear-sounding speech.",
}
MUSIC = {
    "singing voice only": "Keep only the singing voice, drop everything else.",
    "all human voices": "Keep all human voices, speech and singing alike, drop everything else.",
}
# The Prompt Enhancer sends these two tasks in Chinese for every request language.
DEACCENT_ZH = "请去掉这段语音里的方言口音，保持说话人音色一致。"
WHISPER_TO_ZH = "用小声耳语的方式把这段话说出来。"
WHISPER_FROM_ZH = "把这段耳语转换成正常说话的声音。"


@dataclass(frozen=True)
class Task:
    key: str
    label: str
    fields: tuple[str, ...] = ()
    trim: bool = True  # trimmed to the speech span (0.1 s margins) before editing
    duration: str = "equal"  # equal | content | speed | emotion | nonverbal | full
    chunkable: bool = False  # same-length task that may run on long audio piece by piece
    notes: str = ""
    extra: dict = field(default_factory=dict)


TASKS: dict[str, Task] = {task.key: task for task in (
    Task("replace", "Content · replace words", ("orig", "new"), duration="content",
         notes="The words must appear exactly as spoken."),
    Task("insert_before", "Content · insert words before", ("text", "anchor"), duration="content"),
    Task("insert_after", "Content · insert words after", ("text", "anchor"), duration="content"),
    Task("remove", "Content · remove words", ("target",), duration="content"),
    Task("remove_before", "Content · remove words before", ("target", "anchor"), duration="content"),
    Task("remove_after", "Content · remove words after", ("target", "anchor"), duration="content"),
    Task("lyrics", "Lyrics · change sung words", ("orig", "new"), trim=False, duration="content",
         notes="Use an a cappella recording (vocals only); separate the vocals first if needed."),
    Task("pitch", "Acoustic · pitch", ("direction", "semitones"), chunkable=True),
    Task("speed", "Acoustic · speed", ("speed",), duration="speed"),
    Task("volume", "Acoustic · volume", ("direction", "gain_db"), chunkable=True),
    Task("emotion", "Paralinguistic · emotion", ("emotion",), duration="emotion", chunkable=True),
    Task("timbre", "Paralinguistic · timbre (voice conversion)", ("description",), chunkable=True),
    Task("deaccent", "Paralinguistic · remove accent", chunkable=True,
         notes="Trained on Chinese dialects; the instruction is sent in Chinese."),
    Task("nonverbal_remove", "Paralinguistic · remove a sound (breath, laugh, ...)", ("sound", "anchor"), trim=False,
         duration="nonverbal", notes="Leave the anchor blank to remove every occurrence."),
    Task("nonverbal_add", "Paralinguistic · add a sound", ("sound", "position", "anchor"), trim=False,
         duration="nonverbal"),
    Task("to_whisper", "Paralinguistic · speech to whisper", chunkable=True),
    Task("from_whisper", "Paralinguistic · whisper to speech", trim=False, duration="full", chunkable=True),
    Task("enhance", "Enhance · clean up speech", ("enhance",), trim=False, chunkable=True),
    Task("restore", "Enhance · repair an effect", ("effect",), trim=False, chunkable=True),
    Task("separate_order", "Separate · keep one speaker by talking order", ("speaker",), trim=False),
    Task("separate_content", "Separate · keep the speaker who says ...", ("spoken",), trim=False),
    Task("music", "Separate · vocals from music", ("music",), trim=False, chunkable=True),
    Task("custom", "Custom instruction", ("instruction",), trim=False,
         notes="Write the instruction yourself; set a duration unless the output keeps the source length."),
)}

FIELD_DEFAULTS = {
    "orig": "", "new": "", "text": "", "anchor": "", "target": "", "direction": "raise", "semitones": 2,
    "speed": 1.25, "gain_db": 5, "emotion": "happy", "description": "a deep, calm male voice", "sound": "laugh",
    "position": "after", "speaker": 1, "spoken": "", "enhance": "noise and reverberation", "effect": "telephone",
    "music": "singing voice only", "instruction": "",
}


def _quoted(value: str) -> str:
    return str(value or "").strip().replace("'", "’")


def render_instruction(task_key: str, values: dict, language: str = "en") -> str:
    """The instruction sent to AuK for one task and its field values."""
    task = TASKS[task_key]
    v = {**FIELD_DEFAULTS, **{key: value for key, value in (values or {}).items() if value is not None}}
    zh = str(language).lower() == "zh"
    missing = [name for name in task.fields if name in {"orig", "new", "text", "target", "spoken", "description",
                                                        "instruction"} and not str(v[name]).strip()]
    if task_key in {"remove_before", "remove_after", "insert_before", "insert_after"} and not str(v["anchor"]).strip():
        missing.append("anchor")
    if missing:
        raise ValueError("Fill in: " + ", ".join(sorted(set(missing))))
    if task_key == "replace":
        return f"把‘{_quoted(v['orig'])}’改成‘{_quoted(v['new'])}’" if zh else \
            f"Replace '{_quoted(v['orig'])}' with '{_quoted(v['new'])}'."
    if task_key == "insert_before":
        return f"在‘{_quoted(v['anchor'])}’前面加上‘{_quoted(v['text'])}’" if zh else \
            f"Add '{_quoted(v['text'])}' before '{_quoted(v['anchor'])}'."
    if task_key == "insert_after":
        return f"在‘{_quoted(v['anchor'])}’后面加上‘{_quoted(v['text'])}’" if zh else \
            f"Add '{_quoted(v['text'])}' after '{_quoted(v['anchor'])}'."
    if task_key == "remove":
        return f"删掉‘{_quoted(v['target'])}’" if zh else f"Remove '{_quoted(v['target'])}'."
    if task_key in {"remove_before", "remove_after"}:
        side = "before" if task_key == "remove_before" else "after"
        if zh:
            return f"删掉‘{_quoted(v['anchor'])}’{'前' if side == 'before' else '后'}面的‘{_quoted(v['target'])}’"
        return f"Remove '{_quoted(v['target'])}' {side} '{_quoted(v['anchor'])}'."
    if task_key == "lyrics":
        return f"把这段歌词中的“{v['orig'].strip()}”改成“{v['new'].strip()}”。" if zh else \
            f'Change "{v["orig"].strip()}" to "{v["new"].strip()}" in the vocal recording.'
    if task_key == "pitch":
        n = int(v["semitones"])
        word = "semitone" if n == 1 else "semitones"
        return f"{'Raise' if v['direction'] in {'raise', 'increase'} else 'Lower'} the pitch by {n} {word}."
    if task_key == "speed":
        return f"Adjust the speech speed to {float(v['speed']):g}x."
    if task_key == "volume":
        return f"{'Increase' if v['direction'] in {'raise', 'increase'} else 'Decrease'} the volume by {int(v['gain_db'])} dB."
    if task_key == "emotion":
        return f"Change the emotion to {v['emotion']}."
    if task_key == "timbre":
        return f'Keep the spoken content unchanged and change the timbre to: "{v["description"].strip()}".'
    if task_key == "deaccent":
        return DEACCENT_ZH
    if task_key == "nonverbal_remove":
        anchor = str(v["anchor"]).strip()
        return f'Remove the {v["sound"]} after "{anchor}" from the speech.' if anchor else \
            f"Remove all the {v['sound']} from the audio."
    if task_key == "nonverbal_add":
        position, anchor = v["position"], str(v["anchor"]).strip()
        if position in {"before", "after"}:
            if not anchor:
                raise ValueError("Fill in: anchor")
            return f'Add a {v["sound"]} {position} "{anchor}".'
        return f"Add a {v['sound']} at {'the beginning' if position == 'beginning' else 'the end'} of the speech."
    if task_key == "to_whisper":
        return WHISPER_TO_ZH
    if task_key == "from_whisper":
        return WHISPER_FROM_ZH
    if task_key == "enhance":
        return ENHANCE[v["enhance"]]
    if task_key == "restore":
        return EFFECTS[v["effect"]]
    if task_key == "separate_order":
        index = max(1, int(v["speaker"]))
        ordinal = ORDINALS[index - 1] if index <= len(ORDINALS) else f"{index}th"
        return f"Please keep the {ordinal} speaker to start talking and remove the other speakers, outputting a single clean speech track."
    if task_key == "separate_content":
        return (f'Please keep only the speaker who says "{v["spoken"].strip()}" in this input speech, remove the other '
                "speakers, and output a single clean speech track.")
    if task_key == "music":
        return MUSIC[v["music"]]
    return str(v["instruction"]).strip()


def spoken_seconds(text: str) -> float:
    """The Prompt Enhancer's content-edit pace: 0.21 s per CJK character, 0.30 s per English word."""
    text = str(text or "")
    cjk = len(re.findall(r"[㐀-䶿一-鿿]", text))
    words = len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?", text))
    if cjk or words:
        return 0.21 * cjk + 0.30 * words
    return 0.25 * len(text.split())


def target_seconds(task_key: str, values: dict, base_seconds: float, full_seconds: float,
                   transcript: str | None = None) -> float:
    """Output length for a task (seconds); ``base_seconds`` is the trimmed speech span when the task trims."""
    task = TASKS[task_key]
    v = {**FIELD_DEFAULTS, **(values or {})}
    base = float(base_seconds)
    if task.duration == "full":
        return float(full_seconds)
    if task.duration == "speed":
        return base / float(v["speed"])
    if task.duration == "emotion":
        return base * EMOTION_DURATION.get(str(v["emotion"]), EMOTION_DURATION_DEFAULT)
    if task.duration == "nonverbal":
        sound = str(v["sound"]).lower()
        add, remove = next(((a, r) for words, a, r in NONVERBAL_DURATION if any(w in sound for w in words)),
                           NONVERBAL_DURATION_DEFAULT)
        return max(0.1, base + (add if task_key == "nonverbal_add" else -remove))
    if task.duration == "content":
        added = {"replace": v["new"], "lyrics": v["new"], "insert_before": v["text"], "insert_after": v["text"]}.get(task_key, "")
        removed = {"replace": v["orig"], "lyrics": v["orig"], "remove": v["target"], "remove_before": v["target"],
                   "remove_after": v["target"]}.get(task_key, "")
        if transcript and spoken_seconds(transcript) > 0:
            source = spoken_seconds(transcript)
            return base * max(0.05, source + spoken_seconds(added) - spoken_seconds(removed)) / source
        if task_key in {"replace", "lyrics"} and spoken_seconds(removed) > 0:
            return base * spoken_seconds(added) / spoken_seconds(removed)
        return base
    return base
