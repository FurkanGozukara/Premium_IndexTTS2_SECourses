"""Transcript agreement that trusts the speaker's own subtitles for names and terms."""
from indextts.training.dataset_quality import transcript_vocabulary
from indextts.training.speech_metrics import lenient_units, transcript_metrics


def test_split_compounds_and_contractions_are_not_errors():
    row = transcript_metrics("Open SwarmUI now, I'm sure it's fine, OK?", "open swarm ui now i am sure it is fine okay", "EN")
    assert row["errors"] == 0
    assert row["start_matches"] and row["end_matches"]
    assert transcript_metrics("Save the data set now.", "save the dataset now", "EN")["errors"] == 0


def test_project_terms_forgive_recognizer_spellings_but_ordinary_words_still_count():
    lenient = lenient_units(["RunPod", "ACEStep", "Massed", "Compute", "42"], "EN")
    assert {"runpod", "acestep", "massed", "compute"} <= set(lenient)
    assert "forty" not in lenient  # numbers are ordinary vocabulary

    forgiving = transcript_metrics("With RunPod you filter by CUDA.", "with rumpod you filter by cuda", "EN", lenient_terms=lenient)
    assert forgiving["errors"] == 0 and forgiving["forgiven_units"] == 1 and forgiving["start_matches"]
    strict = transcript_metrics("With RunPod you filter by CUDA.", "with rumpod you filter by cuda", "EN")
    assert strict["errors"] == 1 and strict["forgiven_units"] == 0
    assert strict["start_matches"]  # a substituted edge word is counted, but only missing/extra edge words fail the edge check

    split = transcript_metrics("Made by the ACEStep developers.", "made by the eight step developers", "EN", lenient_terms=lenient)
    assert split["errors"] == 0 and split["end_matches"]
    two_words = transcript_metrics("Massed Compute is fast.", "mass compute is fast", "EN", lenient_terms=lenient)
    assert two_words["errors"] == 0 and two_words["start_matches"]

    ordinary = transcript_metrics("Set the models here.", "set the modules here", "EN", lenient_terms=lenient)
    assert ordinary["errors"] == 1 and ordinary["forgiven_units"] == 0
    unrelated = transcript_metrics("Open RunPod now.", "open windows now", "EN", lenient_terms=lenient)
    assert unrelated["errors"] == 1  # a term replaced by a different word is still an error


def test_edges_fail_on_missing_or_extra_words_but_not_on_substituted_words():
    substituted = transcript_metrics("what generation uses.", "what generation use", "EN")
    assert substituted["errors"] == 1 and substituted["end_matches"]  # counted, but the acoustic edge check guards cuts
    split = transcript_metrics("Okay, selected; everything is ready.", "ok select it everything is ready", "EN")
    assert split["start_matches"] and split["errors"] == 2
    assert not transcript_metrics("I will hit generate.", "i will hit generate okay", "EN")["end_matches"]
    assert not transcript_metrics("You can also auto improve.", "can also auto improve", "EN")["start_matches"]
    assert not transcript_metrics("Set our accurate template.", "through set our accurate template", "EN")["start_matches"]
    complete = transcript_metrics("You can also auto improve.", "you can also auto improve", "EN")
    assert complete["start_matches"] and complete["end_matches"]
    empty = transcript_metrics("Hello there.", "", "EN")
    assert empty["error_rate"] == 1.0 and not empty["start_matches"] and not empty["end_matches"]


def test_spoken_forms_of_currency_units_and_decimals_match():
    assert transcript_metrics("For example L40S it is 61 cents.", "for example l40s it is $0.61", "EN")["errors"] == 0
    assert transcript_metrics("It costs $2.50 per hour.", "it costs 2 dollars 50 cents per hour", "EN")["errors"] == 0
    assert transcript_metrics("With as low as 6 GB GPUs.", "with as low as 6 gigabytes gpus", "EN")["errors"] == 0
    assert transcript_metrics("ACEStep 1.5 works.", "acestep one point five works", "EN")["errors"] == 0
    assert transcript_metrics("Wait 300 ms here.", "wait 300 milliseconds here", "EN")["errors"] == 0


def test_music_bleed_insertions_remain_errors_even_with_lenient_terms():
    lenient = lenient_units(["Lego"], "EN")
    row = transcript_metrics("Lego is adding a new stem.", "lego is adding a new stem she was more like a beauty queen on a movie scene", "EN",
                             lenient_terms=lenient)
    assert row["error_rate"] > 0.15
    assert not row["end_matches"]


def test_vocabulary_comes_from_the_transcripts_themselves():
    texts = [
        "So we open SwarmUI and ComfyUI.",
        "Then SwarmUI uses CUDA 13 with bf16 and ComfyUI.",
        "The RunPod template. RunPod is fast, I'm sure.",
        "Use Qwen here and Qwen there, Qwen again.",
        "I'm here. So the I'm test with CUDA.",
    ]
    vocabulary = transcript_vocabulary(texts)
    assert vocabulary[0] == "Qwen"  # most frequent first
    assert {"SwarmUI", "ComfyUI", "RunPod", "CUDA", "Qwen"} <= set(vocabulary)
    assert not {"So", "The", "Then", "Use", "I'm", "here"} & set(vocabulary)
    assert "bf16" not in vocabulary  # a single occurrence is not yet a project spelling
