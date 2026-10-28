import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[3] / 'scripts/behavioral_captions.py'
spec = importlib.util.spec_from_file_location('behavioral_captions', path)
captions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(captions)


def test_speaker_cues_preserve_narration_and_music_timing():
    scenes = [{'role': 'prompt', 'narration': 'Please draft a note. Do not save it.', 'duration_seconds': 5, 'speech_seconds': 4},
              {'role': 'reply', 'narration': 'This remains a draft.', 'duration_seconds': 3},
              {'role': 'music', 'duration_seconds': 4}]
    cues = captions.caption_cues(scenes)
    assert cues[0] == (0, 4, '[Prompt]\nPlease draft a note. Do not save it.')
    assert cues[1][0] == 5 and cues[1][2].startswith('[Model]')
    assert cues[2] == (8, 12, '[Music]')
    assert '00:00:00,000 --> 00:00:04,000' in captions.caption_text(cues)
    assert captions.caption_text(cues, webvtt=True).startswith('WEBVTT\n\n')


def test_long_narration_splits_without_rewriting():
    text = 'Do not turn a failed write into a successful one. ' * 5
    cues = captions.caption_cues([{'role': 'review', 'narration': text, 'duration_seconds': 20}])
    assert len(cues) > 1
    assert ' '.join(' '.join(c[2].splitlines()[1:]) for c in cues) == text.strip()
    assert all(a[1] <= b[0] for a, b in zip(cues, cues[1:]))
    assert cues[-1][1] == pytest.approx(20)


@pytest.mark.parametrize('duration', [0, -1, float('nan')])
def test_invalid_duration_rejected(duration):
    with pytest.raises(ValueError):
        captions.caption_cues([{'role': 'review', 'narration': 'Hello', 'duration_seconds': duration}])


@pytest.mark.parametrize('seconds', [0, 4, 31, float('nan'), float('inf')])
def test_outro_duration_is_bounded(seconds):
    with pytest.raises(ValueError, match='short outro'):
        captions.music_filter(seconds)


def test_outro_filter_is_quiet_and_fades():
    effect = captions.music_filter(18)
    assert 'loudnorm=I=-25:TP=-3' in effect
    assert 'afade=t=in:d=1' in effect and 'afade=t=out:st=15:d=3' in effect
    assert 'amix' not in effect


def test_music_requires_credits_before_reading_sources(tmp_path):
    with pytest.raises(ValueError, match='both'):
        captions.add_captions(tmp_path / 'missing.json', tmp_path / 'output.mp4', music_audio='audio.wav')


def test_existing_sidecars_are_not_overwritten(tmp_path):
    subtitle = tmp_path / 'output.srt'
    subtitle.write_text('Existing captions')
    with pytest.raises(ValueError, match='new output path'):
        captions.add_captions(tmp_path / 'missing.json', tmp_path / 'output.mp4')
    assert subtitle.read_text() == 'Existing captions'


def test_outro_does_not_mix_or_reencode_saved_speech(tmp_path, monkeypatch):
    pytest.importorskip('PIL')
    calls = []
    monkeypatch.setattr(captions.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    original = tmp_path / 'spoken.mp4'
    work = tmp_path / 'work'
    work.mkdir()
    result = captions.append_music(original, tmp_path / 'audio.wav', 'Synthetic test music', 18, work)
    assert len(calls) == 2
    assert str(original) not in calls[0]  # Only the new outro is encoded.
    assert calls[1][calls[1].index('-c') + 1] == 'copy'
    assert (work / 'original.mp4').readlink() == original
    assert result == work / 'combined.mp4'
