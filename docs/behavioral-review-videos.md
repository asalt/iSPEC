# Behavioral review videos

These are local review artifacts, not live conversations or new model benchmarks.
Preserve the recorded reply and tool outcome, including failures. Human review of
private or rewritten dialogue is required before sharing. Never present rewritten
dialogue as a literal recording.

## Saved evidence and captions

`scripts/render_behavioral_replay.py` renders the saved October 25 synthetic
approval/read-only trials. It expects the local evidence paths named in the script;
those ignored files are not bundled in a production checkout. It uses installed
Pillow and ffmpeg/libflite, with different voices for Prompt, Model and Reviewer.
It records source hashes, narration and scene durations in `replay.json`.

```sh
python scripts/render_behavioral_replay.py --output tests/behavioral/local/new-replay
python scripts/behavioral_captions.py tests/behavioral/local/new-replay/replay.json \
  --output tests/behavioral/local/new-replay/behavior-review-captioned.mp4
```

The captioner copies the original audio/video streams, adds a selectable English
subtitle track, and writes SRT/VTT sidecars plus a small provenance receipt. Each
cue has a speaker label. Phrase timing is estimated within measured scene/audio
boundaries, not word-level forced alignment. Use a new output name on every run;
existing movies and sidecars are not overwritten.

## Optional music

Music is opt-in and appended only after the last spoken scene. It is never mixed
under the review. Supply an existing local WAV/audio file and appropriate credits:

```sh
python scripts/behavioral_captions.py tests/behavioral/local/new-replay/replay.json \
  --output tests/behavioral/local/new-replay/behavior-review-music.mp4 \
  --music-audio LOCAL-RENDERED-MUSIC.wav --music-seconds 18 \
  --music-credit 'Composer; piece; source/renderer; applicable license'
```

The 5-30 second outro has a credits card, a `[Music]` cue, quiet normalization and
fade-in/out. It expects the renderer's 1280x720, 12-fps H.264 / mono 48-kHz AAC
format. Dependencies are used only when explicitly running these scripts; no video,
speech or music service is started by iSPEC. No renderer is installed automatically.

The first local example uses J. S. Bach's Fugue in C major, BWV 846, synthesized
with TiMidity from the existing LilyPond MIDI. The engraving is by Knute Snortum,
`github.com/ksnortum/bach-well-tempered-1`, CC BY-SA 4.0. Its credits identify the
source and faded excerpt. Keep the source/license information with any shared
adaptation. Music files, videos and raw evidence remain ignored local artifacts;
the small renderer and tests can ship with the rest of the source on amms06.

Before sharing, check a rendered credits frame, subtitle labels, audio level and
the original audio/video packet prefix. A music edit must not silently change
the saved behavioral evidence. Model outcomes are not re-scored by rendering.
