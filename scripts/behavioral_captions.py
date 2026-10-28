"""Add speaker-labelled captions to saved behavioral video evidence, without inference."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import textwrap

SPEAKERS = {'prompt': 'Prompt', 'reply': 'Model', 'review': 'Reviewer'}


def timestamp(seconds, separator=','):
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}'


def caption_cues(scenes):
    """Approximate phrase timing within each measured audio clip, not word alignment."""
    cues, offset = [], 0.0
    for item in scenes:
        duration = float(item['duration_seconds'])
        spoken = float(item.get('speech_seconds', duration))
        if not math.isfinite(duration) or not math.isfinite(spoken) or not 0 < spoken <= duration:
            raise ValueError('Invalid caption duration')
        if item['role'] == 'music':
            cues.append((offset, offset + duration, '[Music]'))
        else:
            speaker = SPEAKERS[item['role']]
            lines = textwrap.wrap(' '.join(item['narration'].split()), width=44, break_long_words=False)
            chunks = ['\n'.join(lines[i:i + 2]) for i in range(0, len(lines), 2)]
            weights = [len(chunk) for chunk in chunks]
            total = sum(weights)
            elapsed = 0.0
            for chunk, weight in zip(chunks, weights):
                end = elapsed + spoken * weight / total
                cues.append((offset + elapsed, offset + end, f'[{speaker}]\n{chunk}'))
                elapsed = end
        offset += duration
    return cues


def caption_text(cues, *, webvtt=False):
    separator = '.' if webvtt else ','
    blocks = [f'{i}\n{timestamp(start, separator)} --> {timestamp(end, separator)}\n{text}\n'
              for i, (start, end, text) in enumerate(cues, 1)]
    return ('WEBVTT\n\n' if webvtt else '') + '\n'.join(blocks)


def media_duration(path):
    return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                         '-of', 'default=noprint_wrappers=1:nokey=1', str(path)], text=True))


def music_filter(seconds):
    if not math.isfinite(seconds) or not 5 <= seconds <= 30:
        raise ValueError('Use a short outro between 5 and 30 seconds.')
    return (f'atrim=duration={seconds},asetpts=PTS-STARTPTS,apad,atrim=duration={seconds},'
            f'loudnorm=I=-25:TP=-3:LRA=7,afade=t=in:d=1,afade=t=out:st={seconds - 3}:d=3')


def append_music(video, audio, credit, seconds, work):
    """Append music, never mix it into or re-render the saved spoken scenes."""
    from PIL import Image, ImageDraw, ImageFont

    effect = music_filter(seconds)
    lines = textwrap.wrap(' '.join(credit.split()), width=65, break_long_words=False)
    if not lines or len(lines) > 6:
        raise ValueError('Provide concise music/source/license credits (up to six lines).')
    card = Image.new('RGB', (1280, 720), '#101d29')
    draw = ImageDraw.Draw(card)
    font = lambda size: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    draw.text((70, 70), 'iSPEC / Review notes', font=font(44), fill='#a4dfbe')
    draw.text((70, 160), 'Saved evidence. No new inference or live writes.', font=font(28), fill='#e4edf3')
    draw.text((70, 210), 'Failures remain test cases, not production approval.', font=font(28), fill='#e4edf3')
    draw.text((70, 300), 'MUSIC / LOCAL SYNTHESIZED EXCERPT', font=font(22), fill='#7ac8ff')
    for index, line in enumerate(lines):
        draw.text((70, 350 + index * 36), line, font=font(27), fill='#e4edf3')
    card.save(work / 'outro.png')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-loop', '1', '-framerate', '12',
                    '-i', str(work / 'outro.png'), '-i', str(audio), '-t', str(seconds),
                    '-vf', 'format=yuv420p', '-af', effect, '-ar', '48000', '-ac', '1',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'stillimage', '-threads', '2',
                    '-c:a', 'aac', '-b:a', '128k', str(work / 'outro.mp4')], check=True)
    # Fixed local link names keep concat-list quoting independent of input paths.
    (work / 'original.mp4').symlink_to(video)
    (work / 'clips.txt').write_text("file 'original.mp4'\nfile 'outro.mp4'\n")
    combined = work / 'combined.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'concat', '-safe', '1',
                    '-i', str(work / 'clips.txt'), '-c', 'copy', str(combined)], check=True)
    return combined


def add_captions(manifest, output, *, music_audio=None, music_credit=None, music_seconds=18):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if any(output.with_suffix(ext).exists() for ext in ('.mp4', '.srt', '.vtt', '.json')):
        raise ValueError('Choose a new output path; the original video is preserved.')
    if output.suffix != '.mp4':
        raise ValueError('Use an MP4 output path.')
    if bool(music_audio) != bool(music_credit):
        raise ValueError('Music requires both a local audio path and source/license credits.')
    if music_audio:
        music_filter(music_seconds)
        music_audio = Path(music_audio).resolve(strict=True)
    data = json.loads(manifest.read_text())
    scenes = [dict(item) for item in data['scenes']]
    for index, item in enumerate(scenes, 1):
        item['duration_seconds'] = media_duration(manifest.parent / f'scene-{index:02}.mp4')
        item['speech_seconds'] = media_duration(manifest.parent / f'scene-{index:02}.wav')
    cues = caption_cues(scenes)
    original = manifest.parent / 'behavior-review.mp4'
    receipt = {'video': str(output),
               'source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
               'timing': 'Measured clip boundaries; estimated phrase timing, not forced alignment',
               'inference_runs': 0}
    with tempfile.TemporaryDirectory(prefix='behavior-outro-') as temporary:
        video = original
        if music_audio:
            video = append_music(original, music_audio, music_credit, music_seconds, Path(temporary))
            start = media_duration(original)
            cues.append((start, media_duration(video), '[Music]'))
            receipt['music'] = {'credit': music_credit, 'audio_sha256': hashlib.sha256(music_audio.read_bytes()).hexdigest(),
                                'start_seconds': start, 'requested_seconds': music_seconds,
                                'placement': 'outro only; saved narration streams copied'}
        for suffix, webvtt in [('.srt', False), ('.vtt', True)]:
            with output.with_suffix(suffix).open('x') as handle:
                handle.write(caption_text(cues, webvtt=webvtt))
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-i', str(video),
                    '-i', str(output.with_suffix('.srt')), '-map', '0:v:0', '-map', '0:a:0', '-map', '1:0',
                    '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text', '-metadata:s:s:0', 'language=eng',
                    '-metadata:s:s:0', 'title=Speaker-labelled captions', '-disposition:s:0', 'default',
                    '-movflags', '+faststart', str(output)], check=True)
    receipt['cues'] = len(cues)
    with output.with_suffix('.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--music-audio', type=Path, help='Optional already-rendered local WAV/audio for the outro only')
    parser.add_argument('--music-credit', help='Composer, source and license credit shown on the outro card')
    parser.add_argument('--music-seconds', type=float, default=18)
    args = parser.parse_args()
    print(json.dumps(add_captions(args.manifest, args.output, music_audio=args.music_audio,
                                 music_credit=args.music_credit, music_seconds=args.music_seconds)))


if __name__ == '__main__':
    main()
