"""Probe whether this laptop's speaker -> mic path carries near-ultrasonic tones.

Modes:
  python ultrasonic_probe.py              # tone sweep over every config in CONFIGS
  python ultrasonic_probe.py --wave 20000 # play one tone for 8 s; wave your hand to see Doppler
"""
import argparse
import threading
import time

import numpy as np
import sounddevice as sd

TONES = [1000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000]
TONE_S, GAP_S, SILENCE_S = 0.4, 0.15, 0.6
AMP = 0.25
FRAME = 4096

# (label, input device substring, output device substring, host api, input extra_settings)
CONFIGS = [
    ("WASAPI shared (Windows audio processing ON)", "Microphone Array", "Speaker (Realtek", "Windows WASAPI", None),
    ("WASAPI exclusive mic", "Microphone Array", "Speaker (Realtek", "Windows WASAPI", "excl"),
    ("MME 4ch array", "Microphone Array (Intel", "Speaker (Realtek", "MME", None),
    ("WDM-KS array 1", "Microphone Array 1", "Speakers 1", "Windows WDM-KS", None),
    ("WDM-KS array 2", "Microphone Array 2", "Speakers 1", "Windows WDM-KS", None),
    ("WDM-KS array 3 (4ch)", "Microphone Array 3", "Speakers 1", "Windows WDM-KS", None),
]


def find_device(substr, hostapi, kind):
    api_names = [h["name"] for h in sd.query_hostapis()]
    for i, d in enumerate(sd.query_devices()):
        if api_names[d["hostapi"]] != hostapi or substr not in d["name"]:
            continue
        if d[f"max_{kind}_channels"] > 0:
            return i, d
    raise LookupError(f"no {kind} device matching {substr!r} on {hostapi}")


def tone(f, sr, secs):
    t = np.arange(int(sr * secs)) / sr
    y = AMP * np.sin(2 * np.pi * f * t)
    fade = int(sr * 0.02)
    ramp = np.linspace(0, 1, fade)
    y[:fade] *= ramp
    y[-fade:] *= ramp[::-1]
    return y


def play_and_record(signal, sr, in_dev, in_ch, out_dev, extra=None):
    """Record on in_dev while playing a mono signal on out_dev (separate streams)."""
    chunks, lock = [], threading.Lock()

    def cb(indata, frames, t, status):
        with lock:
            chunks.append(indata.copy())

    out_ch = min(2, sd.query_devices(out_dev)["max_output_channels"])
    out = np.repeat(signal[:, None], out_ch, axis=1).astype(np.float32)
    settings = sd.WasapiSettings(exclusive=True) if extra == "excl" else None
    with sd.InputStream(device=in_dev, channels=in_ch, samplerate=sr, callback=cb, dtype="float32",
                        extra_settings=settings):
        time.sleep(0.2)
        sd.play(out, sr, device=out_dev, blocking=True)
        time.sleep(0.3)
    return np.concatenate(chunks)


def band_power(x, sr, f, half_bw=60):
    """Per-frame power in [f-half_bw, f+half_bw] for one channel."""
    win = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1 / sr)
    band = (freqs > f - half_bw) & (freqs < f + half_bw)
    hop = FRAME // 2
    out = []
    for s in range(0, len(x) - FRAME, hop):
        spec = np.abs(np.fft.rfft(x[s:s + FRAME] * win)) ** 2
        out.append(spec[band].sum())
    return np.array(out)


def sweep(label, in_sub, out_sub, api, extra, sr):
    print(f"\n=== {label} @ {sr} Hz ===")
    try:
        in_dev, din = find_device(in_sub, api, "input")
        out_dev, dout = find_device(out_sub, api, "output")
    except LookupError as e:
        print("  skipped:", e)
        return
    in_ch = din["max_input_channels"]
    print(f"  in : [{in_dev}] {din['name']} ({in_ch} ch)")
    print(f"  out: [{out_dev}] {dout['name']}")

    parts = [np.zeros(int(sr * SILENCE_S))]
    for f in TONES:
        parts += [tone(f, sr, TONE_S) if f < sr / 2 else np.zeros(int(sr * TONE_S)),
                  np.zeros(int(sr * GAP_S))]
    try:
        rec = play_and_record(np.concatenate(parts), sr, in_dev, in_ch, out_dev, extra)
    except Exception as e:
        print("  failed:", type(e).__name__, e)
        return

    # Noise floor: the leading silence (skip the first 0.2 s of stream warm-up).
    quiet = rec[int(sr * 0.2):int(sr * (0.2 + SILENCE_S * 0.8))]
    if not np.any(rec):
        print("  recorded all zeros (endpoint opened but delivered no audio)")
        return
    print("  per mic channel: SNR vs. silence / level relative to the 1 kHz tone (dB)")
    print("   freq  " + "".join(f"     ch{c}    " for c in range(in_ch)))
    ref = [np.max(band_power(rec[:, c], sr, 1000)) + 1e-20 for c in range(in_ch)]
    for f in TONES:
        if f >= sr / 2:
            print(f"  {f/1000:4.0f}k   above Nyquist")
            continue
        row = []
        for c in range(in_ch):
            peak = np.max(band_power(rec[:, c], sr, f))
            noise = np.median(band_power(quiet[:, c], sr, f)) + 1e-20
            row.append(f"{10 * np.log10(peak / noise):5.0f} /{10 * np.log10(peak / ref[c]):5.0f}")
        print(f"  {f/1000:4.0f}k  " + "  ".join(row))


def wave(freq, sr, cfg):
    label, in_sub, out_sub, api, extra = cfg
    in_dev, din = find_device(in_sub, api, "input")
    out_dev, _ = find_device(out_sub, api, "output")
    print(f"{label}: playing {freq} Hz for 8 s. Wave your hand toward/away from the screen.")
    rec = play_and_record(tone(freq, sr, 8.0), sr, in_dev, din["max_input_channels"], out_dev, extra)[:, 0]
    hop = sr // 20
    win = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1 / sr)
    carrier = np.abs(freqs - freq) < 25
    up = (freqs > freq + 40) & (freqs < freq + 400)
    down = (freqs < freq - 40) & (freqs > freq - 400)
    for s in range(sr // 2, len(rec) - FRAME, hop):
        spec = np.abs(np.fft.rfft(rec[s:s + FRAME] * win)) ** 2
        c = spec[carrier].sum() + 1e-20
        u = 10 * np.log10(spec[up].sum() / c + 1e-12) + 60
        d = 10 * np.log10(spec[down].sum() / c + 1e-12) + 60
        print(f"{s/sr:5.2f}s  toward {'#' * max(0, int(u)):<40} away {'#' * max(0, int(d))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sr", type=int, nargs="+", default=[48000])
    ap.add_argument("--wave", type=int, metavar="FREQ")
    ap.add_argument("--config", type=int, default=1, help="index into CONFIGS for --wave")
    a = ap.parse_args()
    if a.wave:
        wave(a.wave, a.sr[0], CONFIGS[a.config])
    else:
        for sr in a.sr:
            for cfg in CONFIGS:
                sweep(*cfg, sr)
