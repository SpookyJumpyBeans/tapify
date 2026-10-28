"""Passive tap detector: listens on the raw mic array and reports tap patterns.

  python tap_detect.py                 # live: tap the desk beside the laptop, Ctrl+C to stop
  python tap_detect.py --log taps.csv  # append every tap's features to a CSV
  python tap_detect.py --calibrate     # guided: record labelled close/far left/right taps to data/
  python tap_detect.py --selftest      # offline: real room noise + synthetic thumps, no speaker
"""
import argparse
import collections
import csv
import os
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi

SR = 48000
HOP = 256                 # ~5.3 ms analysis step
PRE = 240                 # samples kept before the onset (~5 ms)
EVENT = 2048              # samples analysed per event (~43 ms)
THRESH_DB = 32            # onset must exceed the in-band noise floor by this much
                          # (room thumps on this laptop measured 23-28 dB)
RISE = 4.0                # ...and jump this many times over the previous hop
REFRACTORY_S = 0.12       # ignore re-triggers from the same tap ringing (0.08 let one through)
GROUP_GAP_S = 0.45        # taps closer than this belong to one pattern
WARMUP_S = 1.0            # stream start-up pops land in here
MAX_LAG = 20              # cross-correlation search, samples (~0.4 ms, > mic spacing)
# Side comes from which mic hears the first 5 ms of the tap louder, in the tap band. Timing
# between the mics failed for desk taps close to the laptop (the desk/chassis path reaches
# both mics almost together); this got 32/32 on data/calib_20261027_203700.npz:
# left taps -0.5..-1.8 dB, right taps +0.4..+1.8 dB.
SIDE_WIN = 480            # first 5 ms pre-roll + 5 ms after onset
SIDE_DB = 0.3             # |level difference| below this is too close to call: ignored
BUF_MAX = 2 * EVENT + PRE + 4 * HOP   # room for a full event after its onset
CLIP_PRE = 1024           # extra pre-roll saved with calibration clips (fits in BUF_MAX)

# A knuckle or fingertip on the desk is a low thump; room clicks (typing, fan, coil
# noise) on this laptop sit around 13-14 kHz with nothing below 1 kHz. Onsets are
# detected only in this band, so those clicks never trigger.
BAND_HZ = (60, 2000)
BAND_SOS = butter(4, BAND_HZ, btype="bandpass", fs=SR, output="sos")

IN_DEVICE = ("Microphone Array 1", "Windows WDM-KS")   # raw path: no Windows noise suppression
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NOISE_FILE = os.path.join(DATA_DIR, "room_noise.npy")   # 5 s of this laptop's mics, nobody tapping


def find(substr, api, kind):
    apis = [h["name"] for h in sd.query_hostapis()]
    for i, d in enumerate(sd.query_devices()):
        if apis[d["hostapi"]] == api and substr in d["name"] and d[f"max_{kind}_channels"] > 0:
            return i, d
    raise LookupError(f"no {kind} device {substr!r} on {api}")


def features(x, floor_e):
    """x: (EVENT, ch) starting just before the onset."""
    mono = x.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1 / SR)
    keep = freqs > 40                      # drop DC / handling rumble
    spec, freqs = spec[keep], freqs[keep]
    total = spec.sum() + 1e-20
    e = np.array([np.mean(mono[i:i + 64] ** 2) for i in range(0, len(mono) - 64, 64)])
    peak = e.max() + 1e-20
    base = np.median(e[-8:]) * 2           # tail level, so room noise can't pin decay at the window
    after = np.nonzero(e[e.argmax():] < max(peak / 100, base))[0]   # 20 dB decay

    f = {
        "peak_db": 10 * np.log10(peak / floor_e),
        "centroid_hz": float((freqs * spec).sum() / total),
        "low_frac": float(spec[freqs < 1000].sum() / total),
        "decay_ms": (after[0] if len(after) else len(e)) * 64 / SR * 1000,
    }
    if x.shape[1] >= 2:
        a, b = x[:1024, 0], x[:1024, 1]
        f["lag_samples"] = gcc_phat_lag(a, b)
        f["side_db"] = early_level_db(x[:SIDE_WIN])
    return f


def early_level_db(x):
    """ch1 minus ch0 energy in the tap band, dB. Negative = left on this laptop."""
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    sel = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    p = (np.abs(np.fft.rfft(x, axis=0)) ** 2)[sel].sum(axis=0)
    return float(10 * np.log10(p[1] / (p[0] + 1e-20) + 1e-20))


def gcc_phat_lag(a, b):
    """Samples by which a lags b (positive = a later), via GCC-PHAT in the tap band.
    Plain cross-correlation over a short window of a decaying tap is biased toward the
    loud early part and drifted to the search edge on synthetic taps."""
    n = 2 * len(a)
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    R = A * np.conj(B)
    R = np.where((freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1]), R / (np.abs(R) + 1e-12), 0)
    cc = np.fft.irfft(R, n)
    cc = np.concatenate([cc[-MAX_LAG:], cc[:MAX_LAG + 1]])
    return int(np.argmax(cc)) - MAX_LAG


def side(f):
    """'left', 'right', or None when the tap is too close to the middle to call."""
    db = f.get("side_db", 0.0)
    return "left" if db <= -SIDE_DB else "right" if db >= SIDE_DB else None


class Detector:
    """Feed it audio blocks; it calls on_event(features) and on_pattern(n_taps, side).
    side is None when the pattern should be ignored (an ambiguous tap, or mixed sides)."""

    def __init__(self, ch, on_event=None, on_pattern=None, thresh_db=THRESH_DB):
        self.thresh_db = thresh_db
        self.zi = sosfilt_zi(BAND_SOS)[:, :, None] * np.zeros((1, 1, ch))
        self.buf = np.zeros((0, ch), np.float32)
        self.recent = collections.deque(maxlen=SR // HOP)
        self.prev_e = 1e-20
        self.samples = 0
        self.last_onset = -1.0
        self.last_tap = -1.0
        self.pending = None
        self.group = []
        self.on_event = on_event or (lambda f: None)
        self.on_pattern = on_pattern or (lambda n, s: None)

    def feed(self, block):
        for s in range(0, len(block), HOP):
            self._hop(block[s:s + HOP])

    def finish(self):
        self._flush()

    def _hop(self, block):
        buf = np.concatenate([self.buf, block])
        drop = max(0, len(buf) - BUF_MAX)
        self.buf = buf[drop:]
        if self.pending is not None:
            self.pending -= drop
        self.samples += len(block)
        now = self.samples / SR   # audio time, immune to processing jitter

        band, self.zi = sosfilt(BAND_SOS, block, axis=0, zi=self.zi)
        e = float(np.mean(band.mean(axis=1) ** 2)) + 1e-20
        # Floor = median of the last second of hops; a 40 ms tap can't move a 1 s median.
        # Skip the first 0.2 s: the stream starts with zeros.
        if now > 0.2:
            self.recent.append(e)
        floor_e = float(np.median(self.recent)) if self.recent else e
        if (self.pending is None and e > floor_e * 10 ** (self.thresh_db / 10) and e > self.prev_e * RISE
                and now - self.last_onset > REFRACTORY_S and now > WARMUP_S):
            self.last_onset = now
            self.pending = len(self.buf) - len(block) - PRE   # where the event starts in buf
            self.pending_floor = floor_e
        self.prev_e = e

        if self.pending is not None and len(self.buf) - self.pending >= EVENT:
            f = features(self.buf[self.pending:self.pending + EVENT], self.pending_floor)
            # Raw audio for calibration: the event plus extra pre-roll, in case the onset
            # (found in the low band) trails the tap's sharper high-frequency start.
            self.last_clip = self.buf[max(0, self.pending - CLIP_PRE):self.pending + EVENT].copy()
            f["t"] = self.last_onset
            f["side"] = side(f)
            self.pending = None
            self.on_event(f)
            self.group.append(f["side"])
            self.last_tap = self.last_onset

        if self.group and self.pending is None and now - self.last_tap > GROUP_GAP_S:
            self._flush()

    def _flush(self):
        if self.group:
            # A misfired command is worse than a missed one: every tap must agree on a side.
            sides = set(self.group)
            self.on_pattern(len(self.group), sides.pop() if len(sides) == 1 else None)
            self.group = []


def fmt(f):
    return "  ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in f.items())


def name(n):
    return {1: "single", 2: "double", 3: "triple"}.get(n, f"{n}x")


def open_mic():
    """Returns (stream, queue of blocks, channels). WDM-KS opens the mic exclusively."""
    in_dev, din = find(*IN_DEVICE, "input")
    ch = min(2, din["max_input_channels"])
    q = queue.Queue()
    try:
        stream = sd.InputStream(device=in_dev, channels=ch, samplerate=SR, blocksize=HOP,
                                dtype="float32", callback=lambda d, *_: q.put(d.copy()))
    except sd.PortAudioError as e:
        # A second listener gets "Invalid device".
        raise SystemExit(f"could not open the mic ({e}).\n"
                         "Another program has it open exclusively - most likely an earlier "
                         "tap_detect.py still running in another terminal. Stop it with Ctrl+C.")
    print(f"listening on [{in_dev}] {din['name']} ({ch} ch).")
    return stream, q, ch


def mic_blocks(q):
    while True:
        try:
            yield q.get(timeout=0.1)
        except queue.Empty:
            pass


CALIB_SECTIONS = [
    ("close-left", "on the desk RIGHT NEXT TO the LEFT side of the laptop"),
    ("close-right", "on the desk RIGHT NEXT TO the RIGHT side of the laptop"),
    ("far-left", "on the desk about 30 cm (a forearm) LEFT of the laptop"),
    ("far-right", "on the desk about 30 cm (a forearm) RIGHT of the laptop"),
]
CALIB_TAPS = 8
ENTER_GUARD_S = 0.6       # the Enter keypress itself thumps the desk; skip taps right after it


def calibrate(blocks, ch, wait_enter, out_path, thresh_db=THRESH_DB):
    """Guided recording of labelled taps. wait_enter(prompt) returns a threading.Event that
    is set once the user is ready; blocks yields audio blocks forever."""
    label, start_t = [None], [0.0]
    clips, labels, side_dbs = [], [], []

    def on_event(f):
        if label[0] is None or f["t"] < start_t[0] + ENTER_GUARD_S:
            return
        clips.append(det.last_clip)
        labels.append(label[0])
        side_dbs.append(f["side_db"])
        print(f"    {labels.count(label[0])}/{CALIB_TAPS}  side_db={f['side_db']:+.2f}  "
              f"({f['side'] or 'unclear'})")

    det = Detector(ch, on_event, None, thresh_db)
    for i, (key, where) in enumerate(CALIB_SECTIONS, 1):
        print(f"\n[{i}/{len(CALIB_SECTIONS)}] Tap {CALIB_TAPS} times {where}, about 1 s apart.")
        ready = wait_enter("    Press Enter to start this section...")
        label[0] = None
        for block in blocks:
            det.feed(block)
            if label[0] is None and ready.is_set():
                label[0], start_t[0] = key, det.samples / SR
            if labels.count(key) >= CALIB_TAPS:
                break
    label[0] = None

    n = max(len(c) for c in clips)
    padded = np.zeros((len(clips), n, ch), np.float32)
    for i, c in enumerate(clips):
        padded[i, n - len(c):] = c        # align on the end: the event sits at the tail
    np.savez(out_path, clips=padded, labels=np.array(labels), side_db=np.array(side_dbs), sr=SR)
    print(f"\nsaved {len(clips)} taps to {out_path}")
    print(f"current detector, side_db per section (left <= -{SIDE_DB}, right >= +{SIDE_DB}):")
    for key, _ in CALIB_SECTIONS:
        ls = [v for v, lab in zip(side_dbs, labels) if lab == key]
        right = sum(side({"side_db": v}) == key.split("-")[1] for v in ls)
        print(f"  {key:12s} {' '.join(f'{v:+.2f}' for v in ls)}   correct: {right}/{len(ls)}")


def enter_event(prompt):
    ev = threading.Event()
    threading.Thread(target=lambda: (input(prompt), ev.set()), daemon=True).start()
    return ev


def live(log_path, thresh_db=THRESH_DB):
    writer = csv.writer(open(log_path, "a", newline="")) if log_path else None

    def on_event(f):
        print("  tap  " + fmt(f))
        if writer:
            writer.writerow([f"{time.time():.3f}"] + [f"{v:.3f}" if isinstance(v, float) else v
                                                      for v in f.values()])

    def on_pattern(n, s):
        print(f"  => {name(n)} tap ({s})" if s else f"  => {name(n)} tap ignored: side unclear")

    stream, q, ch = open_mic()
    det = Detector(ch, on_event, on_pattern, thresh_db)
    print("Tap the desk beside the laptop; Ctrl+C to stop.")
    with stream:
        try:
            for block in mic_blocks(q):
                det.feed(block)
        except KeyboardInterrupt:
            det.finish()


def thump(amp, ch, lag=0, db=0.0):
    """Synthetic knuckle tap: a damped 180 Hz + 900 Hz ring with a fast attack.
    lag > 0 delays channel 0, lag < 0 delays channel 1; channel 1 is db louder than channel 0
    (negative db reads as a left-side tap)."""
    t = np.arange(int(SR * 0.04)) / SR
    y = amp * np.exp(-t / 0.006) * (np.sin(2 * np.pi * 180 * t) + 0.5 * np.sin(2 * np.pi * 900 * t))
    out = np.zeros((len(y) + abs(lag), ch), np.float32)
    out[max(lag, 0):max(lag, 0) + len(y), 0] = y
    out[max(-lag, 0):max(-lag, 0) + len(y), 1:] = y[:, None] * 10 ** (db / 20)
    return out


def selftest():
    noise = np.load(NOISE_FILE)          # ~5 s of this laptop's real room noise, clicks included
    ch = noise.shape[1]
    sig = np.concatenate([noise[:SR], noise, noise])   # 1 s warm-up + ~10 s, all real room noise
    band_rms = np.sqrt(np.mean(sosfilt(BAND_SOS, noise[:, 0]) ** 2))
    # Per-tap (lag, dB) like the calibration run: close taps have ~0 lag and only a level
    # difference; the weakest real ones were -0.46 / +0.41 dB. Expected side is None where
    # the pattern must be ignored (a middle tap, or mixed sides).
    L, R, L_close, R_close, mid = (-8, -1.6), (6, 1.1), (0, -0.46), (2, 0.41), (0, 0.0)
    plan = [(2.0, [L], "left"), (3.5, [R_close, R], "right"), (5.0, [mid, mid, mid], None),
            (6.5, [L_close, R_close], None), (8.0, [L, mid], None), (9.0, [L_close], "left"),
            (10.0, [R], "right")]
    for start, taps, _ in plan:
        for k, (lag, db) in enumerate(taps):
            i = int((start + 0.18 * k) * SR)
            th = thump(band_rms * 300, ch, lag, db)   # ~50 dB over the in-band room noise
            sig[i:i + len(th)] += th
    seen = []
    det = Detector(ch, on_pattern=lambda n, s: seen.append((n, s)))
    for s in range(0, len(sig), 1024):
        det.feed(sig[s:s + 1024])
    det.finish()
    want = [(len(taps), where) for _, taps, where in plan]
    print("selftest:", "PASS" if seen == want else "FAIL", "patterns seen:", seen, "expected:", want)
    return seen == want


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log")
    ap.add_argument("--thresh", type=float, default=THRESH_DB, help="dB over noise floor to count a tap")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="guided recording of labelled taps")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if a.calibrate:
        out = os.path.join(DATA_DIR, time.strftime("calib_%Y%m%d_%H%M%S.npz"))
        stream, q, ch = open_mic()
        with stream:
            calibrate(mic_blocks(q), ch, enter_event, out, a.thresh)
        raise SystemExit(0)
    live(a.log, a.thresh)
