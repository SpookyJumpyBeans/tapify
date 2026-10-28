"""Compare side-detection settings on a calibration recording from tap_detect.py --calibrate.

  python calib_analyze.py data/calib_YYYYMMDD_HHMMSS.npz

For each timing setting (frequency band, window, sub-sample interpolation) it prints the
measured lag of every tap per section, then how well a single cutoff separates left from
right: on the close taps alone and on all taps.
"""
import sys

import numpy as np

from tap_detect import EVENT, MAX_LAG, PRE

UP = 16   # sub-sample interpolation factor


def gcc_lag(a, b, sr, band, up=1):
    """Lag of a behind b in samples (fractional when up > 1), GCC-PHAT restricted to band."""
    n = 2 * len(a)
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    f = np.fft.rfftfreq(n, 1 / sr)
    R = A * np.conj(B)
    R = np.where((f >= band[0]) & (f <= band[1]), R / (np.abs(R) + 1e-12), 0)
    cc = np.fft.irfft(R, n * up)            # zero-padding the spectrum interpolates in time
    m = MAX_LAG * up
    cc = np.concatenate([cc[-m:], cc[:m + 1]])
    return (int(np.argmax(cc)) - m) / up


def level_db(a, b, sr, band):
    f = np.fft.rfftfreq(len(a), 1 / sr)
    sel = (f >= band[0]) & (f <= band[1])
    pa = (np.abs(np.fft.rfft(a)) ** 2)[sel].sum()
    pb = (np.abs(np.fft.rfft(b)) ** 2)[sel].sum()
    return 10 * np.log10(pa / (pb + 1e-20) + 1e-20)


def separation(left, right):
    """Best single-cutoff accuracy (left below, right above) and d' (gap / spread)."""
    vals = np.concatenate([left, right])
    best = 0.0
    for c in np.unique(vals):
        for cut in (c - 1e-6, c + 1e-6):
            ok = (left < cut).sum() + (right > cut).sum()
            best = max(best, ok / len(vals))
    spread = np.sqrt((np.var(left) + np.var(right)) / 2) + 1e-9
    return best, float(np.clip((np.mean(right) - np.mean(left)) / spread, -99, 99))


def main(path):
    d = np.load(path)
    clips, labels, sr = d["clips"], d["labels"], int(d["sr"])
    start = clips.shape[1] - EVENT + PRE          # onset index (clips are end-aligned)
    sections = list(dict.fromkeys(labels))
    print(f"{len(clips)} taps, sections: {', '.join(f'{s} ({(labels == s).sum()})' for s in sections)}\n")

    windows = {"onset+21ms": (start - PRE, start - PRE + 1024),
               "pre-roll 10ms + 21ms": (start - PRE - 480, start - PRE + 1024),
               "onset+5ms": (start - PRE, start - PRE + 480)}
    bands = [(60, 2000), (200, 4000), (200, 8000), (1000, 12000), (2000, 16000)]

    rows = []
    for wname, (w0, w1) in windows.items():
        for band in bands:
            for up in (1, UP):
                lags = np.array([gcc_lag(c[w0:w1, 0], c[w0:w1, 1], sr, band, up) for c in clips])
                rows.append((f"lag  {band[0]}-{band[1]} Hz, {wname}, {'sub-sample' if up > 1 else 'whole'}",
                             lags))
        for band in [(60, 2000), (2000, 16000)]:
            lv = np.array([level_db(c[w0:w1, 0], c[w0:w1, 1], sr, band) for c in clips])
            # negated to ch1 - ch0, the same sign as tap_detect's side_db (negative = left)
            rows.append((f"level {band[0]}-{band[1]} Hz, {wname} (dB, ch1 - ch0)", -lv))

    results = []
    for name, v in rows:
        close = separation(v[labels == "close-left"], v[labels == "close-right"])
        left = np.isin(labels, ["close-left", "far-left"])
        right = np.isin(labels, ["close-right", "far-right"])
        allsep = separation(v[left], v[right])
        results.append((close[0], close[1], allsep[0], name, v))

    results.sort(key=lambda r: (-r[0], -abs(r[1])))
    print("ranked by close-tap accuracy (single best cutoff; d' = mean gap / spread, |d'| > 2 is clean):")
    print(f"  {'close acc':>9} {'close d':>8} {'all acc':>8}  setting")
    for acc, dp, aacc, name, _ in results:
        print(f"  {acc:9.0%} {dp:8.2f} {aacc:8.0%}  {name}")

    print("\nper-tap values for the current setting and the top 3:")
    current = next(r for r in results if r[3].startswith("lag  60-2000 Hz, onset+21ms, whole"))
    for acc, dp, aacc, name, v in [current] + [r for r in results[:3] if r is not current]:
        print(f"\n  {name}  (close acc {acc:.0%})")
        for s in sections:
            print(f"    {s:12s} " + " ".join(f"{x:+5.1f}" for x in v[labels == s]))


if __name__ == "__main__":
    main(sys.argv[1])
