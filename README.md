# Tapify

Control a laptop by tapping the desk beside it, sensed with nothing but the laptop's
built-in microphones. A tap to the left or right of the laptop is detected, grouped into
single / double / triple patterns, and labelled with the side it came from.

```
  tap  peak_db=39.6  centroid_hz=134.7  ...  side_db=-1.5  side=left
  => single tap (left)
  tap  peak_db=38.1  centroid_hz=284.9  ...  side_db=2.0   side=right
  => single tap (right)
```

Developed and measured on a Windows 11 laptop with an Intel Smart Sound two-mic array.

## Usage

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

.venv/Scripts/python tap_detect.py                # listen; tap the desk, Ctrl+C to stop
.venv/Scripts/python tap_detect.py --calibrate    # guided recording of labelled taps -> data/
.venv/Scripts/python calib_analyze.py data/calib_<timestamp>.npz   # compare side-detection settings
.venv/Scripts/python -m pytest                    # offline tests, no mic needed
```

`--thresh <dB>` sets how far above the background a tap must be (default 32).

## How it works

1. **Raw capture.** The mics are read through WDM-KS, which bypasses Windows' shared-mode
   audio processing. In shared mode, noise suppression gates the background to near zero
   and filters out everything above speech frequencies.
2. **Onset detection in the tap band.** A desk tap is a low thump (centroid ~130–300 Hz).
   Onsets are detected only in 60–2000 Hz, against a noise floor that is the rolling median
   of the last second. The room's steady 13–14 kHz clicks (keyboard, fan, coil noise) never
   trigger, and room thumps at 23–28 dB stay under the 32 dB threshold.
3. **Grouping.** Taps less than 0.45 s apart form one pattern. Re-triggers within 0.12 s of
   a tap are treated as the same tap ringing.
4. **Side.** The tap's side comes from which mic hears the first 5 ms of the tap louder, in the
   tap band. If that difference is within ±0.3 dB, or a pattern mixes sides, the pattern is
   ignored: a wrongly fired command is worse than a missed one.

## What the measurements showed

**Timing between the mics fails close to the laptop.** The two mics are about 6 cm
apart, so the largest possible arrival-time difference is about 9 samples at 48 kHz. Taps far
out to the side gave a clean ±6–9 samples. Taps right next to the laptop gave −1 to +3 on
*both* sides. Desk vibration enters through the laptop's feet and reaches both mics through the
chassis at nearly the same time.

**Which mic hears the tap louder still separates them.** On a labelled calibration run
(`data/calib_20261027_203700.npz`, 32 taps), 36 timing and loudness settings were compared.
Loudness in the first 5 ms, 60–2000 Hz, was the only one to get every tap right:

| Section | Level difference, ch1 − ch0 (dB) |
|---|---|
| close-left | −0.46 … −1.57 |
| close-right | +0.49 … +1.83 |
| far-left | −1.44 … −1.82 |
| far-right | +0.41 … +1.37 |

The current timing method scored 62% on the close taps. A follow-up live session got 11/11
close taps right, with the smallest difference at 0.6 dB.

The cutoff depends on where the laptop sits on the desk. After moving it, re-run
`--calibrate` and check the summary it prints.

## Repository

| Path | |
|---|---|
| `tap_detect.py` | Detector, live listener, calibration mode, self-test |
| `calib_analyze.py` | Ranks side-detection settings on a calibration recording |
| `tools/ultrasonic_probe.py` | Measures how well the speaker-to-mic path carries 15–23 kHz tones, a first step toward sonar hand gestures |
| `data/room_noise.npy` | 5 s of real background audio from this laptop's mics, used by the self-test |
| `data/calib_20261027_203700.npz` | Labelled calibration taps (close/far, left/right) |
| `tests/` | Self-test, the calibration recording as a regression check, timing-estimate check |

## Next

- Map tap patterns to actions (media keys, virtual-desktop switching).
- Ultrasonic sonar for in-air gestures. With processing bypassed, the probe measured
  60–70 dB SNR at 15–19 kHz and a sharp drop above 20 kHz on this machine.
