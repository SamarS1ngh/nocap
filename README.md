# nocap

On-device Android notification triage. Learns what you find important from how you interact with notifications, then quietly hides the junk.

Builds a hybrid classifier — a small neural net + kNN over a memory of past notifications — that runs entirely on the phone after a brief cold-start where a cloud LLM fills the gap.

For the deep architecture rationale see [`DESIGN.md`](DESIGN.md).

## Status

- 4 Kotlin modules wired into the app + the existing Compose UI
- 27 unit tests across encoder tokenizer / vector store / online head — green
- TFLite model + vocab not committed; produced via `tools/convert_minilm.py`
- APK builds to ~100MB once the model asset is in place
- No releases yet — install via `:app:installDebug` from this repo

## Module layout

```
nocap/
├── app/                       adapter — listener, Compose UI, Gemini fallback
├── embedding-engine/          MiniLM-L6-v2 TFLite + WordPiece tokenizer
├── vector-store/              Room + brute-force cosine kNN
├── online-head/               two-layer net, manual backprop, structured features
├── hybrid-predictor/          orchestrator — kNN ⊕ head ⊕ LLM fallback
├── tools/                     dev-only — convert_minilm.py
└── DESIGN.md                  architecture spec + plain-language walkthrough
```

## Prerequisites

- JDK 17
- Android SDK with API 35 platform
- A phone with USB debugging on, or an Android emulator
- Python 3.10 / 3.11 (one-time, for the model conversion script)
- (Optional) an OpenAI or Gemini API key for the cold-start / fallback path

## First build

```bash
git clone <repo-url>
cd nocap
./gradlew :app:assembleDebug
```

That builds an APK that works with the LLM fallback only — hybrid is dormant until the TFLite asset lands.

## Produce the TFLite model

The MiniLM weights aren't redistributed in this repo (90 MB, upstream owns them). Convert once:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r tools/requirements.txt
python tools/convert_minilm.py
```

That writes:

- `embedding-engine/src/main/assets/minilm.tflite` (~90 MB)
- `embedding-engine/src/main/assets/vocab.txt` (~230 KB)

Then rebuild and install — see next section.

The script verifies the produced graph signature before exiting. See `tools/README.md` for troubleshooting.

## Install on a phone

```bash
adb devices                   # confirm phone is connected + authorized
./gradlew :app:installDebug
```

After install:

1. Open **Settings → Notifications → Notification access** and toggle **nocap** on.
2. Open the app → **Settings** → paste an **OpenAI** key (primary) or **Gemini** key (fallback). At least one is needed for the cold-start phase.
3. (Optional) Toggle **Filtering enabled** and pick a threshold.

## Run on an emulator

The full pipeline works on the emulator too. Notification sources are limited compared to a real phone — you can post test notifications via `adb`:

```bash
adb shell cmd notification post -S bigtext -t "Test" tag "hello body"
```

## Using the app

| Action | Effect |
|---|---|
| Notification arrives | Captured + classified (hybrid → LLM → failed) |
| Tap notification in shade | Implicit label = WANT, head updates, store grows |
| Swipe notification away within 5s | Implicit label = SKIP, head updates, store grows |
| Tap "Want" / "Skip" in app | Explicit override, marked as manual |
| Threshold slider in Settings | Anything below the threshold gets hidden from the shade if filtering is on |
| α slider in Settings | Bias the blend toward kNN memory vs the learned head |
| **Diag** in top bar | Per-app stats, head loss chart, disagreement log |
| Failed-to-classify card | Three buttons — Important / Neutral / Junk — for manual override |

The predictor moves through phases automatically as the vector store grows:

| Store size | Behavior |
|---|---|
| < 30 | LLM only (cold start) |
| 30 – 500 | kNN primary, head suppressed |
| 500 – 2 000 | Blend, LLM tiebreaks on disagreement |
| 2 000 – 5 000 | Blend (mature), LLM rare |
| 5 000 + | Head dominant, LLM mostly idle |

## Running tests

```bash
./gradlew testDebugUnitTest
```

Coverage:

| Module | What's tested |
|---|---|
| `embedding-engine` | WordPiece tokenizer — specials, lowercase, ## continuations, UNK, truncation, padding |
| `vector-store` | Cosine identity / opposite / orthogonal / zero, normalized fast path; FloatArray ↔ ByteArray round-trip |
| `online-head` | Forward output range, learning direction, XOR convergence, save/load round-trip, shape mismatch; structured features dimensionality + content flags + cyclic hour encoding |

The TFLite engine itself is not unit-tested on the JVM — the model file is needed and TFLite is native. Smoke-test on a device after producing the model.

## Settings + Diagnostics — at a glance

### Settings tab

- API keys (OpenAI primary, Gemini fallback)
- Filtering on/off + threshold slider
- "Show filtered in list" toggle (audit trail)
- Hybrid mode toggle (defaults on; flip off for LLM-only A/B)
- α slider for the kNN/head blend
- **Reset learning** — wipes the vector store and deletes head weights

### Diag tab

- **Predictor health** — phase, store size, total captured, α, head update count, recent loss, LLM call count, disagreement rate
- **Head loss chart** — line graph of the last 1 000 update losses
- **Per-app stats** — counts, hidden %, average importance, ✓/✗ tallies
- **Disagreement log** — predictions where kNN and head disagreed by > 0.4 and the LLM was called as a tiebreaker

## Privacy

- Everything except the optional cold-start / disagreement LLM call runs on-device.
- API keys live in DataStore on the phone. The app never sends them anywhere except the OpenAI / Gemini endpoints you explicitly configured.
- Notification content is stored in the on-device SQLite DB (Room). It does not leave the device unless an LLM call needs it.
- Reset learning wipes the vector store and head weights. Captured notifications remain in the DB (so you keep your history) but the model starts fresh.

## Roadmap

Tracked in [`DESIGN.md`](DESIGN.md). The hybrid pipeline is feature-complete; the open items are tuning + ergonomics:

- Tune α empirically against observed accuracy
- Asset Pack delivery so the APK stays small for distribution
- Sender-aware structured features (currently a 4-bucket hash)
- Notification action buttons (Want / Skip) from the shade itself
- Diagnostics: rolling accuracy estimate per source

## License

MIT — see [`LICENSE`](LICENSE).
