# nocap — on-device personalization architecture

Single source of truth for the hybrid on-device classifier we're building inside nocap. Reference for future projects (Jarvis voice memory, personal RAG, etc.) that need the same pattern. **Re-implement, don't import** — the pattern is portable across languages, the code is not.

## Problem statement

Single-user notification triage. Inputs: stream of `(packageName, title, body, postedAt)`. Outputs: importance score 0-10 + category, used to decide whether to cancel the system notification from the shade.

Requirements:
- Personalizes to one user's behaviour over time.
- Runs entirely on-device after initial setup.
- No per-classification cloud cost at steady state.
- Cold start works without thousands of labels (LLM fallback covers day 1).
- Learns continuously from implicit signals (clicks, swipes) and explicit signals (Want/Skip buttons).

## Plain-language walkthrough (read this first)

Skip this section if comfortable with ML. Otherwise read here before the technical spec.

### Section 1: The problem
Phone gets ~200 notifications/day. Most = junk. Some = important. Phone should learn YOUR taste and hide junk automatically. Phone doesn't know what you find important — must learn by watching you.

### Section 2: Turning words into numbers (the encoder)
Computers can't compare text directly. "Pizza order ready" vs "Your food arrived" mean the same thing, but letters look different.

Trick: convert each notification into 384 numbers. Chosen so similar-meaning sentences get similar numbers. Like GPS coordinates for meaning:
- "Pizza ready" → `[0.2, 0.8, 0.1, ...]` (food zone)
- "Food arrived" → `[0.2, 0.7, 0.1, ...]` (very close — also food zone)
- "Stock market crash" → `[0.9, 0.1, 0.5, ...]` (far away — money zone)

The converter = encoder (MiniLM-L6-v2). Pre-built. We never train it. The 384 numbers = an *embedding* or *vector*.

Module: `:embedding-engine`.

### Section 3: The memory book (vector-store)
Every notification you see → store its 384 numbers + whether you wanted it (1) or junked it (0).

```
Row 1: [0.2, 0.8, ...] | wanted=1 | "Pizza ready"
Row 2: [0.2, 0.7, ...] | wanted=1 | "Food here"
Row 3: [0.9, 0.1, ...] | wanted=0 | "Stock alert"
Row 4: [0.3, 0.8, ...] | wanted=1 | "Uber Eats arriving"
```

When new notification arrives ("Domino's delivered"):
1. Encode → `[0.2, 0.8, 0.1, ...]`
2. Look through book: which 5 past rows are MOST similar?
3. Found rows 1, 2, 4 — all food, all wanted=1
4. Vote: 3/3 say wanted → probably want this one

That = kNN (k-nearest-neighbours), k=5. "Similar" measured by cosine similarity (-1 to 1, higher = more similar). Use 0.7 cutoff for "actually similar."

Module: `:vector-store`.

**Why kNN good:** works from row 1; memorizes rare exact cases ("r/AskHistorians is the one Reddit sub I like").
**Why kNN bad alone:** slow if memory huge; only matches text similarity, misses "any notification at 3am is junk" patterns.

### Section 4: The pattern-learner (online-head)
kNN only looks at text meaning. More info available:
- What app sent it (WhatsApp? Reddit? Bank?)
- Time (3am? lunch?)
- Day of week
- Has emoji? URL? digits?

Total 51 extra numbers = structured features. Stick them next to the 384 meaning-numbers = **435 numbers** total.

Pattern-learner = tiny brain. Takes 435 numbers, outputs ONE number: "how likely is user to want this?" (0 to 1).

Shape:
```
435 inputs → 100 middle neurons → 1 output
```

Each arrow = a weight. ~43,000 weights. Start random. Get nudged better with every label.

**Why 2 layers not 1:**
- 1 layer = each input gets ONE vote. Can't say "Reddit is junk EXCEPT 9am from r/news."
- 2 layers = middle neurons combine inputs. One neuron learns "Reddit + r/funny." Another learns "WhatsApp + 3am + urgent word." Output combines them.

That combination ability is what makes it a "neural network."

**How it learns (SGD):** each click/swipe:
1. Run notification through brain → brain says "0.3 chance wanted"
2. Truth: user clicked → actual = 1.0
3. Wrong by 0.7
4. Backprop math figures out which weights pushed wrong → nudge them
5. Each nudge small (learning rate = 0.01). Thousands of examples → brain gets good.

Module: `:online-head`. "Online" = learns one example at a time, live.

**Why kNN + head together:** kNN = exact memory. Head = smooth general rules. Together = both.

### Section 5: The boss (hybrid-predictor)
```
new notification arrives
   ↓
ask kNN  → P_knn  = 0.8
ask head → P_head = 0.6
   ↓
boss blends: P_final = 0.6 × 0.8 + 0.4 × 0.6 = 0.72
   ↓
0.72 > threshold → keep notification
```

α (alpha) = 0.6 = how much to trust kNN vs head. Tunable.

**When boss panics (LLM fallback):** if kNN says 0.9 and head says 0.1 — disagree big. Call Gemini cloud LLM as tiebreaker. Costs money/latency, used rarely. Also if memory <30 entries → LLM for everything until enough data.

### Section 6: Where labels come from (no human work)
| You did | Label |
|---|---|
| Clicked notification | 1.0 |
| Swiped <5s | 0.0 |
| Tapped "Want" | 1.0 |
| Tapped "Skip" | 0.0 |

Normal phone use → ~150 labels/day → 2-3 weeks → 2000 labels → head trained well → LLM rarely needed → free + fast forever.

### Section 7: Full picture
```
notification arrives
   │
   ├─→ encoder → 384 meaning numbers
   ├─→ structured features → 51 situation numbers
   ├─→ vector-store.nearest(384 nums) → P_knn
   ├─→ online-head(435 nums) → P_head
   └─→ hybrid-predictor
         ├─ blend → P_final
         ├─ if disagree → call LLM
         └─ if P_final low → cancel notification

later, user clicks/swipes
   │
   └─→ hybrid-predictor.learn(label)
         ├─ vector-store.append(vec, label)  ← memory grows
         └─ online-head.update(vec, label)   ← weights nudge
```

### Section 8: Vocabulary
| Word | Meaning |
|---|---|
| Embedding/vector | list of 384 numbers representing meaning of text |
| Encoder | turns text into embedding |
| kNN | "find k most similar past examples" |
| Cosine similarity | how aligned 2 vectors are. 1=same, 0=unrelated |
| Neural network | layers of neurons with weights between them |
| Weight | number on an arrow inside a neural net |
| Layer | one row of neurons |
| Hidden layer | middle layer (not input, not output) |
| ReLU | `max(0, x)`. Used between layers |
| Sigmoid | squashes any number to 0-1. Used at output |
| Backprop | math for nudging weights |
| SGD | nudging algorithm |
| Online learning | learn one example at a time, live |
| Label | truth answer (wanted=1, junk=0) |
| α (alpha) | kNN vs head mixing weight |
| LLM | big cloud model (Gemini). Tiebreaker |

---

## Architecture overview

Three reusable modules + one orchestrator + one app-specific adapter.

```
┌─────────────────────────────────────────────────────────────────┐
│                       NOCAP APP (adapter)                       │
│   listener → captured row → call hybrid-predictor → cancel?     │
└────────────────────────────────────────────────────────────────┬┘
                                                                  │
                          ┌───────────────────────────────────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │  hybrid-predictor    │  ← orchestrator
              │  P_final = α·P_knn + │
              │            (1-α)·P_head
              │  + disagreement fallback to LLM
              └──┬──────────────┬────┘
                 │              │
       ┌─────────▼──┐     ┌─────▼────────┐    ┌──────────────────┐
       │ vector-    │     │  online-     │    │ embedding-engine │
       │ store      │     │  head        │    │ (encoder)        │
       │ (kNN)      │     │  (SGD)       │    │                  │
       └────────────┘     └──────────────┘    └──────────────────┘
            ▲                  ▲                      ▲
            │                  │                      │
            └──────────────────┴──────────────────────┘
                       all three share the encoder
```

### Modules

#### `:embedding-engine`
- **Purpose:** turn text → fixed-size semantic vector.
- **Implementation:** bundles MiniLM-L6-v2 TFLite (~90 MB). Runs locally on phone.
- **Interface:** `encode(text: String): FloatArray` returning 384 floats.
- **State:** stateless. Encoder weights are frozen, never updated.
- **Reuse note:** in JS swap for `@xenova/transformers` MiniLM. In Python, use `sentence-transformers`. Same conceptual interface.

#### `:vector-store`
- **Purpose:** persisted memory of every (vector, label, metadata) we've ever seen + cosine-similarity kNN search.
- **Implementation:** Room-backed FloatArray column + brute-force cosine search (sufficient up to ~100K rows).
- **Interface:**
  - `append(vector: FloatArray, label: Float, metadata: Map<String, String>)`
  - `nearest(query: FloatArray, k: Int): List<Hit>` where `Hit = (id, similarity, label, metadata)`
  - `size(): Int`
  - `prune(strategy)` — for when store grows too large.
- **State:** grows monotonically (with optional pruning).
- **Reuse note:** in JS use `hnswlib-node` or `vectra`. Same operations.

#### `:online-head`
- **Purpose:** small trainable classifier that learns smooth patterns kNN can't generalize, including non-linear feature interactions.
- **Architecture:** two-layer feed-forward network in pure Kotlin.
  - Input layer: 435 dims (384 embedding + 51 structured).
  - Hidden layer: 100 neurons + ReLU activation.
  - Output layer: 1 neuron + sigmoid.
  - Approx weight count:
    - input → hidden:   435 × 100 = 43,500
    - hidden biases:    100
    - hidden → output:  100
    - output bias:      1
    - **Total: ~43,701 weights** (rounded as "~50K" elsewhere).
- **Why two-layer (not single-layer logistic regression):** enables learning AND-patterns and conditional interactions ("Reddit AND r/funny → junk; Reddit AND r/AskHistorians → maybe useful"; "3am AND WhatsApp AND urgent-words → escalate"). Single-layer collapses each feature to one coefficient and cannot model such interactions.
- **Trainer:** manual backpropagation in pure Kotlin (forward pass → loss → gradient through output → gradient through hidden → SGD update). Learning rate ~0.01, tunable.
- **Inputs:** concatenation of embedding (384) + structured features (~51 dims — package one-hot for top 30 apps + "other", hour-of-day sin/cos, day-of-week one-hot, title/body length log-normalized, has-digit/URL/emoji flags, inline-reply available, Android importance hint, sender hash for messaging apps).
- **Outputs:** importance probability 0-1.
- **Interface:**
  - `predict(input: FloatArray): Float`
  - `update(input: FloatArray, label: Float)` — one SGD step through both layers.
  - `persist() / load()`.
- **State:** ~50K float weights, persisted to disk.
- **Reuse note:** in JS use `tensorflow.js` or pure JS matrix math.

#### `:hybrid-predictor`
- **Purpose:** orchestrate all three above. Make the final prediction.
- **Interface:**
  - `predict(text: String, structured: Map): Prediction` where `Prediction = (importance, source, neighbours, confidence)`
  - `learn(text: String, structured: Map, label: Float)` — call after each interaction.
- **Logic (predict):**
  1. `vec = embeddingEngine.encode(text)`
  2. `neighbours = vectorStore.nearest(vec, k = 5)`
  3. `p_knn = mean(neighbours.label)` — only if `neighbours.size >= 3 && top_similarity >= 0.7`
  4. `p_head = onlineHead.predict([vec, structured])`
  5. If kNN confident: `p = α·p_knn + (1-α)·p_head` (default α = 0.6)
  6. If kNN low-confidence or disagrees sharply with head: flag as uncertain, fall back to LLM
  7. Otherwise use combined p directly.
- **Logic (learn):**
  - `vectorStore.append(vec, label, metadata)`
  - `onlineHead.update([vec, structured], label)`
- **State:** stateless itself; delegates to the three modules.

## Data flow

### On notification capture (predict path)
```
notification arrives
   ↓ listener
encoder.encode(title + body) → vec
   ↓
vector-store.nearest(vec, k=5) → P_knn + neighbour list
online-head.predict([vec, structured]) → P_head
   ↓ hybrid-predictor
P_final = α·P_knn + (1-α)·P_head        (if kNN confident)
P_final = fallback to LLM               (if kNN uncertain)
   ↓
if P_final < threshold && filtering enabled:
    cancelNotification(key)
mark row with prediction, source, neighbours
```

### On user interaction (learn path)
```
user clicks notification (REASON_CLICK) → label = 1.0
user swipes <5s (REASON_CANCEL + low dwell) → label = 0.0
user manually taps Want / Skip → label = 1.0 / 0.0 (manual)
   ↓ hybrid-predictor.learn
vector-store.append(vec, label, metadata)
online-head.update([vec, structured], label)
   ↓
DB updated. Future predictions improved.
```

### Cold start strategy
Two-layer head needs more data to converge than single-layer. kNN and LLM cover the gap.

| Vector-store size | What runs primary |
|---|---|
| < 30 | LLM only (head + kNN ignored — confidence too low) |
| 30 – 500 | kNN primary, head suppressed until ~500 labels seen |
| 500 – 2000 | blend (kNN + head), LLM on disagreement |
| 2000+ | head + kNN primary, LLM only when disagreement > 0.4 |
| 5000+ | head reliably beats single-layer baseline; LLM mostly idle |

Expected wall-clock to reach 2000 labels: ~2-3 weeks of normal phone usage (assuming ~150 inferred labels/day from clicks + fast swipes). Head saturates around 5000 labels (~4-6 weeks).

## Reuse pattern

When transplanting to another project (Jarvis voice memory, personal RAG):

1. Pick a domain-appropriate encoder. Text? Reuse MiniLM. Voice? `whisper-base` for transcripts then MiniLM. Images? CLIP.
2. Decide what's an "item" and what's a "label." Notifications → importance. Voice memory → relevance to current query. Photos → "I'd save this album."
3. Structured features change per domain. Re-derive them.
4. The orchestrator logic (kNN + head + LLM fallback + disagreement) is identical across domains.
5. Module boundaries and SQL schemas are reusable as design templates.

This design holds in any language. Re-implement in target stack using native libraries (JS: transformers.js + hnswlib-node + tf.js; Python: sentence-transformers + faiss + scikit-learn; Swift: CoreML + Annoy).

## Build phases (nocap-specific)

1. **Foundation:** `:embedding-engine`, `:vector-store`. Standalone, tested. ~1.5 weeks.
2. **Retrieval-only integration:** wire into nocap, run alongside LLM, compare predictions. Validates encoder + store. ~3-4 days.
3. **Trainable two-layer head:** add `:online-head` with hidden layer. Forward pass, manual backprop, persistence, tests. ~10-12 days (longer than single-layer due to backprop math + hidden-size tuning).
4. **Hybrid orchestrator:** `:hybrid-predictor`. Replace LLM-primary. ~3-4 days.
5. **Polish:** α tuning, kNN K + similarity threshold, hidden-size tuning, learning-rate decay, store pruning, monitoring. Ongoing.

## Open decisions

- α (kNN vs head weight): start 0.6 favoring kNN, tune empirically. Shift toward 0.4 (favor head) once head trains past ~2000 labels.
- k (neighbours): start 5.
- Similarity threshold to trust kNN: start 0.7 cosine.
- **Head architecture: two-layer (435 → 100 hidden → 1) with ReLU.** Hidden size 100 is starting point; tunable in 64-256 range.
- Learning rate: start 0.01. Add decay (e.g., halve every 5000 updates) once data is plentiful.
- Activation: ReLU in hidden layer. Sigmoid on output. Standard.
- Store pruning: by age, by inverse-utility, or never? Decide at 50K+ rows.
- Disagreement threshold to trigger LLM fallback: |P_knn − P_head| > 0.4 — start there.
- When to suppress head's vote: until vector-store has ~500 labels (head considered unreliable below that).

## Non-goals

- No multi-user support. Personalization is single-user.
- No federated learning across devices.
- No cloud fine-tuning. All training is online, on-device.
- No fine-tuning the encoder itself. Frozen.
- No long-term memory of pre-classification context (notification timing patterns, app-launch correlations). Could be a Phase 6 addition.

## Failure modes to monitor

- kNN over-confidence on low-data slices (e.g., one user marks one Reddit post Want by accident → next Reddit posts get predicted Want)
- Head drift after contradictory labels (gets stuck oscillating)
- **Hidden-layer overfitting on small data:** 50K weights memorize noise if turned on too early. Mitigated by suppressing head vote until 500 labels collected, optional L2 regularization on weights.
- **Dying ReLU:** if hidden neurons saturate to 0, they stop learning. Mitigated by Leaky ReLU or careful init.
- Encoder bias on non-English text or emoji-heavy notifications
- Store bloat impacting search time (mitigated by pruning at 50K rows)
- α never gets tuned and stays at default 0.6 forever

## Glossary

- **Embedding:** fixed-size float vector that represents the meaning of a piece of text.
- **Encoder:** pretrained model that produces embeddings.
- **Cosine similarity:** number in [-1, 1] measuring angular alignment of two vectors. 1 = same direction, 0 = orthogonal, -1 = opposite.
- **kNN:** k-nearest-neighbours. Retrieval method that finds the k closest stored vectors.
- **SGD:** stochastic gradient descent. Algorithm that nudges model weights one example at a time.
- **Head:** small trainable network sitting on top of frozen encoder.
- **Hidden layer:** intermediate layer between input and output. Enables non-linear interactions between features.
- **ReLU:** Rectified Linear Unit. Activation function `max(0, x)`. Standard non-linearity for hidden layers.
- **Backprop:** backpropagation. Algorithm that computes gradients for every weight in a multi-layer network using the chain rule.
- **Dim / dimensions:** number of slots in a vector. "384-dim vector" = list of 384 numbers.
- **Online learning:** updating the model one example at a time as data arrives, vs batch training all at once.
- **α (alpha):** mixing weight in `α·P_knn + (1-α)·P_head`.
