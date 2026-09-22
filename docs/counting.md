# Teaching it to count

A lab notebook. The model writes perfectly-formed journal entries and then
numbers them at random — `DAY 42`, `DAY 1903`, `DAY 2336`. The corpus never
states the rule, but it is always true: each day is 1 to 3 later than the
one before. Learning it means carrying a number across hundreds of
characters of unrelated text.

Eleven models. Two wrong conclusions along the way, both recorded here,
because how they went wrong is more useful than the answer.

## Experiment 1 — widen the window

The obvious explanation: the model cannot see the answer. Day headers in
`corpus/chronicles.txt` sit an average of **361 characters** apart and the
first model had a 96-character window, so the previous day number was never
in context.

Five models, identical except for the window. Same corpus, same
architecture (3 layers, 4 heads, width 128), same **6,144 tokens per step** —
batch size adjusted per window so a step is the same amount of reading in
every run, otherwise we would be comparing token budgets instead of
windows. `run_sweep.py`.

| window | sees prev. day | bits/char | forward | legal step |
|---|---|---|---|---|
| *the corpus* | — | — | 100% | 100% |
| 96 | 0.0% | 0.39 | 47% | 0% |
| 192 | 10.7% | 0.35 | 54% | 0% |
| 256 | 24.8% | 0.34 | 55% | 0% |
| 384 | 58.2% | 0.36 | 55% | 0% |
| 512 | 85.7% | 0.32 | 51% | 0% |

*forward* = is the next day later than the last (random: ~50%).
*legal step* = is it exactly 1–3 later (random: ~0%).

All five converged. All five count exactly as well as a random number
generator, including the one that can see the previous day 86% of the time,
at its full 3,000 steps. Widening the window alone does nothing.

## An interruption: the metric was lying

At this point the numbers showed a median delta of 56 — small enough that I
briefly believed the model had learned approximate magnitude even if the
arithmetic failed.

It had not. That was a **signed** median, averaging large errors in both
directions into a small-looking number. The fix was to measure `|gap|` and
add a shuffle control: take the model's own day numbers, destroy only their
ordering, and score that. It keeps the model's range and distribution and
removes exactly the thing being tested.

```
ledger-192   |gap| 3176     shuffled 2740
```

Indistinguishable. Any metric that can only flatter the model is worth this
kind of suspicion.

## Experiment 2 — a denser corpus

Next: maybe the digits are too rare to be worth learning. In the
chronicles, day digits are ~1.1% of all characters, and a training step at
window 512 contains only ~17 day headers, so there are few examples of the
task and little loss to be saved by getting it right.

So, a second corpus — `LEDGER`, one line per day (`make_corpus.py --only
ledger`). Headers **97 characters apart** instead of 361, digits **4.2%** of
the text instead of 1.1%, and ~63 headers per training step instead of 17.

## Experiment 3 — the conclusion I got wrong

I checked `ledger-192` at step 500 (0–1% legal), checked a deterministic-step
variant at step 1000 (39% legal), and concluded that the random 1–3 step was
the blocker — that ambiguity capped the reward at `log(3)` bits while the
mechanism stayed expensive to learn.

**That was comparing two checkpoints at different training steps and
reading a cause into the difference.** Trained to the same 2,000 steps, the
random-step model does just as well:

| corpus | step rule | window | sees prev. day | legal | \|gap\| | shuffled |
|---|---|---|---|---|---|---|
| ledger | random 1–3 | 64 | 0.2% | 0% | 3594 | 3476 |
| ledger | random 1–3 | **192** | 100% | **85%** | **2** | 137 |
| ledger1 | +1 | 64 | 0.2% | 0% | 1403 | 1546 |
| ledger1 | +1 | **192** | 100% | **88%** | **1** | 190 |

85% versus 88%. Determinism is worth almost nothing. What separates the
rows is the window — and counting appeared somewhere between step 500 and
step 2000, which is why checking early produced a confident wrong answer.

Here it is, mid-count:

```
DAY 572 -- the long pier
Bastian the cartographer waited at the long pier until the lamps came on.
DAY 573 -- the orchard of bells
Sable walked to the winter archive without telling anyone.
DAY 574 -- the salt gardens
```

## Where that leaves it

Two things are both necessary, and neither is sufficient:

- **The number has to be in the window.** Window 64 scores 0% on the dense
  corpus, where everything else is favorable.
- **The task has to come up often enough.** Window 512 scores 0% on the
  chronicles at 86% visibility and a full 3,000 steps, while window 192
  scores 85% on the ledger at 100% visibility. The ledger shows the model
  ~4x more day headers per step, at a quarter of the retrieval distance.

Those two corpora differ in visibility *and* in density, so the comparison
between them was still confounded. `ledger-128` separates them: the dense
corpus at 94.7% visibility, which is close to what the failing chronicles
model had.

| corpus | headers apart | window | visibility | legal | \|gap\| | shuffled |
|---|---|---|---|---|---|---|
| chronicles | 361 | 512 | 85.7% | 0% | 1068 | 896 |
| ledger | 97 | 64 | 0.2% | 0% | 3594 | 3476 |
| ledger | 97 | 128 | 94.7% | **19%** | **8** | 3431 |
| ledger | 97 | 192 | 100% | **85%** | **2** | 137 |

Both factors are real, and they are separable:

- **Density matters.** At roughly the same visibility, the dense corpus
  counts (19%) and the sparse one does not (0%). The ledger shows the model
  ~63 day headers per training step against the chronicles' ~17, at a
  quarter of the retrieval distance.
- **Visibility has to be near-total, not merely high.** Going from 94.7% to
  100% takes the score from 19% to 85%. Those cases where the number has
  fallen out of the window do not just contribute nothing — they appear to
  actively wash out the rule, since the model is being trained to produce a
  day number from a context that does not contain the evidence for it.

The second point is the more interesting one, and it is the reverse of how
I framed this at the start. I treated the window as a threshold: below it
the model is blind, above it the model can see. It behaves much more like a
dose.

One run remains: `count-768` puts the chronicles at 99.8% visibility while
leaving density low. If near-total visibility is sufficient on its own it
should count; if density is genuinely required it should not.

## What I would take from it

Three explanations were available — it cannot see, it does not care, it
cannot compute — and they are distinguishable by experiment rather than by
argument. I was confident about the first, published it in the README as
settled, and it was wrong. Then I was confident about the third on the
strength of two checkpoints that had not had the same amount of training,
and that was wrong too.

The thing that caught both was the same: keep running the experiment past
the point where it starts agreeing with you.

## Reproduce it

```bash
python run_sweep.py                        # experiment 1 (5 models)
python corpus/make_corpus.py --only ledger --out corpus/ledger.txt
python corpus/make_corpus.py --only ledger --increment one --out corpus/ledger1.txt
python eval_counting.py checkpoints/count-*.npz   --corpus corpus/chronicles.txt
python eval_counting.py checkpoints/ledger*.npz   --corpus corpus/ledger.txt
```

## Caveats, stated rather than buried

- Equal tokens per step means unequal batch sizes (64 sequences at window
  96, 12 at window 512), so gradient noise differs across the sweep.
  Equalizing both is impossible; equal reading seemed the more honest choice
  for a question about context.
- The three narrowest sweep models were stopped at 2,000–2,500 steps once
  their loss had plateaued and their counting score was flat at zero. The
  384 and 512 runs — where the effect was predicted — got the full 3,000.
- All numbers are at temperature 0.8, top-k 20, over 8 samples per model.
  Counting is somewhat better at lower temperature and worse at higher.
