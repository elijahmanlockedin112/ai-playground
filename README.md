# Our Own AI Playground

A language model built from nothing but NumPy. No PyTorch, no autograd, no
pretrained weights, no downloaded corpus. Every matrix multiply, every
derivative, and every word it will ever read was made here.

It is, at the moment, extremely dumb. That is the point — we get to watch it
stop being dumb.

```
python gradcheck.py                        # prove the math is right
python corpus/make_corpus.py               # write the library it learns from
python train.py --preset tiny --steps 3000 # teach it
python server.py                           # http://localhost:8123
```

The web playground streams characters as the model produces them, so you
watch it think instead of waiting for a paragraph. The model dropdown
rescans `checkpoints/` every 15 seconds — leave it open while `train.py`
runs and new models appear in it by themselves.

Prefer a terminal? `python sample.py --interactive`, or
`python sample.py --compare` for the same prompt at five temperatures.

## Setup

```bash
pip install -r requirements.txt     # numpy and pillow. That is all.
```

Weights and corpora are **not** in the repo — they are generated, so
shipping them would be shipping the output of a seeded function. Build them:

```bash
python corpus/make_corpus.py                       # the text corpus
python train.py --preset tiny --steps 3000         # a language model (~8 min)
python train_cnn.py --steps 5000                   # the puzzle classifier (~90 min)
python server.py                                   # http://localhost:8123
```

Everything is reproducible: `make_corpus.py` is seeded, and the training
scripts take `--seed`.

## Two models live here

**A language model** (`playground/`) — a decoder-only transformer trained on
text we generate. See below.

**A convolutional net** (`vision/`) — it looks at a 64×64 image and says
whether there is a Rubik's cube in it.

```bash
python gradcheck_cnn.py                    # prove the conv gradients
python train_cnn.py --steps 2000           # ~25 minutes
python eval_cnn.py                         # what it gets wrong, with pictures
python eval_cnn.py --photos ~/my_photos    # try it on real photographs
```

They share the optimizer (`playground/optim.py`) and the same rule: no
autograd, no pretrained weights, no downloaded data.

### The cube classifier

Same principle as the corpus — we generate the images. `vision/render.py`
builds a cube in 3D, rotates it, culls the back faces, projects it, and
rasterises each visible face as a 3×3 grid of stickers with plastic showing
through the gaps. About 4 ms an image, which means training data is
effectively infinite and **every batch is rendered fresh** — there is no
training set to overfit, so train and validation loss mean the same thing.

**The negatives are the whole design.** If "not a cube" meant spheres and
noise, the model could score 99% by learning *is it boxy* and we'd have
learned nothing. So it also faces:

| distractor | why it is hard |
|---|---|
| plain box | right shape, right 3D, no stickers |
| checkerboard box | a grid on a cube — but the wrong grid |
| flat 3×3 colour grid | right stickers, no box |

Between them, those three can only be separated by actually looking at the
sticker layout on a three-dimensional body.

**And the images are deliberately ruined.** A clean synthetic render is far
easier than a photograph, and a model trained on clean renders learns "crisp
polygons on smooth background" — a feature that does not exist in real life.
So `vision/augment.py` throws away every tell of the renderer: multi-scale
textured backgrounds, other objects in shot, occluders across the subject,
defocus and motion blur, vignetting, chromatic fringing, wrong white
balance, brightness-dependent sensor noise, and a genuine JPEG round-trip.

Compare `docs/render_sheet.png` (clean) with `docs/render_hard.png` (what it
actually trains on).

**Result** — 126,242 parameters, 2,000 steps, 25 minutes, measured on 3,000
images it had never seen:

```
overall accuracy                        99.73%
spotted 1539 of 1541 cubes   recall     99.87%
cried cube at 6 non-cubes    precision  99.61%

rubik     99.9%    plain     100.0%    checker    98.3%
gridflat 100.0%    sphere    100.0%    cylinder  100.0%
```

The two shortcut-proof distractors — plain boxes (right shape, no stickers)
and flat 3×3 grids (right stickers, no shape) — are both at **100%**. It
learned the conjunction, not a shortcut.

Six of the eight remaining errors are coarse checkerboards called cubes, and
looking at them, that is close to a labelling disagreement: a 4×4 checker on
a three-dimensional box really is a grid of coloured squares on a cube face.
The other two are cubes with an occluding bar across most of the stickers,
which is arguably my augmentation destroying the evidence rather than the
model failing. `docs/cube_eval.png` shows them, mistakes first.

**One honest caveat.** 99.7% is accuracy on *our own renderer*. However hard
the augmentation, synthetic-to-real is a gap that synthetic data cannot
measure about itself. `python eval_cnn.py --photos <folder>` runs it on real
photographs, and the web playground takes drag-and-drop — that number is the
truthful one.

## What's in here

| file | what it is |
|---|---|
| `vision/render.py` | Draws the training images: cubes in 3D, plus the distractors. |
| `vision/augment.py` | Ruins them on purpose, so the renderer's tells cannot be learned. |
| `vision/layers.py` | Conv / pool / linear, with hand-derived backward. im2col inside. |
| `vision/cnn.py` | The convnet: four conv stages, then a classifier head. |
| `gradcheck_cnn.py` | Same numerical check, for the convolution gradients. |
| `train_cnn.py` | Trains the classifier on freshly rendered batches. |
| `eval_cnn.py` | Accuracy by distractor type, a contact sheet of failures, and real photos. |
| `playground/model.py` | The whole model: embeddings, causal multi-head attention, MLP blocks, LayerNorm, tied output head — forward **and** a hand-derived backward pass. |
| `playground/optim.py` | AdamW with decoupled weight decay, gradient clipping, warmup + cosine LR. |
| `playground/tokenizer.py` | Character-level tokenizer. One token per character, ~70 symbols. |
| `playground/data.py` | Corpus loading and next-character batching. |
| `corpus/make_corpus.py` | Generates the training text: eleven document types, all invented here. |
| `gradcheck.py` | Nudges individual weights and checks that `backward()` predicted the change. |
| `test_cache.py` | Checks the KV cache against the slow path it replaces. |
| `train.py` | The training loop. |
| `sample.py` | Generation: one-shot, interactive, or a temperature comparison. |
| `server.py` | Local web server, standard library only. Streams tokens as NDJSON. |
| `ui/index.html` | The playground page. One file, no build step, no CDN. |
| `eval_counting.py` | Measures whether the model knows that days go forward, against a shuffle control. |
| `run_sweep.py` | Runs the context-window experiment: N models in parallel, one BLAS thread each. |

## The one thing worth knowing

There is no autograd here. When you write `loss.backward()` in PyTorch, a
tape replays the forward pass in reverse. This model has no tape — the
chain rule is spelled out by hand in `GPT.backward()`, about 60 lines that
walk back through the head, the final LayerNorm, then each block's MLP,
attention, and residual connections in reverse order.

Hand-derived gradients are wrong the first time roughly always, so
`gradcheck.py` settles it empirically. It changes a single weight by ±1e-5,
measures how much the loss *actually* moved, and compares that to what
`backward()` claimed. Every parameter agrees to ~1e-11 absolute:

```
parameter        shape         max abs err    rel err   verdict
b0.attn.w        (16, 48)         2.64e-11   7.94e-06   ok
...
every gradient matches. worst absolute error: 3.25e-09
```

Run it after touching anything in `model.py`. It takes four seconds and it
is the difference between "the loss isn't going down" and "the loss isn't
going down *because of a transpose on line 240*".

## The corpus

*The Playground Library* — 1.2 million characters across **eleven document
types**, all invented by `corpus/make_corpus.py`, so there are no licensing
questions and we control exactly how much structure there is to learn.

The mix is the point. A model fed only journal entries learns one register,
and every prompt comes back sounding like a journal entry — we had exactly
that problem for the first few hours. Mixing genres forces it to work out
what kind of document it is in and write accordingly. Each type has its own
header, so typing one steers the model:

| type it with | and you get |
|---|---|
| `RECIPE: ` | serves, time, ingredient list, numbered method |
| `Q: ` | a question and an answer |
| `HOW TO ` | numbered instructions |
| `INVENTORY: ` | quantities, items, conditions, in columns |
| `WEATHER LOG -- ` | an hourly table of temp / tide / sky |
| `GLOSSARY` | headwords, parts of speech, definitions |
| `LETTER, from ` | salutation, body, sign-off |
| `NOTICE: ` | a short public announcement |
| `MEETING NOTES -- ` | attendees and bulleted non-decisions |
| `DAY 14 -- ` | a journal entry |
| `LEDGER -- ` | one terse line per day |

Underneath, three layers of structure that a model learns in order:
spelling, then sentence shape, then long-range structure — dialogue
attributed to whoever is speaking, a chapter's small cast recurring, day
numbers counting upward. That last one turned into its own investigation;
see **[docs/counting.md](docs/counting.md)**.

Three corpora ship, because two experiments needed their own:

```bash
python corpus/make_corpus.py                                   # the mixed library
python corpus/make_corpus.py --only chronicle --out corpus/chronicles.txt
python corpus/make_corpus.py --only ledger --increment one --out corpus/ledger1.txt
```

Want it to learn something else? Any UTF-8 text file works — there is
nothing corpus-specific in the model, and the vocabulary is built from
whatever you hand it:

```bash
python train.py --data path/to/your.txt --preset small
```

## Reading the numbers

Training prints **bits per character** alongside the loss. It is the honest
scoreboard: how many yes/no questions you would need to guess the next
character.

Measured, from the first real run — `tiny`, 3000 steps, 7.3 minutes:

| step | bits/char | what it could do |
|---|---|---|
| 1 | 6.09 | nothing. all 68 symbols equally likely |
| 30 | 4.27 | letter frequencies and spaces: `adoo lg t. AYeruhhig waesiml` |
| 250 | 2.40 | word-shaped clumps, some real words |
| 500 | 0.83 | spelling, mostly. sentences that end. |
| 1000 | 0.50 | the `DAY 185 -- the lantern quarter` header form, with glitches |
| 3000 | **0.37** | clean entries, dialogue, correct speaker attribution |

At step 3000 it writes this unprompted:

```
DAY 42 -- the orchard of bells
"Why does everyone lie about the bells?" asked Nessa.
"Bring a brass key and I will tell you." said Orpheus.
```

Note that 0.37 bits/char is *low*, and that is a statement about the corpus
more than the model — `make_corpus.py` builds from templates, so there is a
hard floor of real uncertainty (which name, which place) and not much else.
Point it at messier text and the same model will plateau far higher.

## The counting experiment

It never learned to count the days. It writes `DAY 42`, then `DAY 1903`,
then `DAY 2336` — the format is perfect, the sequence is nonsense.

The obvious explanation is that it cannot see. Day headers in the
chronicles corpus sit an average of **361 characters apart**, so with a
96-character window the previous day number is *never* in context:

```
block_size  96: can see the previous day number   0.0% of the time
block_size 192: can see the previous day number  10.7%
block_size 256: can see the previous day number  24.8%
block_size 384: can see the previous day number  58.2%
block_size 512: can see the previous day number  85.7%
```

So we trained one model per window — same architecture, same corpus, same
6,144 tokens per step, only the window changing (`run_sweep.py`), and
scored them with `eval_counting.py`. **The obvious explanation was wrong.**

| window | can see it | bits/char | forward | legal step |
|---|---|---|---|---|
| *the corpus* | — | — | 100% | 100% |
| 96 | 0% | 0.39 | 47% | 0% |
| 192 | 10.7% | 0.35 | 54% | 0% |
| 256 | 24.8% | 0.34 | 55% | 0% |
| 384 | 58.2% | 0.36 | 55% | 0% |
| 512 | 85.7% | 0.32 | 51% | 0% |

Random numbers score ~50% forward and 0% legal. Every model, at its full
3,000 steps — including the one that can see the previous day 86% of the
time — is indistinguishable from random. Giving it the information did not
make it use the information.

What worked was a denser corpus. `LEDGER` puts one line per day: headers 97
characters apart instead of 361, digits 4.2% of the text instead of 1.1%,
and ~63 day headers per training step instead of ~17. On that corpus the
window becomes decisive:

| corpus | headers apart | window | sees prev. day | legal | \|gap\| | shuffled |
|---|---|---|---|---|---|---|
| chronicles | 361 | 512 | 85.7% | 0% | 1068 | 896 |
| ledger | 97 | 64 | 0.2% | 0% | 3594 | 3476 |
| ledger | 97 | 128 | 94.7% | **19%** | **8** | 3431 |
| ledger | 97 | **192** | 100% | **85%** | **2** | 137 |

And it counts:

```
DAY 572 -- the long pier
Bastian the cartographer waited at the long pier until the lamps came on.
DAY 573 -- the orchard of bells
Sable walked to the winter archive without telling anyone.
DAY 574 -- the salt gardens
```

Two separable effects, and neither alone is enough:

- **Density.** At roughly matched visibility, the dense corpus counts (19%)
  and the sparse one does not (0%).
- **Visibility has to be near-total, not merely high.** 94.7% → 100% takes
  the score from 19% to 85%. I had treated the window as a threshold —
  below it blind, above it fine. It behaves far more like a dose.

The write-up is in **[docs/counting.md](docs/counting.md)**, including the
two conclusions I got wrong on the way — one published before testing, and
one from comparing checkpoints that hadn't had the same amount of training.

## Sizes

`--preset` picks the shape. Bigger learns more and thinks slower; NumPy on a
CPU is the speed limit.

| preset | block | layers | heads | width | params | ms/step |
|---|---|---|---|---|---|---|
| `nano` | 64 | 2 | 2 | 64 | 108,544 | ~50 |
| `tiny` | 96 | 3 | 3 | 96 | 351,456 | ~135 (measured) |
| `small` | 128 | 4 | 4 | 128 | 818,432 | ~400 |
| `medium` | 192 | 6 | 6 | 192 | 2,719,488 | ~1800 |

## Training it more

Checkpoints hold the weights, the vocabulary, and the optimizer state, so
training picks up exactly where it stopped — momentum included.

```bash
python train.py --resume checkpoints/tiny.npz --steps 3000
```

Ctrl-C is safe. It saves before it exits.

## Knobs to play with

```bash
python sample.py --compare            # the same prompt at 5 temperatures
python sample.py -t 0.2               # cautious, repetitive, safe
python sample.py -t 1.3               # unhinged, inventive, misspelled
python sample.py -t 0 -k 0            # fully deterministic (greedy)
python sample.py --prompt '"Who left' # make it finish your sentence
```

Temperature divides the logits before sampling. Low temperature sharpens the
distribution toward whatever the model is most confident about; high
temperature flattens it. Top-k keeps only the `k` most likely characters,
which stops a rare bad draw from derailing a whole paragraph.

## Things this model does not have

Deliberately, so the code stays readable end to end. Each is a good afternoon:

- **Dropout.** With a small corpus it will eventually memorize. Watch for the
  val loss turning back upward while train loss keeps falling.
- **A real tokenizer.** BPE would let it work in word-pieces instead of
  letters, so it stops spending capacity on spelling.
- **Rotary position embeddings.** The learned position table means it can
  never read more than `block_size` characters of context.
- **Instruction tuning.** It continues text. It does not answer questions —
  yet. Teaching it to would mean fine-tuning on prompt/response pairs.

## The KV cache

It *does* have one of these now. Attention keys and values, once computed
for a position, never change — but the naive sampling loop recomputed all of
them for every single new character. `GPT.forward_incremental()` keeps them
in a preallocated buffer and only computes the new column:

```
200 tokens (96-wide window):  uncached  426 tok/s   cached  4125 tok/s   9.8x
```

One wrinkle worth knowing. When the window fills, you cannot just drop the
oldest entry and shift — position embeddings here are *absolute*, so every
cached key would suddenly be sitting at the wrong position and the model
would quietly produce garbage. Instead the cache is rebuilt from the most
recent 3/4 of the window, which costs one prefill every `block_size/4`
tokens instead of a full pass every character. (Rotary embeddings are the
usual fix for this, and are on the list above.)

A cache is a correctness hazard — a subtly wrong one still produces
plausible text — so `test_cache.py` checks it against the slow path:
identical logits on prefill, identical one token at a time, no peeking at
the future, and the same 120 greedy tokens across a window rebuild.
