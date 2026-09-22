"""Generates the training corpus. Everything the model reads comes from here.

It is all invented, so the corpus is ours and free of anyone's copyright,
and we control exactly how much structure there is to learn.

The corpus is a MIX of document types, not one genre. That matters: a model
fed only journal entries learns one register and one shape, and every prompt
comes back sounding like a journal entry. Mixing recipes, letters, glossary
definitions, Q&A, inventories, weather logs and so on forces it to notice
what kind of document it is in and switch register accordingly -- which is
the first thing that makes it feel like it is responding to you rather than
reciting.

Each type has a distinctive header, so you can steer it from the playground
by typing that header:

    RECIPE:            -> ingredients and steps
    Q:                 -> a question and an answer
    DAY 14 --          -> a journal entry
    GLOSSARY           -> definitions
    HOW TO             -> numbered instructions
    NOTICE:            -> a short public announcement

    python corpus/make_corpus.py --chars 1200000
    python corpus/make_corpus.py --only chronicle --out corpus/chronicles.txt
"""

import argparse
import os
import random

# ----------------------------------------------------------------- material

NAMES = [
    "Marlow", "Juniper", "Ovid", "Tessa", "Bramble", "Cassia", "Fen",
    "Orpheus", "Wren", "Dalia", "Corvin", "Nessa", "Pike", "Ambrose",
    "Isolde", "Rook", "Verity", "Silas", "Marta", "Quill", "Halden",
    "Odile", "Bastian", "Lark", "Emmaline", "Grover", "Pell", "Sable",
]

ROLES = [
    "the cartographer", "the clockmender", "the beekeeper", "the archivist",
    "the ferryman", "the glassblower", "the bell-ringer", "the astronomer",
    "the gardener", "the locksmith", "the printer", "the lamplighter",
    "the tinsmith", "the baker", "the diver", "the weaver", "the surveyor",
]

PLACES = [
    "the glass market", "the drowned library", "the north bridge",
    "the orchard of bells", "the lantern quarter", "the salt gardens",
    "the observatory", "the hollow tower", "the paper mill",
    "the harbor stairs", "the winter archive", "the copper baths",
    "the fog station", "the long pier", "the quiet arcade",
]

OBJECTS = [
    "a brass key", "an unfinished map", "a jar of honey", "a folded letter",
    "a cracked lens", "a small green door", "a ledger of tides",
    "a music box", "a bundle of rosemary", "a broken compass",
    "a blue umbrella", "a tin of matches", "a borrowed coat",
    "a bird made of paper", "a river stone", "a pocket watch",
]

THINGS = [
    "brass key", "lantern", "ledger", "compass", "kettle", "hand mirror",
    "coil of rope", "tin whistle", "fishing float", "wax seal", "inkpot",
    "folding knife", "sextant", "birdcage", "pressed flower", "door hinge",
    "weather vane", "sounding line", "glass float", "signal flag",
]

ADJECTIVES = [
    "quiet", "stubborn", "enormous", "half-forgotten", "patient", "restless",
    "narrow", "luminous", "peculiar", "steady", "unhurried", "crooked",
    "gentle", "impossible", "ordinary", "reluctant", "bright", "weathered",
]

FOODS = [
    "honey bread", "salt cod stew", "pear tart", "barley soup",
    "rosemary biscuits", "smoked herring", "plum cake", "onion pie",
    "apple butter", "brown loaf", "fish chowder", "walnut cake",
    "pickled beets", "lemon pudding", "mushroom broth", "ginger snaps",
]

INGREDIENTS = [
    "flour", "butter", "honey", "sea salt", "black pepper", "rosemary",
    "thyme", "onions", "cold water", "brown sugar", "eggs", "cream",
    "barley", "dried plums", "walnuts", "lemon peel", "bay leaf", "milk",
    "cider vinegar", "mustard seed", "parsley", "smoked paprika",
]

COOK_STEPS = [
    "Warm the {ing} until it loosens, then set it aside.",
    "Mix the {ing} and the {ing2} until there are no lumps left.",
    "Rest the dough somewhere cool for {n} minutes.",
    "Fold in the {ing} a little at a time. Do not rush this.",
    "Bake for {n} minutes, until the top is dark at the edges.",
    "Add the {ing} last, off the heat, or it will turn bitter.",
    "Simmer for {n} minutes, skimming whatever rises.",
    "Let it stand for {n} minutes before cutting, however tempting.",
    "Taste it. Add salt if it is flat, vinegar if it is dull.",
]

WEATHER = [
    "It rained all morning", "The fog came up from the water",
    "The sky stayed the color of tin", "A warm wind moved through the streets",
    "Snow settled on the rooftops", "The afternoon was clear and cold",
    "Thunder rolled somewhere past the hills", "The light went gold early",
]

SKY = ["clear", "overcast", "low fog", "broken cloud", "steady rain",
       "sleet", "high haze", "wind from the east", "snow, light"]

TIMES = [
    "At dawn", "Just after noon", "By evening", "Late that night",
    "Before the bells", "On the hour", "Toward morning", "At low tide",
]

ACTIONS = [
    "walked to {place} without telling anyone",
    "left {object} on the step and did not explain why",
    "counted the windows of {place} twice, and got two different numbers",
    "traded {object} for a promise",
    "waited at {place} until the lamps came on",
    "wrote the whole thing down and then lost the notebook",
    "opened every drawer in {place} looking for {object}",
    "carried {object} across town in the rain",
    "measured the distance to {place} in footsteps",
    "argued with the tide and lost",
]

OBSERVATIONS = [
    "Nothing came of it, which was its own kind of answer.",
    "It seemed important at the time.",
    "That is how these things usually begin.",
    "No one mentioned it again for a week.",
    "The whole business took less time than expected.",
    "Afterward the streets smelled of rain and iron.",
    "It was the sort of mistake worth making twice.",
    "Everyone agreed, which should have been a warning.",
    "The clocks disagreed by four minutes, as always.",
    "By then it hardly mattered.",
]

QUESTIONS = [
    "Do you remember the {adj} one?",
    "How far is it to {place}?",
    "Who left {object} here?",
    "Is it always this {adj}?",
    "What do we do if the water rises?",
    "Have you been to {place} in winter?",
    "Why does everyone lie about the bells?",
    "Can you keep {object} for me?",
]

ANSWERS = [
    "Only the {adj} part.",
    "Farther than it looks, and closer than you want.",
    "I have been asking that since Tuesday.",
    "Ask me again when the fog lifts.",
    "Nobody knows, and nobody says so.",
    "It depends entirely on the tide.",
    "That is not a question, that is a confession.",
    "Bring {object} and I will tell you.",
    "Yes. No. Both, on alternate days.",
]

CHAPTER_OPENERS = [
    "This is the part of the record that everyone skips.",
    "The account below was copied twice and corrected once.",
    "What follows was told to me by someone who was not there.",
    "I am writing this down before I talk myself out of it.",
    "The following was found folded inside a ledger.",
]

CHAPTER_CLOSERS = [
    "And that was the end of it, more or less.",
    "The rest of the page is water-damaged.",
    "Here the handwriting changes.",
    "Nothing further was recorded that season.",
    "The account breaks off there.",
]

TOPICS = [
    "tides", "bells", "fog", "lamp oil", "bridges", "paper", "honey",
    "clocks", "salt", "rope", "glass", "ferries", "rain", "keys", "maps",
]

FEELINGS = ["pleased", "uneasy", "tired", "hopeful", "impatient", "grateful",
            "sorry", "certain", "unconvinced", "cheerful"]

# How far the day number advances between entries. The default is a random
# 1-3, which means even a model that understands the rule perfectly can only
# predict a distribution. `--increment one` makes it exactly +1, so the next
# number becomes fully determined by the previous one -- useful for asking
# whether a failure to count is about arithmetic or about uncertainty.
DAY_STEP = (1, 3)


# ------------------------------------------------------------------ helpers

def fill(rng, template):
    """Substitute {place}, {object}, {adj}, {name}, {role}, {thing}, {ing}."""
    for _ in range(8):
        if "{" not in template:
            break
        template = (template
                    .replace("{place}", rng.choice(PLACES), 1)
                    .replace("{object}", rng.choice(OBJECTS), 1)
                    .replace("{adj}", rng.choice(ADJECTIVES), 1)
                    .replace("{name}", rng.choice(NAMES), 1)
                    .replace("{role}", rng.choice(ROLES), 1)
                    .replace("{thing}", rng.choice(THINGS), 1)
                    .replace("{topic}", rng.choice(TOPICS), 1)
                    .replace("{ing2}", rng.choice(INGREDIENTS), 1)
                    .replace("{ing}", rng.choice(INGREDIENTS), 1))
    return template


def who(rng):
    n = rng.choice(NAMES)
    return f"{n} {rng.choice(ROLES)}" if rng.random() < 0.35 else n


# -------------------------------------------------------- document: journal

def sentence(rng, cast):
    speaker = rng.choice(cast)
    k = rng.random()
    if k < 0.30:
        return fill(rng, f"{speaker} {rng.choice(ACTIONS)}.")
    if k < 0.45:
        return fill(rng, f"{rng.choice(TIMES)}, {speaker} {rng.choice(ACTIONS)}.")
    if k < 0.58:
        return fill(rng, f"{rng.choice(WEATHER)}, and {speaker} "
                         f"{rng.choice(ACTIONS)}.")
    if k < 0.70:
        return fill(rng, f"{speaker} had {{object}}, which was {{adj}} enough "
                         f"to mention.")
    if k < 0.82:
        return fill(rng, f"There is a {{adj}} rule about {{place}}: "
                         f"do not go looking for {{object}}.")
    return rng.choice(OBSERVATIONS)


def dialogue(rng, cast):
    a, b = rng.sample(cast, 2)
    lines = [f'"{fill(rng, rng.choice(QUESTIONS))}" asked {a}.',
             f'"{fill(rng, rng.choice(ANSWERS))}" said {b}.']
    if rng.random() < 0.35:
        lines.append(f'"{rng.choice(OBSERVATIONS)}" added {a}.')
    return "\n".join(lines)


def chronicle(rng, state):
    """A chapter of journal entries with a small recurring cast and day
    numbers that always count upward."""
    cast = [f"{n} {r}" if rng.random() < 0.35 else n
            for n, r in zip(rng.sample(NAMES, 4), rng.sample(ROLES, 4))]
    state["chapter"] = state.get("chapter", 0) + 1
    day = state.get("day", 1)
    out = [f"==== CHAPTER {state['chapter']}: {rng.choice(PLACES).upper()} ====",
           rng.choice(CHAPTER_OPENERS), ""]
    for _ in range(rng.randint(3, 7)):
        out.append(f"DAY {day} -- {rng.choice(PLACES)}")
        for _ in range(rng.randint(2, 5)):
            out.append(dialogue(rng, cast) if rng.random() < 0.28
                       else sentence(rng, cast))
        out.append("")
        day += rng.randint(*DAY_STEP)
    state["day"] = day
    out.append(rng.choice(CHAPTER_CLOSERS))
    return "\n".join(out)


# -------------------------------------------------------- document: ledger

def ledger(rng, state):
    """The same day-by-day record, stripped to one line per day.

    This exists to separate two things the chronicle confounds. In the
    chronicle, day headers are ~361 characters apart AND the digits are
    barely 1% of the text -- so a model can fail to count either because it
    cannot see the previous number, or because getting it right buys it
    almost nothing on the loss. Here the headers are ~60 characters apart
    and the digits are a far larger share of what there is to predict.
    """
    state["ledger_day"] = state.get("ledger_day", 1)
    day = state["ledger_day"]
    out = [f"LEDGER -- {rng.choice(PLACES)}"]
    for _ in range(rng.randint(8, 16)):
        out.append(f"DAY {day} -- {rng.choice(PLACES)}")
        out.append(fill(rng, f"{who(rng)} {rng.choice(ACTIONS)}."))
        day += rng.randint(*DAY_STEP)
    state["ledger_day"] = day
    return "\n".join(out)


# -------------------------------------------------------- document: recipe

def recipe(rng, state):
    dish = rng.choice(FOODS)
    n_ing = rng.randint(4, 7)
    out = [f"RECIPE: {dish}",
           f"Serves {rng.randint(2, 8)}. Takes about "
           f"{rng.choice([20, 25, 30, 40, 45, 60, 90])} minutes.",
           "", "Ingredients:"]
    for ing in rng.sample(INGREDIENTS, n_ing):
        amount = rng.choice(["a handful of", "two cups of", "one spoon of",
                             "a pinch of", "three ounces of", "half a cup of",
                             "a little", "as much as you like of"])
        out.append(f"  - {amount} {ing}")
    out.append("")
    out.append("Method:")
    for i, step in enumerate(rng.sample(COOK_STEPS, rng.randint(3, 5)), 1):
        out.append(f"  {i}. " + fill(rng, step).replace(
            "{n}", str(rng.choice([5, 10, 15, 20, 25, 30, 40]))))
    out.append("")
    out.append(rng.choice([
        "Keeps for three days, wrapped in cloth.",
        "Better the next morning, for reasons no one can explain.",
        "Do not make this in a hurry.",
        "It is supposed to look like that.",
        "Serve warm, with more butter than seems wise.",
    ]))
    return "\n".join(out)


# -------------------------------------------------------- document: letter

def letter(rng, state):
    a, b = rng.sample(NAMES, 2)
    out = [f"LETTER, from {a} to {b}", "", f"Dear {b},", ""]
    out.append(fill(rng, rng.choice([
        "I am writing about the {topic} again, and I am sorry about that.",
        "You asked what became of {object}. The answer is complicated.",
        "It has been {adj} here since you left.",
        "The business at {place} is finished, though not the way we planned.",
    ])))
    for _ in range(rng.randint(1, 3)):
        out.append(fill(rng, rng.choice([
            "I went to {place} on Tuesday and thought of you.",
            "Everyone here is {adj} about the {topic}, which helps nobody.",
            "Bring {object} when you come. You will understand why.",
            "The weather has been {adj} and the ferries are unreliable.",
            "I am {feel} about the whole thing, if I am honest.",
        ])).replace("{feel}", rng.choice(FEELINGS)))
    out += ["", rng.choice(["Yours,", "With affection,", "As ever,",
                            "In haste,", "Fondly,"]), a]
    return "\n".join(out)


# ------------------------------------------------------ document: glossary

def glossary(rng, state):
    out = ["GLOSSARY", ""]
    for thing in rng.sample(THINGS, rng.randint(4, 7)):
        kind = rng.choice(["noun", "noun", "verb", "adjective"])
        out.append(f"{thing} ({kind})")
        out.append("    " + fill(rng, rng.choice([
            "A {adj} thing, usually found near {place}.",
            "Anything used to settle an argument about {topic}.",
            "What you call it after it has broken twice.",
            "A {adj} tool. Nobody agrees who invented it.",
            "In the old records, a kind of payment.",
            "Something borrowed and rarely returned.",
        ])))
        out.append("")
    return "\n".join(out).rstrip()


# ----------------------------------------------------------- document: Q&A

def qa(rng, state):
    out = []
    for _ in range(rng.randint(3, 6)):
        out.append("Q: " + fill(rng, rng.choice(QUESTIONS + [
            "What is {object} worth these days?",
            "How do you get to {place} after dark?",
            "Is the {topic} still a problem?",
            "Should I bring {object} or leave it?",
            "Who do I ask about the {topic}?",
        ])))
        out.append("A: " + fill(rng, rng.choice(ANSWERS + [
            "Less than you paid, more than you think.",
            "You do not. You wait until morning like everyone else.",
            "It is always a problem. That is what it is for.",
            "Leave it. You will only lose it at {place}.",
            "Ask {name}, and do not believe the first answer.",
        ])))
        out.append("")
    return "\n".join(out).rstrip()


# --------------------------------------------------------- document: howto

def howto(rng, state):
    task = rng.choice([
        "mend a clock that runs fast", "read a tide table",
        "dry rosemary properly", "unstick a swollen door",
        "keep bees through a wet spring", "fold a paper bird",
        "clean a cracked lens", "tie a line that will not slip",
        "find north without a compass", "store apples until spring",
    ])
    out = [f"HOW TO {task}", ""]
    for i in range(1, rng.randint(4, 7)):
        out.append(f"{i}. " + fill(rng, rng.choice([
            "Start with the {thing}. Everything else depends on it.",
            "Work slowly. This is where most people ruin it.",
            "Check twice, then check once more in better light.",
            "If it resists, stop. Forcing it makes the next step harder.",
            "Set the {thing} aside somewhere you will not knock it over.",
            "Wait. It will look wrong for a while, and then it will not.",
            "Mark where it was before you move anything.",
            "Clean the {thing} first, even if it looks clean.",
        ])))
    out.append("")
    out.append(rng.choice([
        "If none of this works, it was probably broken before you started.",
        "There is a faster way. It does not work as well.",
        "Done properly, this lasts about a year.",
    ]))
    return "\n".join(out)


# ------------------------------------------------- document: inventory/list

def inventory(rng, state):
    where = rng.choice(PLACES)
    out = [f"INVENTORY: {where}",
           f"counted by {who(rng)}", ""]
    for thing in rng.sample(THINGS, rng.randint(5, 9)):
        qty = rng.choice([1, 1, 2, 3, 4, 6, 8, 12, 20, 40])
        cond = rng.choice(["good", "fair", "poor", "broken", "as new",
                           "unusable", "needs cleaning", "missing a part"])
        out.append(f"  {qty:>3} x {thing:<18} {cond}")
    out.append("")
    out.append(rng.choice([
        "Two items unaccounted for. See the previous count.",
        "Nothing here matches the list from last year.",
        "Recount scheduled, then postponed.",
        "Everything present and correct, which is unusual.",
    ]))
    return "\n".join(out)


# ------------------------------------------------------ document: log table

def weatherlog(rng, state):
    place = rng.choice(PLACES)
    out = [f"WEATHER LOG -- {place}", "", "  hour   temp   tide   sky"]
    for hour in range(rng.choice([4, 6, 8]), 22, rng.choice([2, 3, 4])):
        out.append(f"  {hour:02d}:00  {rng.randint(-4, 26):>3}C   "
                   f"{rng.choice(['low', 'rising', 'high', 'falling']):<7} "
                   f"{rng.choice(SKY)}")
    out.append("")
    out.append(rng.choice(WEATHER) + ".")
    return "\n".join(out)


# -------------------------------------------------------- document: notice

def notice(rng, state):
    out = ["NOTICE: " + fill(rng, rng.choice([
        "{place} closed until further notice",
        "the bells will not ring this week",
        "found, {object}",
        "the ferry timetable has changed again",
        "do not drink the water at {place}",
        "lost, {object}, of no value to anyone else",
    ]))]
    out.append("")
    out.append(fill(rng, rng.choice([
        "Apply to {name} at {place}. Bring proof, or something like it.",
        "This is the third such notice. It will be the last.",
        "Nobody is happy about it, including whoever wrote this.",
        "Questions to {name}, who does not know either.",
        "Normal service resumes when the water goes down.",
    ])))
    return "\n".join(out)


# --------------------------------------------------- document: meeting note

def meeting(rng, state):
    people = rng.sample(NAMES, rng.randint(3, 5))
    out = [f"MEETING NOTES -- {rng.choice(PLACES)}",
           "present: " + ", ".join(people), ""]
    for _ in range(rng.randint(2, 4)):
        out.append(fill(rng, rng.choice([
            "- {name} raised the matter of the {topic}. Discussed at length.",
            "- Agreed: no decision until the next count.",
            "- {name} objected. The objection was noted and ignored.",
            "- The question of {object} was deferred again.",
            "- Someone will look into it. It is not clear who.",
        ])).replace("{name}", rng.choice(people), 1))
    out.append("")
    out.append(rng.choice([
        "Next meeting: when there is something to report.",
        "Meeting closed early on account of the weather.",
        "No minutes were kept for the second half.",
    ]))
    return "\n".join(out)


# ------------------------------------------------------------------ corpus

KINDS = {
    "chronicle": (chronicle, 26),
    "ledger":    (ledger, 6),
    "recipe":    (recipe, 12),
    "letter":    (letter, 11),
    "qa":        (qa, 11),
    "howto":     (howto, 10),
    "glossary":  (glossary, 8),
    "inventory": (inventory, 8),
    "weather":   (weatherlog, 6),
    "notice":    (notice, 4),
    "meeting":   (meeting, 4),
}


def build(target_chars, seed=20260921, only=None):
    rng = random.Random(seed)
    kinds = {only: KINDS[only]} if only else KINDS
    names = list(kinds)
    weights = [kinds[k][1] for k in names]

    parts = ["THE PLAYGROUND LIBRARY",
             "an assortment, assembled by several hands, none of them careful",
             ""]
    total = sum(len(p) for p in parts)
    state = {}
    counts = {k: 0 for k in names}
    while total < target_chars:
        kind = rng.choices(names, weights=weights)[0]
        text = kinds[kind][0](rng, state)
        counts[kind] += 1
        parts.append("\n" + text + "\n")
        total += len(text) + 2
    return "\n".join(parts) + "\n", counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=1_200_000)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--only", choices=sorted(KINDS),
                    help="generate a single document type")
    ap.add_argument("--increment", choices=("random", "one"), default="random",
                    help="how much the day number advances between entries")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "playground.txt"))
    args = ap.parse_args()

    global DAY_STEP
    if args.increment == "one":
        DAY_STEP = (1, 1)
    text, counts = build(args.chars, args.seed, args.only)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    vocab = sorted(set(text))
    shown = "".join(vocab).replace("\n", "\\n")
    print(f"wrote {args.out}")
    print(f"  {len(text):,} characters, {len(text.split()):,} words")
    print(f"  {len(vocab)} distinct symbols: {shown}")
    print("\n  documents:")
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<11} {n:>5}")


if __name__ == "__main__":
    main()
