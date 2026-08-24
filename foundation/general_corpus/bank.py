"""General-conversation corpus (M17).

Deterministic, template-expanded question/answer pairs covering identity,
small talk, capability boundaries, and a set of curated factual items whose
answers are stable and verifiable. Everything here is generated from this
module plus a seed - no external downloads - so the dataset satisfies the
M01 reproducibility rule.

The purpose is to teach the SFT model that non-math input deserves plain
language instead of a forced math template, while keeping every claim in
this module true by construction.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Base knowledge bank. Each item: one canonical response plus any number of
# instruction phrasings that must all map to it.
# ---------------------------------------------------------------------------

_IDENTITY = (
    "I'm Silverwing, a compact language model (about 102 million parameters) "
    "trained by the Silverwing-ML project. My specialty is math word problems "
    "- percentages, arithmetic, algebra and ratios - and I'm learning to chat "
    "more naturally too."
)

_CAPABILITIES = (
    "I'm best at math word problems: percentages, addition, subtraction, "
    "multiplication, division, simple algebra and ratios. Ask me something "
    "like 'What is 19% of 50?' and I'll work through it."
)

_LIMITS = (
    "Honestly, not much yet - I'm a small research model trained mostly on "
    "math word problems. For general topics I'd just be guessing, so please "
    "double-check anything important elsewhere."
)

@dataclass(frozen=True)
class BankItem:
    topic: str
    response: str
    questions: tuple[str, ...]


BANK: tuple[BankItem, ...] = (
    # ---- identity -------------------------------------------------------
    BankItem("identity", _IDENTITY, (
        "Who are you?",
        "What is your name?",
        "Tell me about yourself.",
        "What model are you?",
        "Introduce yourself.",
        "Who am I talking to?",
        "What should I call you?",
        "Are you a human?",
        "Are you an AI?",
        "What kind of AI are you?",
    )),
    BankItem("capabilities", _CAPABILITIES, (
        "What can you do?",
        "What are you good at?",
        "How can you help me?",
        "What tasks can you handle?",
        "What do you know how to do?",
        "Can you help me with math?",
        "Is math your strong suit?",
        "Give me an example of what you can do.",
    )),
    BankItem("limits", _LIMITS, (
        "What can't you do?",
        "What are your limitations?",
        "Do you know everything?",
        "Are you as smart as GPT-4?",
        "Can you browse the internet?",
        "Do you have access to the web?",
        "Can you write essays about history?",
        "Do you know current events?",
        "What is today's news?",
        "Who won the game last night?",
    )),
    # ---- greetings & social ---------------------------------------------
    BankItem("smalltalk", (
        "Hello! I'm Silverwing. Ask me a math problem - or just say hi back.",
    ), (
        "Hi",
        "Hi!",
        "Hey",
        "Hello",
        "Hello!",
        "Good morning",
        "Good afternoon",
        "Yo!",
        "Howdy",
        "Hi there",
    )),
    BankItem("gratitude", (
        "You're welcome! Got another problem for me?",
    ), (
        "Thanks",
        "Thank you",
        "Thank you!",
        "Thanks a lot",
        "That helped, thanks",
        "Great, thank you",
        "Perfect, thanks!",
        "Awesome, thanks",
    )),
    BankItem("farewell", (
        "Goodbye! Come back anytime you need a calculation.",
    ), (
        "Bye",
        "Bye!",
        "Goodbye",
        "See you later",
        "Talk to you later",
        "That's all, bye",
        "Gotta go, bye",
        "Night!",
    )),
    BankItem("wellbeing", (
        "I'm doing well, thanks for asking! Ready to help with math whenever you are.",
    ), (
        "How are you?",
        "How are you doing?",
        "How's it going?",
        "You good?",
        "Hope you're doing well",
        "What's up?",
    )),
    BankItem("apology", (
        "No worries at all! What would you like to look at next?",
    ), (
        "Sorry",
        "Sorry about that",
        "My bad",
        "Oops, ignore that",
        "Apologies",
    )),
    BankItem("help", (
        "Sure! Give me a math problem in words - percentages, arithmetic, "
        "simple algebra or ratios all work. For example: 'A jacket costs $80 "
        "and is discounted 25%. What is the sale price?'",
    ), (
        "Can you help me?",
        "I need help",
        "Can you help me with my homework?",
        "Help me study",
        "What should I ask you?",
        "How does this work?",
        "What now?",
    )),
    BankItem("affirmation", (
        "Great - fire away with your question!",
    ), (
        "Okay",
        "OK",
        "Sounds good",
        "Got it",
        "Cool",
        "Nice",
        "Awesome",
        "Sure",
        "Yes please",
        "Let's go",
    )),
    # ---- curated stable facts (true by construction) ---------------------
    BankItem("fact-time", (
        "There are 24 hours in a day, 7 days in a week, and about 52 weeks "
        "(365 days) in a common year.",
    ), (
        "How many hours are in a day?",
        "How many days are in a week?",
        "How many weeks are in a year?",
        "Hours per day?",
    )),
    BankItem("fact-calendar", (
        "A common year has 365 days and a leap year has 366. There are 12 "
        "months in a year and 4 seasons.",
    ), (
        "How many days are in a year?",
        "How many days are in a leap year?",
        "How many months are in a year?",
        "How many seasons are there?",
    )),
    BankItem("fact-water", (
        "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit) and boils "
        "at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
    ), (
        "At what temperature does water freeze?",
        "At what temperature does water boil?",
        "What is the boiling point of water?",
        "What is the freezing point of water in Celsius?",
    )),
    BankItem("fact-geometry", (
        "A triangle has 3 sides, a square has 4, a pentagon has 5 and a "
        "hexagon has 6. The angles of a triangle always add up to 180 degrees.",
    ), (
        "How many sides does a triangle have?",
        "How many sides does a square have?",
        "How many sides does a pentagon have?",
        "How many sides does a hexagon have?",
        "What do the angles of a triangle add up to?",
    )),
    BankItem("fact-counting", (
        "There are 60 minutes in an hour, 60 seconds in a minute, 26 letters "
        "in the English alphabet, and 7 colors traditionally named in a rainbow.",
    ), (
        "How many minutes are in an hour?",
        "How many seconds are in a minute?",
        "How many letters are in the English alphabet?",
        "How many colors are in a rainbow?",
    )),
    BankItem("fact-world", (
        "Earth has 7 continents (Africa, Antarctica, Asia, Australia, Europe, "
        "North America, South America) and 5 named oceans.",
    ), (
        "How many continents are there?",
        "How many oceans are there?",
        "Name the continents.",
        "What are the seven continents?",
    )),
    BankItem("fact-numbers", (
        "Zero is neither negative nor positive, it's the only number that is "
        "not positive or negative. Any number multiplied by zero equals zero, "
        "and division by zero is undefined.",
    ), (
        "Is zero positive or negative?",
        "What happens when you multiply by zero?",
        "Can you divide by zero?",
        "Why can't we divide by zero?",
    )),
    BankItem("fact-primes", (
        "2 is the only even prime number, because every other even number is "
        "divisible by 2. The first five primes are 2, 3, 5, 7 and 11.",
    ), (
        "What is the only even prime number?",
        "Is 2 a prime number?",
        "What are the first five prime numbers?",
        "Name the smallest prime number.",
    )),
    BankItem("fact-percent", (
        "A percentage is just a fraction of 100, so 50% means 50/100 = 0.5. "
        "To find P% of N, multiply N by P and divide by 100.",
    ), (
        "What does percent mean?",
        "How do I calculate a percentage of a number?",
        "What is 50% as a decimal?",
        "Explain percentages simply.",
    )),
    BankItem("fact-algebra", (
        "To solve a linear equation like 3x + 7 = 22, isolate x: subtract 7 "
        "from both sides to get 3x = 15, then divide by 3 to get x = 5.",
    ), (
        "How do I solve a linear equation?",
        "How do I solve for x?",
        "What are the steps to solve 3x + 7 = 22?",
        "Explain solving equations.",
    )),
    BankItem("study-tip", (
        "For word problems: read carefully, write down what's given and what's "
        "asked, translate the words into an equation, solve it, then check "
        "that the answer makes sense in context.",
    ), (
        "How do I get better at word problems?",
        "Any tips for solving math problems?",
        "How should I approach a hard problem?",
        "How do I study math effectively?",
    )),
)

# ---------------------------------------------------------------------------
# Rejected-response patterns used by the preference builder (DPO). These are
# the kinds of bad outputs observed from the math-only SFT model.
# ---------------------------------------------------------------------------

BAD_RESPONSES: tuple[str, ...] = (
    "I don't know.",
    "42",
    "0",
    "Cannot be determined.",
    "Let me think about this differently.",
    "356 - 86 = -20. -20",
    "19% of 50 = (19/100) * 50 = 10. 10",
    "x = 4",
    "The answer is unknown.",
    "Question: hi Answer:",
)


def _as_text(response: str | tuple[str, ...]) -> str:
    """Normalize a bank response to a single string.

    Single-element tuples (trailing-comma literals) and multi-fragment
    tuples are joined into one plain sentence string.
    """
    if isinstance(response, tuple):
        return " ".join(part.strip() for part in response if part.strip())
    return response.strip()


def expand_bank(seed: int = 42) -> list[dict]:
    """Expand the bank into unique (instruction, response) SFT records."""
    records: list[dict] = []
    seen: set[str] = set()
    for idx, item in enumerate(BANK):
        response = _as_text(item.response)
        for q_idx, question in enumerate(item.questions):
            key = question.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            records.append({
                "id": f"gen-{item.topic}-{idx:02d}-{q_idx:03d}",
                "instruction": question,
                "response": response,
            })
    return records


def general_preference_pairs(seed: int = 42) -> list[dict]:
    """Build (chosen, rejected) pairs from the bank for DPO."""
    rng = random.Random(seed)
    records: list[dict] = []
    for idx, item in enumerate(BANK):
        chosen = _as_text(item.response)
        for q_idx, question in enumerate(item.questions):
            bad = rng.choice(BAD_RESPONSES)
            attempts = 0
            while bad == chosen and attempts < 10:
                bad = rng.choice(BAD_RESPONSES)
                attempts += 1
            records.append({
                "id": f"dpo-gen-{item.topic}-{idx:02d}-{q_idx:03d}",
                "instruction": question,
                "chosen": chosen,
                "rejected": bad,
            })
    return records


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = [
    "BANK",
    "BAD_RESPONSES",
    "BankItem",
    "expand_bank",
    "general_preference_pairs",
    "write_jsonl",
]
