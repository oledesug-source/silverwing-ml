"""Massive SFT data expansion — multi-turn, tool chains, code, reasoning."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
(Path(__file__).resolve().parent / "__init__.py").touch(exist_ok=True)

random.seed(42)


def _id(text: str) -> str:
    return f"sw-exp-{hashlib.md5(text.encode()).hexdigest()[:8]}"


def _multi_turn_conversations() -> list[dict]:
    """Real multi-turn conversations."""
    convos = [
        # --- Math tutoring ---
        ("I need help with math.", "Of course! What topic are you working on?"),
        ("Fractions. I don't get them.", "Fractions represent parts of a whole. The top number (numerator) tells you how many parts you have. The bottom (denominator) tells you how many parts make the whole. For example, 3/4 means 3 out of 4 equal parts. What specifically is confusing?"),
        ("How do I add fractions with different denominators?", "You need a common denominator. For example, 1/3 + 1/4: find the least common multiple of 3 and 4, which is 12. Convert: 1/3 = 4/12 and 1/4 = 3/12. Now add: 4/12 + 3/12 = 7/12."),
        ("Can you show me another example?", "Sure! Let's add 2/5 + 1/3. Common denominator is 15. 2/5 = 6/15, 1/3 = 5/15. Sum = 11/15. Try one yourself: what is 1/2 + 1/6?"),
        ("Is it 4/6?", "Close! 1/2 = 3/6 and 1/6 = 1/6. So 3/6 + 1/6 = 4/6, which simplifies to 2/3. You had the right idea! The key is converting to the same denominator first."),
        # --- Coding help ---
        ("How do I read a file in Python?", "Use the open() function. Here's the simplest way:\n\nwith open('file.txt', 'r') as f:\n    content = f.read()\n\nThe 'with' statement ensures the file is closed properly."),
        ("What if the file doesn't exist?", "Use a try/except block:\n\ntry:\n    with open('file.txt', 'r') as f:\n        content = f.read()\nexcept FileNotFoundError:\n    print('File not found!')\n\nYou can also use pathlib for a cleaner approach."),
        ("How about writing to a file?", "Use 'w' mode to write (overwrites) or 'a' to append:\n\n# Write (creates new file or overwrites)\nwith open('output.txt', 'w') as f:\n    f.write('Hello, world!\\n')\n\n# Append\nwith open('output.txt', 'a') as f:\n    f.write('Another line\\n')"),
        # --- Science question ---
        ("Why is the sky blue?", "It's due to Rayleigh scattering. Sunlight contains all colors. When it hits Earth's atmosphere, shorter wavelengths (blue) scatter more than longer ones (red). Since blue light scatters in all directions, we see blue from every angle."),
        ("Does that mean sunsets are red for the same reason?", "Exactly! At sunset, sunlight travels through more atmosphere. Most blue light has been scattered away before reaching your eyes. What's left is the longer wavelengths — reds, oranges, and yellows. That's why sunsets are warm-colored."),
        ("What about the sun itself — is it actually white?", "Great question! The sun emits all visible wavelengths, so it's actually white. But Earth's atmosphere scatters blue light, making it appear yellow to us. From space, the sun looks white. Astronauts confirm this."),
        # --- Cooking ---
        ("How do I make pasta from scratch?", "Basic pasta dough: 1 cup flour + 1 egg per person. Make a well in the flour, crack the egg in the center, and gradually mix flour into the egg. Knead for 10 minutes until smooth. Rest 30 minutes. Roll thin and cut into shapes."),
        ("What type of flour should I use?", "All-purpose flour works, but '00' flour (doppio zero) is traditional for Italian pasta. It's finely milled and gives a silkier texture. Semolina flour makes a chewier pasta, great for dried shapes."),
        ("How long do I cook fresh pasta?", "Fresh pasta cooks much faster than dried — usually 2-4 minutes in boiling salted water. It's done when it floats to the top. The key is to taste-test a piece. Overcooked fresh pasta becomes mushy."),
        # --- History ---
        ("Tell me about ancient Rome.", "Ancient Rome spanned from 753 BCE to 476 CE. It grew from a small city-state to a vast empire covering Europe, North Africa, and the Middle East. Key contributions: Roman law, engineering (aqueducts, roads), Latin language, and republican government."),
        ("What caused Rome to fall?", "Multiple factors over centuries: military overspending, political corruption, economic troubles, barbarian invasions, and division into Western and Eastern empires. The traditional 'fall' date is 476 CE when the last Western emperor was deposed, though the Eastern Empire (Byzantine) survived until 1453."),
        ("How is Roman law still relevant today?", "Roman law is the foundation of most Western legal systems. Concepts like 'innocent until proven guilty,' written contracts, property rights, and legal precedent all originated in Roman law. Civil law systems (used in most of Europe) directly descend from Roman legal codes."),
        # --- Health ---
        ("How can I improve my sleep?", "Evidence-based tips: 1) Keep a consistent schedule — same bedtime and wake time. 2) Avoid screens 1 hour before bed (blue light suppresses melatonin). 3) Keep the room cool (65-68°F). 4) Avoid caffeine after 2 PM. 5) Get morning sunlight to set your circadian rhythm."),
        ("Why is sleep so important?", "Sleep is when your brain consolidates memories, clears metabolic waste (via the glymphatic system), repairs tissues, and regulates hormones. Chronic sleep deprivation increases risk of heart disease, diabetes, obesity, and cognitive decline. Adults need 7-9 hours."),
        ("What about naps?", "Short naps (10-20 minutes) can boost alertness and performance without causing grogginess. Longer naps (60-90 minutes) allow deeper sleep stages but may cause sleep inertia. Avoid napping after 3 PM as it can interfere with nighttime sleep."),
        # --- Finance ---
        ("How should I start investing?", "Start with: 1) Build an emergency fund first (3-6 months expenses). 2) Pay off high-interest debt. 3) Contribute to a retirement account (401k/IRA). 4) Start with index funds — they're diversified and low-cost. 5) Invest consistently regardless of market conditions (dollar-cost averaging)."),
        ("What's the difference between stocks and bonds?", "Stocks are ownership shares in a company — higher risk but higher potential return. Bonds are loans to companies/governments — lower risk, fixed interest payments. A balanced portfolio typically has both. Younger investors can afford more stocks; closer to retirement, shift toward bonds."),
        ("How much should I save?", "The 50/30/20 rule: 50% for needs (housing, food, utilities), 30% for wants (entertainment, dining), 20% for savings and debt repayment. At minimum, save enough to get any employer 401k match — that's free money. Aim to increase savings rate over time."),
    ]

    examples = []
    i = 0
    while i < len(convos) - 1:
        q, a = convos[i]
        examples.append({
            "id": _id(f"multiturn-{len(examples)}"),
            "field": "conversation",
            "instruction": q,
            "response": a,
            "type": "multi_turn",
        })
        i += 2
    return examples


def _tool_chain_examples() -> list[dict]:
    """Multi-step tool use."""
    examples = [
        {
            "instruction": "What is (15 * 7) + (23 / 4)?",
            "response": (
                "I'll compute each part separately.\n"
                "<tool:calculator>\nexpression=15 * 7\n</tool>\nResult: 105\n\n"
                "<tool:calculator>\nexpression=23 / 4\n</tool>\nResult: 5.75\n\n"
                "<tool:calculator>\nexpression=105 + 5.75\n</tool>\nResult: 110.75\n\n"
                "The answer is 110.75."
            ),
        },
        {
            "instruction": "List files in the current directory, then read the first Python file you find.",
            "response": (
                "<tool:list_dir>\npath=.\n</tool>\nResult:\n"
                "  d experiments\nd foundation\nd intelligence\nd serving\n"
                "  pyproject.toml\n\n"
                "I see a pyproject.toml. Let me read it.\n"
                "<tool:read_file>\npath=pyproject.toml\n</tool>\nResult:\n"
                "[project]\nname = 'silverwing-ml'\nversion = '0.1.0'..."
            ),
        },
        {
            "instruction": "Calculate the area of a circle with radius 5, then multiply by 3.",
            "response": (
                "First, I'll calculate π × r² where r = 5.\n"
                "<tool:calculator>\nexpression=3.14159 * 5 ** 2\n</tool>\n"
                "Result: 78.53975\n\n"
                "Now multiply by 3.\n"
                "<tool:calculator>\nexpression=78.53975 * 3\n</tool>\n"
                "Result: 235.61925\n\n"
                "The area of a circle with radius 5 is ≈ 78.54, and 3 times that is ≈ 235.62."
            ),
        },
        {
            "instruction": "What's 2^10 minus the square root of 144?",
            "response": (
                "Let me calculate each part.\n"
                "<tool:calculator>\nexpression=2 ** 10\n</tool>\nResult: 1024\n\n"
                "<tool:calculator>\nexpression=144 ** 0.5\n</tool>\nResult: 12\n\n"
                "<tool:calculator>\nexpression=1024 - 12\n</tool>\nResult: 1012\n\n"
                "The answer is 1012."
            ),
        },
    ]
    return [
        {"id": _id(f"toolchain-{i}"), "field": "tool_use", "instruction": e["instruction"],
         "response": e["response"], "type": "tool_chain"}
        for i, e in enumerate(examples)
    ]


def _code_examples() -> list[dict]:
    """More code generation examples."""
    examples = [
        {"instruction": "Write a Python function that finds the longest word in a sentence.",
         "response": "def longest_word(sentence):\n    words = sentence.split()\n    return max(words, key=len)\n\n# With ties broken by first occurrence:\ndef longest_word_first(sentence):\n    words = sentence.split()\n    longest = words[0]\n    for word in words[1:]:\n        if len(word) > len(longest):\n            longest = word\n    return longest"},
        {"instruction": "Write a Python function that checks if two strings are anagrams.",
         "response": "def are_anagrams(s1, s2):\n    return sorted(s1.lower().replace(' ', '')) == sorted(s2.lower().replace(' ', ''))\n\n# More efficient:\nfrom collections import Counter\ndef are_anagrams_fast(s1, s2):\n    return Counter(s1.lower().replace(' ', '')) == Counter(s2.lower().replace(' ', ''))"},
        {"instruction": "Write a Python function to compute Fibonacci numbers efficiently.",
         "response": "def fibonacci(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fibonacci(n-1) + fibonacci(n-2)\n    return memo[n]\n\n# Iterative (no recursion limit):\ndef fibonacci_iter(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"},
        {"instruction": "Write a Python function to count the frequency of each character in a string.",
         "response": "def char_frequency(s):\n    freq = {}\n    for char in s:\n        freq[char] = freq.get(char, 0) + 1\n    return freq\n\n# Using Counter:\nfrom collections import Counter\ndef char_frequency(s):\n    return dict(Counter(s))"},
        {"instruction": "Write a Python class for a simple bank account.",
         "response": "class BankAccount:\n    def __init__(self, owner, balance=0):\n        self.owner = owner\n        self._balance = balance\n        self._transactions = []\n\n    @property\n    def balance(self):\n        return self._balance\n\n    def deposit(self, amount):\n        if amount <= 0:\n            raise ValueError('Deposit must be positive')\n        self._balance += amount\n        self._transactions.append(('deposit', amount))\n        return self._balance\n\n    def withdraw(self, amount):\n        if amount <= 0:\n            raise ValueError('Withdrawal must be positive')\n        if amount > self._balance:\n            raise ValueError('Insufficient funds')\n        self._balance -= amount\n        self._transactions.append(('withdraw', amount))\n        return self._balance\n\n    def get_history(self):\n        return self._transactions.copy()"},
        {"instruction": "Write a Python function to merge two dictionaries.",
         "response": "def merge_dicts(d1, d2):\n    result = d1.copy()\n    result.update(d2)\n    return result\n\n# Or using unpacking (Python 3.5+):\ndef merge_dicts(d1, d2):\n    return {**d1, **d2}\n\n# Or using | operator (Python 3.9+):\ndef merge_dicts(d1, d2):\n    return d1 | d2"},
        {"instruction": "Write a Python function to remove duplicates from a list while preserving order.",
         "response": "def remove_duplicates(lst):\n    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result"},
        {"instruction": "Write a Python function to find the GCD of two numbers.",
         "response": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\n# Or using math:\nfrom math import gcd\ndef gcd(a, b):\n    return math.gcd(a, b)"},
    ]
    return [
        {"id": _id(f"code-{i}"), "field": "programming", "instruction": e["instruction"],
         "response": e["response"], "type": "code_generation"}
        for i, e in enumerate(examples)
    ]


def _reasoning_chains() -> list[dict]:
    """Extended reasoning examples."""
    examples = [
        {"instruction": "A store has a 20% off sale. You also have a 10% coupon. How much do you pay for a $50 item?",
         "response": "Step 1: Apply 20% discount first: $50 × 0.80 = $40\nStep 2: Apply 10% coupon: $40 × 0.90 = $36\nNote: Discounts don't stack as 30% off. Each discount applies to the already-reduced price.\nAnswer: $36"},
        {"instruction": "If you drive at 45 mph for 2 hours, then at 60 mph for 3 hours, what's your average speed?",
         "response": "Step 1: Distance 1 = 45 × 2 = 90 miles\nStep 2: Distance 2 = 60 × 3 = 180 miles\nStep 3: Total distance = 90 + 180 = 270 miles\nStep 4: Total time = 2 + 3 = 5 hours\nStep 5: Average speed = 270 / 5 = 54 mph\nNote: Average speed is total distance / total time, NOT the average of the two speeds."},
        {"instruction": "A rectangle has perimeter 24. If the length is twice the width, find the area.",
         "response": "Step 1: Let width = w, length = 2w\nStep 2: Perimeter = 2(l + w) = 2(2w + w) = 6w\nStep 3: 6w = 24, so w = 4\nStep 4: Length = 2(4) = 8\nStep 5: Area = l × w = 8 × 4 = 32\nAnswer: 32 square units"},
        {"instruction": "What is the probability of drawing 2 aces from a deck of 52 cards without replacement?",
         "response": "Step 1: P(1st ace) = 4/52 = 1/13\nStep 2: After removing one ace: 3 aces left, 51 cards total\nStep 3: P(2nd ace | 1st ace) = 3/51 = 1/17\nStep 4: P(both aces) = (1/13) × (1/17) = 1/221\nAnswer: 1/221 ≈ 0.45%"},
        {"instruction": "Explain how HTTPS works.",
         "response": "Step 1: Browser connects to server and requests HTTPS\nStep 2: Server sends its SSL certificate (public key)\nStep 3: Browser verifies the certificate with a Certificate Authority\nStep 4: Browser generates a random session key and encrypts it with the server's public key\nStep 5: Server decrypts the session key with its private key\nStep 6: Both sides use the session key for symmetric encryption\nStep 7: All data is now encrypted with the session key\nThis combines asymmetric encryption (for key exchange) with symmetric encryption (for data)."},
        {"instruction": "Why do we see lightning before hearing thunder?",
         "response": "Step 1: Lightning is a visible flash of light (electromagnetic radiation)\nStep 2: Thunder is a sound wave (mechanical wave)\nStep 3: Light travels at ~300,000 km/s (nearly instant)\nStep 4: Sound travels at ~343 m/s (much slower)\nStep 5: Since light is ~874,000 times faster than sound, we see lightning almost immediately, but thunder takes time to reach us\nFun fact: Count seconds between flash and boom, divide by 3 to get distance in km (or divide by 5 for miles)."},
    ]
    return [
        {"id": _id(f"reasoning-{i}"), "field": "reasoning", "instruction": e["instruction"],
         "response": e["response"], "type": "chain_of_thought"}
        for i, e in enumerate(examples)
    ]


def _world_knowledge() -> list[dict]:
    """General world knowledge."""
    examples = [
        {"instruction": "What are the planets in our solar system?", "response": "From the Sun outward: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune. Pluto was reclassified as a dwarf planet in 2006. Jupiter is the largest; Mercury is the smallest of the eight planets."},
        {"instruction": "What is the tallest mountain in the world?", "response": "Mount Everest at 8,849 meters (29,032 feet) above sea level. It's located in the Himalayas on the border of Nepal and Tibet. The mountain grows about 4mm per year due to tectonic activity."},
        {"instruction": "How many countries are there?", "response": "There are 195 countries recognized by the United Nations: 193 member states plus 2 observer states (Vatican City and Palestine). Some territories and regions have varying degrees of recognition."},
        {"instruction": "What is the speed of sound?", "response": "In air at 20°C (68°F): approximately 343 meters per second (1,125 feet per second, 767 mph). It varies with temperature, humidity, and medium. In water: ~1,480 m/s. In steel: ~5,960 m/s."},
        {"instruction": "What is DNA made of?", "response": "DNA is made of nucleotides, each containing: a sugar (deoxyribose), a phosphate group, and a nitrogenous base. The four bases are: Adenine (A), Thymine (T), Guanine (G), and Cytosine (C). A pairs with T, G pairs with C. These base pairs form the famous double helix structure."},
        {"instruction": "What is the Internet of Things (IoT)?", "response": "IoT refers to the network of physical objects embedded with sensors, software, and connectivity that allows them to collect and exchange data. Examples: smart thermostats, fitness trackers, connected cars, industrial sensors. By 2030, an estimated 50+ billion devices will be connected."},
        {"instruction": "What is 5G technology?", "response": "5G is the 5th generation of cellular network technology. It offers: speeds up to 10 Gbps (100x faster than 4G), ultra-low latency (<1ms), and capacity for 1 million devices per km². It enables: autonomous vehicles, remote surgery, smart cities, and real-time AR/VR."},
        {"instruction": "What is quantum entanglement?", "response": "Quantum entanglement is a phenomenon where two particles become correlated such that the state of one instantly determines the state of the other, regardless of distance. Einstein called it 'spooky action at a distance.' It doesn't transmit information faster than light, but it's crucial for quantum computing and cryptography."},
        {"instruction": "What is nanotechnology?", "response": "Nanotechnology is the manipulation of matter at the atomic and molecular scale (1-100 nanometers). Applications: targeted drug delivery, stronger materials, solar cells, water purification, electronics. A human hair is about 80,000-100,000 nanometers wide."},
        {"instruction": "What is CRISPR gene editing?", "response": "CRISPR-Cas9 is a revolutionary gene-editing tool. It uses a guide RNA to direct the Cas9 enzyme to a specific location in the genome, where it cuts the DNA. This allows scientists to delete, insert, or replace genetic material with unprecedented precision. Applications include treating genetic diseases, improving crops, and creating disease-resistant organisms."},
    ]
    return [
        {"id": _id(f"world-{i}"), "field": "world_knowledge", "instruction": e["instruction"],
         "response": e["response"], "type": "knowledge_base"}
        for i, e in enumerate(examples)
    ]


def _system_prompts() -> list[dict]:
    """System prompt / identity training."""
    examples = [
        {"instruction": "Who are you?", "response": "I'm SilverWing, an AI assistant built on a custom decoder transformer architecture. I was trained on a diverse dataset covering mathematics, science, programming, history, and many other fields. I aim to provide accurate, helpful, and clear responses. I can help with questions, problem-solving, coding, analysis, and creative tasks."},
        {"instruction": "What can you help me with?", "response": "I can help with a wide range of tasks: answering questions across all fields of knowledge (math, science, history, programming, health, business, and more), solving math and science problems step by step, writing and debugging code, explaining complex concepts simply, creative writing, analyzing data, and much more. Just ask!"},
        {"instruction": "What are your limitations?", "response": "I have a few limitations: I don't have access to real-time information or the internet unless given tools. I can make mistakes, especially with very recent events or highly specialized topics. I don't have personal experiences or emotions. I should not be used as a substitute for professional medical, legal, or financial advice. I'll always let you know when I'm uncertain."},
        {"instruction": "How do I use your tools?", "response": "I can use tools when needed. For example, I can use a calculator for math computations, read files from the filesystem, or list directory contents. When I use a tool, I'll show you exactly what I'm doing and the result. You can also ask me to use a specific tool if you'd like."},
    ]
    return [
        {"id": _id(f"system-{i}"), "field": "system", "instruction": e["instruction"],
         "response": e["response"], "type": "system_prompt"}
        for i, e in enumerate(examples)
    ]


def _translation_language() -> list[dict]:
    """Language understanding and translation."""
    examples = [
        {"instruction": "What does 'bonjour' mean?", "response": "'Bonjour' is French for 'hello' or 'good day.' It's a standard greeting used in French-speaking countries, typically used before around 6 PM. After 6 PM, you'd say 'bonsoir' (good evening)."},
        {"instruction": "Translate 'The quick brown fox jumps over the lazy dog' to Spanish.", "response": "'El rápido zorro marrón salta sobre el perro perezoso.' This is a famous pangram in English — a sentence containing every letter of the alphabet."},
        {"instruction": "What is the difference between 'their,' 'there,' and 'they're'?", "response": "'Their' = possessive (their house). 'There' = location or exists (go there / there are three). 'They're' = contraction of 'they are' (they're coming). Easy trick: if you can replace it with 'they are,' use 'they're.'"},
        {"instruction": "What is a metaphor vs a simile?", "response": "A simile compares using 'like' or 'as': 'Her smile was like sunshine.' A metaphor states identity directly: 'Her smile was sunshine.' Metaphors are stronger and more direct; similes acknowledge the comparison explicitly. Both are figures of speech that create vivid imagery."},
    ]
    return [
        {"id": _id(f"lang-{i}"), "field": "language", "instruction": e["instruction"],
         "response": e["response"], "type": "knowledge_base"}
        for i, e in enumerate(examples)
    ]


def generate_massive_expansion(output_path: str) -> int:
    """Generate and append massive expansion."""
    all_generators = [
        ("multi_turn", _multi_turn_conversations),
        ("tool_chains", _tool_chain_examples),
        ("code", _code_examples),
        ("reasoning", _reasoning_chains),
        ("world_knowledge", _world_knowledge),
        ("system_prompts", _system_prompts),
        ("language", _translation_language),
    ]

    total = 0
    all_items = []
    for name, gen in all_generators:
        items = gen()
        all_items.extend(items)
        print(f"  {name}: {len(items)} examples")
        total += len(items)

    with open(output_path, "a", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Added: {total} examples")
    return total


def main():
    output = "datasets/processed/silverwing-comprehensive-v1.jsonl"
    print("\n  SilverWing Massive Expansion")
    print("  ============================\n")
    generate_massive_expansion(output)

    with open(output, encoding="utf-8") as f:
        count = sum(1 for _ in f)
    print(f"  Total in file: {count}")


if __name__ == "__main__":
    main()
