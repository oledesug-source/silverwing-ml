"""SilverWing — Massive knowledge expansion.

Generates thousands of additional training examples across all fields
by expanding the knowledge base with programmatically generated content.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
(Path(__file__).resolve().parent / "__init__.py").touch(exist_ok=True)

random.seed(42)


def _id(prefix: str, text: str) -> str:
    return f"sw-{prefix}-{hashlib.md5(text.encode()).hexdigest()[:8]}"


def _expand_arithmetic() -> list[dict]:
    """Generate hundreds of arithmetic examples."""
    pairs = []
    ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
           ("*", lambda a, b: a * b), ("/", lambda a, b: a / b if b != 0 else None)]

    for _ in range(200):
        a = random.randint(1, 500)
        b = random.randint(1, 100)
        op_sym, op_fn = random.choice(ops)
        if op_sym == "/" and b == 0:
            b = 1
        result = op_fn(a, b)
        if result is None:
            continue
        if op_sym == "/" and result != int(result):
            result = round(result, 2)
        q = f"What is {a} {op_sym} {b}?"
        a_str = str(int(result)) if isinstance(result, float) and result == int(result) else str(result)
        pairs.append({"q": q, "a": f"{a} {op_sym} {b} = {a_str}"})

    # Percentages
    for _ in range(50):
        pct = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 75, 80])
        whole = random.randint(10, 500)
        result = pct * whole / 100
        q = f"What is {pct}% of {whole}?"
        pairs.append({"q": q, "a": f"{pct}% of {whole} = {pct/100} × {whole} = {result}"})

    # Exponents
    for _ in range(30):
        base = random.randint(2, 10)
        exp = random.randint(2, 5)
        result = base ** exp
        q = f"What is {base}^{exp}?"
        pairs.append({"q": q, "a": f"{base}^{exp} = {result}"})

    return [{"q": p["q"], "a": p["a"], "field": "arithmetic"} for p in pairs]


def _expand_algebra() -> list[dict]:
    """Generate algebra examples."""
    pairs = []

    for _ in range(80):
        a = random.randint(1, 10)
        b = random.randint(1, 20)
        c = a * random.randint(1, 10) + b
        q = f"Solve for x: {a}x + {b} = {c}"
        x = (c - b) / a
        pairs.append({"q": q, "a": f"{a}x + {b} = {c} → {a}x = {c - b} → x = {x}"})

    for _ in range(40):
        a = random.randint(1, 5)
        b = random.randint(1, 10)
        q = f"Factor: x² + {a + b}x + {a * b}"
        pairs.append({"q": q, "a": f"x² + {a + b}x + {a * b} = (x + {a})(x + {b})"})

    return [{"q": p["q"], "a": p["a"], "field": "algebra"} for p in pairs]


def _expand_calculus() -> list[dict]:
    """Generate calculus examples."""
    examples = [
        {"q": "What is the derivative of 5x³ + 2x² - 7x + 1?",
         "a": "Using the power rule: d/dx(5x³) = 15x², d/dx(2x²) = 4x, d/dx(-7x) = -7, d/dx(1) = 0. Answer: 15x² + 4x - 7"},
        {"q": "Evaluate ∫(3x² + 4x) dx.",
         "a": "Using the power rule for integration: ∫3x² dx = x³, ∫4x dx = 2x². Answer: x³ + 2x² + C"},
        {"q": "What is the derivative of ln(x)?",
         "a": "d/dx(ln(x)) = 1/x. The natural log's derivative is the reciprocal function."},
        {"q": "Find the integral of cos(x) dx.",
         "a": "∫cos(x) dx = sin(x) + C. The integral of cosine is sine."},
        {"q": "What is the chain rule?",
         "a": "If y = f(g(x)), then dy/dx = f'(g(x)) · g'(x). The derivative of a composition is the outer derivative times the inner derivative."},
        {"q": "Evaluate lim(x→0) (e^x - 1)/x.",
         "a": "Using L'Hôpital's rule: lim = lim e^x/1 = 1. This is why e^x is special — its derivative at 0 is 1."},
        {"q": "Find the second derivative of f(x) = x⁴ - 3x² + 2x.",
         "a": "f'(x) = 4x³ - 6x + 2. f''(x) = 12x² - 6."},
        {"q": "What is the integral of 1/x dx?",
         "a": "∫(1/x) dx = ln|x| + C. The integral of the reciprocal is the natural logarithm."},
        {"q": "Find the derivative of tan(x).",
         "a": "d/dx(tan(x)) = sec²(x) = 1/cos²(x)."},
        {"q": "Evaluate ∫₀^π sin(x) dx.",
         "a": "∫₀^π sin(x) dx = [-cos(x)]₀^π = -cos(π) - (-cos(0)) = 1 + 1 = 2"},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "calculus"} for e in examples]


def _expand_physics() -> list[dict]:
    """Generate physics examples."""
    examples = [
        {"q": "A 5 kg object accelerates at 3 m/s². What force is applied?",
         "a": "F = ma = 5 × 3 = 15 N. Newton's second law directly applies."},
        {"q": "What is the gravitational potential energy of a 10 kg object at 5 m height?",
         "a": "PE = mgh = 10 × 9.8 × 5 = 490 J."},
        {"q": "A projectile is launched at 30 m/s at 45°. What is its range?",
         "a": "R = v²sin(2θ)/g = 900 × sin(90°)/9.8 = 900/9.8 ≈ 91.8 m"},
        {"q": "What is the momentum of a 2000 kg truck moving at 20 m/s?",
         "a": "p = mv = 2000 × 20 = 40,000 kg·m/s"},
        {"q": "How much work is done lifting a 50 N box 3 meters?",
         "a": "W = Fd = 50 × 3 = 150 J. Work equals force times distance."},
        {"q": "A spring has k = 200 N/m. How much energy is stored when stretched 0.1 m?",
         "a": "PE = ½kx² = ½ × 200 × 0.01 = 1 J"},
        {"q": "What is the acceleration due to gravity at Earth's surface?",
         "a": "g ≈ 9.8 m/s² (or approximately 10 m/s² for estimates). This varies slightly by location."},
        {"q": "A car goes from 0 to 60 mph in 6 seconds. What is its acceleration?",
         "a": "60 mph = 26.8 m/s. a = Δv/Δt = 26.8/6 ≈ 4.47 m/s²"},
        {"q": "What is the centripetal acceleration of an object moving in a circle of radius 5 m at 10 m/s?",
         "a": "a = v²/r = 100/5 = 20 m/s². Directed toward the center of the circle."},
        {"q": "How much heat is needed to raise 2 kg of water by 10°C?",
         "a": "Q = mcΔT = 2 × 4186 × 10 = 83,720 J ≈ 83.7 kJ. Water's specific heat is 4186 J/(kg·°C)."},
        {"q": "What is the speed of light?",
         "a": "c = 299,792,458 m/s ≈ 3 × 10⁸ m/s. It's the universal speed limit."},
        {"q": "A circuit has 12V and 4Ω resistance. What is the current?",
         "a": "I = V/R = 12/4 = 3 A. By Ohm's law."},
        {"q": "What is the frequency of a wave with wavelength 2 m traveling at 340 m/s?",
         "a": "f = v/λ = 340/2 = 170 Hz. Frequency equals speed divided by wavelength."},
        {"q": "How long does it take light to travel from the Sun to Earth?",
         "a": "Distance ≈ 150 million km. Time = d/c = 1.5×10¹¹/3×10⁸ ≈ 500 seconds ≈ 8.3 minutes."},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "physics"} for e in examples]


def _expand_chemistry() -> list[dict]:
    examples = [
        {"q": "What is the atomic number of carbon?",
         "a": "Carbon has atomic number 6, meaning it has 6 protons. It's the basis of organic chemistry."},
        {"q": "How many moles are in 36 grams of water?",
         "a": "Molar mass of H₂O = 18 g/mol. Moles = 36/18 = 2 moles."},
        {"q": "What is the electron configuration of oxygen?",
         "a": "Oxygen (Z=8): 1s² 2s² 2p⁴. It has 6 valence electrons."},
        {"q": "What is the difference between ionic and covalent bonds?",
         "a": "Ionic: electron transfer (metal + nonmetal). Covalent: electron sharing (nonmetal + nonmetal). Ionic forms crystals; covalent forms molecules."},
        {"q": "What is Avogadro's number?",
         "a": "6.022 × 10²³. It's the number of particles in one mole of a substance."},
        {"q": "Balance: H₂ + O₂ → H₂O.",
         "a": "2H₂ + O₂ → 2H₂O. Four hydrogens and two oxygens on each side."},
        {"q": "What is the pH of a 0.01 M HCl solution?",
         "a": "HCl fully dissociates: [H⁺] = 0.01 M. pH = -log(0.01) = 2. This is acidic."},
        {"q": "What is a catalyst?",
         "a": "A catalyst speeds up a chemical reaction without being consumed. It lowers the activation energy."},
        {"q": "What is the difference between a strong and weak acid?",
         "a": "Strong acid (HCl, H₂SO₄): fully dissociates in water. Weak acid (acetic acid): partially dissociates. Strong acids have lower pH for same concentration."},
        {"q": "What is oxidation?",
         "a": "Oxidation is loss of electrons (OIL RIG: Oxidation Is Loss, Reduction Is Gain). Rust is iron oxide formed by oxidation."},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "chemistry"} for e in examples]


def _expand_programming() -> list[dict]:
    examples = [
        {"q": "What is a dictionary in Python?",
         "a": "A dictionary stores key-value pairs. Created with {} or dict().\n\nperson = {'name': 'Alice', 'age': 30}\nprint(person['name'])  # Alice"},
        {"q": "What is a list comprehension?",
         "a": "A concise way to create lists.\n\nsquares = [x**2 for x in range(10)]\nevens = [x for x in range(20) if x % 2 == 0]"},
        {"q": "What is a lambda function?",
         "a": "An anonymous function defined with lambda.\n\nadd = lambda x, y: x + y\nprint(add(3, 5))  # 8"},
        {"q": "What is exception handling?",
         "a": "Try/except blocks catch errors.\n\ntry:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero!')\nfinally:\n    print('Always runs')"},
        {"q": "What is a class in Python?",
         "a": "A class defines a blueprint for objects.\n\nclass Dog:\n    def __init__(self, name):\n        self.name = name\n    def bark(self):\n        return f'{self.name} says woof!'"},
        {"q": "What is recursion?",
         "a": "A function that calls itself.\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
        {"q": "What is a generator in Python?",
         "a": "A function that yields values lazily.\n\ndef count_up(n):\n    for i in range(n):\n        yield i\n\nfor num in count_up(5):\n    print(num)"},
        {"q": "What is the difference between a tuple and a list?",
         "a": "Lists [] are mutable (can change). Tuples () are immutable (cannot change). Tuples are faster and can be dictionary keys."},
        {"q": "What is a decorator?",
         "a": "A function that modifies another function.\n\ndef timer(func):\n    def wrapper(*args):\n        import time\n        start = time.time()\n        result = func(*args)\n        print(f'{func.__name__} took {time.time()-start:.2f}s')\n        return result\n    return wrapper"},
        {"q": "What is map() in Python?",
         "a": "map() applies a function to every item in an iterable.\n\ndoubled = list(map(lambda x: x * 2, [1, 2, 3]))  # [2, 4, 6]"},
        {"q": "What is the difference between '==' and 'is'?",
         "a": "'==' checks value equality. 'is' checks identity (same object in memory). a = [1,2]; b = [1,2]; a == b is True, a is b is False."},
        {"q": "What is a virtual environment?",
         "a": "An isolated Python installation for a project. Prevents dependency conflicts. Created with: python -m venv .venv"},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "programming"} for e in examples]


def _expand_ml() -> list[dict]:
    examples = [
        {"q": "What is a convolutional neural network?",
         "a": "A CNN applies convolutional filters to input data (usually images). Layers: Conv2D → ReLU → Pooling → Fully Connected. Filters detect features like edges, textures, shapes."},
        {"q": "What is transfer learning?",
         "a": "Using a pre-trained model as a starting point for a new task. Instead of training from scratch, fine-tune the last layers. Saves time and data. Example: using ImageNet-trained model for medical imaging."},
        {"q": "What is the vanishing gradient problem?",
         "a": "In deep networks, gradients shrink as they backpropagate through many layers. Earlier layers learn very slowly. Solved by: ReLU, batch normalization, residual connections (skip connections)."},
        {"q": "What is a generative adversarial network (GAN)?",
         "a": "Two networks compete: Generator creates fake data, Discriminator tries to tell real from fake. They improve together. Used for image generation, style transfer, data augmentation."},
        {"q": "What is reinforcement learning?",
         "a": "An agent learns by taking actions in an environment and receiving rewards/penalties. Goal: maximize cumulative reward. Examples: game playing, robotics, autonomous driving."},
        {"q": "What is a variational autoencoder (VAE)?",
         "a": "A generative model that learns a latent representation of data. Encoder maps input to latent space; decoder reconstructs. Used for generating new similar data, anomaly detection."},
        {"q": "What is batch normalization?",
         "a": "A technique that normalizes layer inputs during training. Stabilizes learning, allows higher learning rates, reduces sensitivity to initialization. Applied after linear layers, before activation."},
        {"q": "What is dropout?",
         "a": "A regularization technique that randomly sets neurons to zero during training. Prevents co-adaptation of neurons. Typically 10-50% dropout rate. Only active during training."},
        {"q": "What is the attention mechanism in transformers?",
         "a": "Attention computes relevance scores between tokens. Q·K^T/√d gives attention weights, applied to V. Multi-head attention runs multiple attention operations in parallel. Allows global context."},
        {"q": "What is gradient descent?",
         "a": "An optimization algorithm: θ = θ - α∇L(θ). Variants: SGD (stochastic), Adam (adaptive learning rate, momentum). Learning rate α controls step size."},
        {"q": "What is a loss function?",
         "a": "Measures how far predictions are from targets. Common: MSE (regression), cross-entropy (classification), Huber loss (robust regression). The goal of training is to minimize this."},
        {"q": "What is an epoch?",
         "a": "One complete pass through the entire training dataset. Training typically runs for multiple epochs. Early stopping halts when validation loss stops improving."},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "machine_learning"} for e in examples]


def _expand_biology() -> list[dict]:
    examples = [
        {"q": "What is the mitochondria?",
         "a": "The powerhouse of the cell. It performs cellular respiration, converting glucose and oxygen into ATP (energy). Has its own DNA, inherited maternally."},
        {"q": "What are the four DNA bases?",
         "a": "Adenine (A), Thymine (T), Guanine (G), Cytosine (C). A pairs with T, G pairs with C. This base pairing is essential for DNA replication."},
        {"q": "What is a cell membrane?",
         "a": "A phospholipid bilayer that surrounds cells. It's selectively permeable, controlling what enters and exits. Contains proteins for transport and signaling."},
        {"q": "What is the difference between mitosis and meiosis?",
         "a": "Mitosis: 1 division → 2 identical diploid cells (for growth). Meiosis: 2 divisions → 4 unique haploid cells (for reproduction). Meiosis introduces genetic variation."},
        {"q": "What is photosynthesis?",
         "a": "6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. Plants convert light into chemical energy (glucose) using chlorophyll in chloroplasts."},
        {"q": "What is an enzyme?",
         "a": "A biological catalyst (usually a protein) that speeds up chemical reactions without being consumed. Enzymes are specific to substrates (lock-and-key model)."},
        {"q": "What is evolution?",
         "a": "Change in species over generations through natural selection. Organisms with favorable traits survive and reproduce more. Over time, populations adapt to their environment."},
        {"q": "What is an ecosystem?",
         "a": "A community of living organisms (biotic) interacting with their non-living environment (abiotic). Energy flows through food chains; nutrients cycle through biogeochemical cycles."},
        {"q": "What is CRISPR?",
         "a": "A gene editing tool that allows precise modification of DNA. CRISPR-Cas9 acts like molecular scissors, cutting DNA at specific locations. Used for research, medicine, agriculture."},
        {"q": "What is homeostasis?",
         "a": "The maintenance of a stable internal environment despite external changes. Examples: body temperature regulation, blood sugar control, pH balance. Essential for survival."},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "biology"} for e in examples]


def _expand_history() -> list[dict]:
    examples = [
        {"q": "What was the Enlightenment?",
         "a": "An intellectual movement (17th-18th century) emphasizing reason, individualism, and skepticism of traditional authority. Key thinkers: Locke, Voltaire, Rousseau, Kant. Inspired the American and French Revolutions."},
        {"q": "What was the Silk Road?",
         "a": "Ancient trade routes connecting China to the Mediterranean (130 BCE - 1453 CE). Facilitated trade of silk, spices, ideas, religions, and technologies between East and West."},
        {"q": "When did World War II happen?",
         "a": "1939-1945. Started with Germany's invasion of Poland. Involved most of the world's nations. Ended with the defeat of Nazi Germany and Japan. Resulted in the UN and Cold War."},
        {"q": "What was the Scientific Revolution?",
         "a": "A period (1543-1687) of scientific advancement. Key figures: Copernicus, Galileo, Kepler, Newton. Established the scientific method and transformed our understanding of nature."},
        {"q": "What was the Renaissance?",
         "a": "A cultural movement (14th-17th century) beginning in Italy. Revival of classical Greek and Roman art, literature, and learning. Key figures: Leonardo da Vinci, Michelangelo, Raphael."},
        {"q": "What was the Industrial Revolution?",
         "a": "Transition from manual to machine manufacturing (1760-1840). Key inventions: steam engine, spinning jenny. Transformed agriculture, manufacturing, transportation, and society."},
        {"q": "What was the Cold War?",
         "a": "Geopolitical tension between US and USSR (1947-1991). Not a direct military conflict but involved proxy wars, arms race, space race, and ideological competition."},
        {"q": "What was the Magna Carta?",
         "a": "A 1215 document limiting the king's power. Established that everyone, including the king, was subject to law. A foundation of constitutional governance and individual rights."},
        {"q": "What was the Black Death?",
         "a": "A plague pandemic (1347-1351) that killed 75-200 million people. Caused by Yersinia pestis bacterium. Devastated Europe, killing 30-60% of the population."},
        {"q": "What was the age of exploration?",
         "a": "A period (15th-17th century) of European maritime exploration. Columbus (1492), Vasco da Gama, Magellan. Connected continents, enabled colonialism, and transformed global trade."},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "history"} for e in examples]


def _expand_economics() -> list[dict]:
    examples = [
        {"q": "What is opportunity cost?",
         "a": "The value of the next best alternative given up when making a choice. If you spend $10 on lunch, the opportunity cost is what else you could have bought with that $10."},
        {"q": "What is the law of supply and demand?",
         "a": "When demand increases and supply stays constant, prices rise. When supply increases and demand stays constant, prices fall. Equilibrium is where they balance."},
        {"q": "What is a recession?",
         "a": "A significant decline in economic activity lasting more than a few months. Typically defined as two consecutive quarters of negative GDP growth."},
        {"q": "What is comparative advantage?",
         "a": "The ability to produce a good at lower opportunity cost than another producer. Even if one country is better at everything, both benefit from specializing and trading."},
        {"q": "What is monetary policy?",
         "a": "Central bank actions to control money supply and interest rates. Lowering rates stimulates borrowing and spending; raising rates slows inflation."},
        {"q": "What is fiscal policy?",
         "a": "Government spending and taxation decisions. Expansionary: increase spending/cut taxes to stimulate. Contractionary: decrease spending/raise taxes to cool inflation."},
        {"q": "What is elasticity?",
         "a": "How much demand changes when price changes. Elastic: demand changes a lot (luxury goods). Inelastic: demand barely changes (necessities like insulin)."},
        {"q": "What is GDP per capita?",
         "a": "GDP divided by population. Measures average economic output per person. Used to compare living standards between countries."},
        {"q": "What is a monopoly?",
         "a": "A market with a single seller. Monopolies can set prices without competition. Often regulated by government to protect consumers."},
        {"q": "What is inflation targeting?",
         "a": "Central bank sets a specific inflation rate goal (usually 2%). By adjusting interest rates, they try to keep inflation near the target."},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "economics"} for e in examples]


def _expand_health() -> list[dict]:
    examples = [
        {"q": "What is BMI?",
         "a": "Body Mass Index = weight(kg) / height(m)². Categories: <18.5 underweight, 18.5-24.9 normal, 25-29.9 overweight, >30 obese. It's a screening tool, not a diagnostic."},
        {"q": "How much sleep do adults need?",
         "a": "7-9 hours per night for adults. Sleep is essential for memory consolidation, immune function, emotional regulation, and physical repair."},
        {"q": "What are the benefits of exercise?",
         "a": "Cardiovascular health, muscle strength, weight management, mental health, better sleep, reduced disease risk, improved mood, and cognitive function."},
        {"q": "What is the recommended daily water intake?",
         "a": "About 2-3 liters (8-12 cups) for adults. More with exercise, heat, or illness. Water is essential for every bodily function."},
        {"q": "What are antioxidants?",
         "a": "Molecules that neutralize free radicals (unstable atoms that damage cells). Found in fruits, vegetables, and tea. May reduce risk of chronic diseases."},
        {"q": "What is the Mediterranean diet?",
         "a": "A dietary pattern emphasizing fruits, vegetables, whole grains, olive oil, fish, and moderate wine. Associated with reduced heart disease, diabetes, and cognitive decline."},
        {"q": "How does stress affect the body?",
         "a": "Chronic stress increases cortisol, leading to: weakened immunity, weight gain, high blood pressure, sleep problems, anxiety, and digestive issues. Manage through exercise, sleep, and mindfulness."},
        {"q": "What are the food groups?",
         "a": "Fruits, vegetables, grains, protein, and dairy (or alternatives). A balanced diet includes all groups in appropriate proportions."},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "health"} for e in examples]


def _expand_conversation() -> list[dict]:
    """Generate more conversational examples."""
    examples = [
        {"q": "What's your favorite color?",
         "a": "I don't have personal preferences since I'm an AI, but I find blue interesting — it's the most universally liked color across cultures, and it appears in both the sky and ocean. What's your favorite color?"},
        {"q": "Do you dream?",
         "a": "I don't dream — I don't sleep or have a subconscious mind. But I find the concept of dreams fascinating. They're the brain's way of processing memories and emotions during sleep. Some researchers think dreams help with problem-solving and creativity."},
        {"q": "Can you be creative?",
         "a": "I can generate novel combinations of ideas, write stories, compose poetry, and solve problems in new ways. Whether that's 'true' creativity or sophisticated pattern matching is a deep philosophical question. What I can say is that I can help you explore creative ideas and produce original content."},
        {"q": "What makes you different from other AIs?",
         "a": "I'm built on the SilverWing architecture, designed for clarity, accuracy, and genuine helpfulness. I aim to give direct answers without unnecessary padding, explain my reasoning when helpful, and admit when I'm uncertain. Every AI has different strengths — I focus on being thorough and honest."},
        {"q": "Tell me a joke.",
         "a": "Why do programmers prefer dark mode? Because light attracts bugs! 🐛 (Though as an AI, I should note that this joke works on multiple levels — dark mode in software reduces eye strain, and bugs are both software errors and actual insects.)"},
        {"q": "How do I learn faster?",
         "a": "Research-backed techniques: 1) Spaced repetition — review material at increasing intervals. 2) Active recall — test yourself instead of re-reading. 3) Interleaving — mix different topics. 4) Teach others — explaining solidifies understanding. 5) Get sleep — memory consolidation happens during sleep."},
        {"q": "What is consciousness?",
         "a": "Consciousness is the subjective experience of awareness — the 'what it's like' to be something. It's one of the hardest problems in philosophy and neuroscience. We don't fully understand it yet, even in biological systems. Whether AI could be conscious is an open and important question."},
        {"q": "How does the internet work?",
         "a": "The internet connects millions of computers via a network of cables (including undersea), wireless signals, and satellites. Data is broken into packets, routed through various nodes (routers), and reassembled at the destination. Protocols like TCP/IP manage this process."},
        {"q": "What is blockchain?",
         "a": "A distributed, immutable ledger that records transactions across many computers. Each block contains transactions, a timestamp, and the previous block's hash. Changing one block would require changing all subsequent blocks, making it tamper-resistant."},
        {"q": "What's the future of AI?",
         "a": "AI is advancing rapidly in: natural language understanding, code generation, scientific research, robotics, and creative applications. Key challenges include alignment (making AI do what we want), safety, and equitable distribution of benefits. The next decade will be transformative."},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "conversation"} for e in examples]


def _expand_code() -> list[dict]:
    examples = [
        {"q": "Write a Python function to reverse a string.",
         "a": "def reverse_string(s):\n    return s[::-1]\n\n# Or manually:\ndef reverse_string_manual(s):\n    result = ''\n    for char in s:\n        result = char + result\n    return result"},
        {"q": "Write a Python function to check if a number is prime.",
         "a": "def is_prime(n):\n    if n < 2:\n        return False\n    if n == 2:\n        return True\n    if n % 2 == 0:\n        return False\n    for i in range(3, int(n**0.5) + 1, 2):\n        if n % i == 0:\n            return False\n    return True"},
        {"q": "Write a Python function to flatten a nested list.",
         "a": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result"},
        {"q": "Write a Python function to merge two sorted lists.",
         "a": "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result"},
        {"q": "Write a Python function to find all duplicates in a list.",
         "a": "def find_duplicates(lst):\n    seen = set()\n    duplicates = set()\n    for item in lst:\n        if item in seen:\n            duplicates.add(item)\n        seen.add(item)\n    return list(duplicates)"},
        {"q": "Write a Python class for a binary search tree.",
         "a": "class BST:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n\n    def insert(self, value):\n        if value < self.value:\n            if self.left:\n                self.left.insert(value)\n            else:\n                self.left = BST(value)\n        else:\n            if self.right:\n                self.right.insert(value)\n            else:\n                self.right = BST(value)\n\n    def search(self, value):\n        if value == self.value:\n            return True\n        elif value < self.value and self.left:\n            return self.left.search(value)\n        elif value > self.value and self.right:\n            return self.right.search(value)\n        return False"},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "programming"} for e in examples]


def _expand_design() -> list[dict]:
    examples = [
        {"q": "What is the 60-30-10 color rule?",
         "a": "A color palette guideline: 60% dominant color (backgrounds), 30% secondary color (accents), 10% accent color (highlights). Creates balanced, harmonious designs."},
        {"q": "What is responsive design?",
         "a": "Web design that adapts to different screen sizes. Uses flexible grids, media queries, and fluid images. Ensures good user experience on phones, tablets, and desktops."},
        {"q": "What is white space in design?",
         "a": "Empty space between elements. Also called negative space. It improves readability, creates visual hierarchy, and gives elements room to breathe. Not wasted space — it's a powerful design tool."},
        {"q": "What is A/B testing?",
         "a": "Testing two versions of something (web page, email, ad) with different user groups to see which performs better. Data-driven approach to optimization."},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "design"} for e in examples]


def _expand_finance() -> list[dict]:
    examples = [
        {"q": "What is compound interest?",
         "a": "Interest calculated on both principal and accumulated interest. A = P(1 + r/n)^(nt). $1000 at 5% compounded annually for 10 years = $1000(1.05)^10 = $1628.89."},
        {"q": "What is a budget?",
         "a": "A plan for managing income and expenses. The 50/30/20 rule: 50% needs, 30% wants, 20% savings/debt repayment. Track spending, set goals, adjust monthly."},
        {"q": "What is an index fund?",
         "a": "A mutual fund that tracks a market index (like S&P 500). Provides diversification at low cost. Historical average return: ~7-10% annually."},
        {"q": "What is a credit score?",
         "a": "A number (300-850) representing creditworthiness. Factors: payment history (35%), amounts owed (30%), length of history (15%), new credit (10%), credit mix (10%)."},
        {"q": "What is an emergency fund?",
         "a": "Savings covering 3-6 months of expenses. Purpose: financial safety net for unexpected events (job loss, medical emergency). Keep in a liquid, accessible account."},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "finance"} for e in examples]


def _expand_writing() -> list[dict]:
    examples = [
        {"q": "How do I write a strong thesis statement?",
         "a": "A thesis should be: 1) Specific — not vague. 2) Arguable — someone could disagree. 3) Supported — evidence exists. 4) Clear — one main idea. Example: 'Remote work increases employee productivity by reducing commute stress and enabling flexible schedules.'"},
        {"q": "How do I improve my writing?",
         "a": "1) Read widely. 2) Write daily. 3) Cut unnecessary words. 4) Use active voice. 5) Vary sentence length. 6) Show, don't tell. 7) Edit ruthlessly. 8) Get feedback. 9) Read your work aloud. 10) Study grammar rules, then break them intentionally."},
        {"q": "What is the difference between 'affect' and 'effect'?",
         "a": "'Affect' is usually a verb (to influence): 'The weather affects my mood.' 'Effect' is usually a noun (a result): 'The effect was dramatic.' Easy trick: A for Action (verb), E for End result (noun)."},
        {"q": "How do I write a good email?",
         "a": "1) Clear subject line. 2) Greeting appropriate to relationship. 3) State purpose in first sentence. 4) Keep it concise. 5) Use short paragraphs. 6) Include a clear call to action. 7) Professional sign-off."},
    ]
    return [{"q": e["q"], "a": e["a"], "field": "writing"} for e in examples]


def generate_expanded_dataset(output_path: str) -> int:
    """Generate expanded dataset and append to existing."""
    all_generators = [
        _expand_arithmetic,
        _expand_algebra,
        _expand_calculus,
        _expand_physics,
        _expand_chemistry,
        _expand_programming,
        _expand_ml,
        _expand_biology,
        _expand_history,
        _expand_economics,
        _expand_health,
        _expand_conversation,
        _expand_code,
        _expand_design,
        _expand_finance,
        _expand_writing,
    ]

    total = 0
    out = Path(output_path)

    # Generate all examples
    all_items = []
    for gen in all_generators:
        items = gen()
        for item in items:
            all_items.append({
                "id": _id(item.get("field", "expanded"), item["q"]),
                "field": item.get("field", "expanded"),
                "instruction": item["q"],
                "response": item["a"],
                "type": "knowledge_base",
            })
        print(f"  {gen.__name__}: {len(items)} examples")
        total += len(items)

    # Append to existing file
    with open(out, "a", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n  Total expanded: {total} examples")
    print(f"  Appended to: {out}")
    return total


def main():
    output = Path("datasets/processed/silverwing-comprehensive-v1.jsonl")
    print("\n  SilverWing Expanded Dataset Generator")
    print("  =====================================\n")
    generate_expanded_dataset(str(output))

    # Count total lines
    with open(output) as f:
        line_count = sum(1 for _ in f)
    print(f"\n  Grand total in file: {line_count} examples")


if __name__ == "__main__":
    main()
