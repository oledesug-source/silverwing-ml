"""SilverWing — Comprehensive training data generator.

Generates high-quality SFT, reasoning, tool-use, and conversational
training data across ALL fields of human knowledge.

Usage::

    python dataset_gen/generate_all.py --output datasets/processed --samples-per-field 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure datasets/ has __init__.py for import
(Path(__file__).resolve().parent / "__init__.py").touch(exist_ok=True)

from dataset_gen.fields import (
    TOOL_USE,
)

random.seed(42)

# ─────────────────────────────────────────────────────
# KNOWLEDGE BASE — structured facts for every field
# ─────────────────────────────────────────────────────

KNOWLEDGE_BASE: dict[str, list[dict]] = {}

# --- Mathematics ---
KNOWLEDGE_BASE["arithmetic"] = [
    {"q": "What is 15 + 27?", "a": "15 + 27 = 42"},
    {"q": "What is 144 divided by 12?", "a": "144 ÷ 12 = 12"},
    {"q": "What is 25 times 8?", "a": "25 × 8 = 200"},
    {"q": "What is 17 minus 9?", "a": "17 - 9 = 8"},
    {"q": "What is 7 cubed?", "a": "7³ = 7 × 7 × 7 = 343"},
    {"q": "What is 15% of 200?", "a": "15% of 200 = 0.15 × 200 = 30"},
    {"q": "Simplify 12/16.", "a": "12/16 = 3/4 (dividing both by 4)"},
    {"q": "What is 2^10?", "a": "2^10 = 1024"},
    {"q": "Calculate 999 + 1.", "a": "999 + 1 = 1000"},
    {"q": "What is 0.75 as a fraction?", "a": "0.75 = 75/100 = 3/4"},
]

KNOWLEDGE_BASE["algebra"] = [
    {"q": "Solve for x: 2x + 5 = 13.", "a": "2x + 5 = 13 → 2x = 8 → x = 4"},
    {"q": "Factor x² - 9.", "a": "x² - 9 = (x + 3)(x - 3) — difference of squares"},
    {"q": "Solve x² - 5x + 6 = 0.", "a": "x² - 5x + 6 = (x - 2)(x - 3) = 0, so x = 2 or x = 3"},
    {"q": "What is the slope of y = 3x + 7?", "a": "The slope is 3 (coefficient of x)"},
    {"q": "Solve: 3(x - 2) = 15.", "a": "3(x - 2) = 15 → x - 2 = 5 → x = 7"},
]

KNOWLEDGE_BASE["calculus"] = [
    {"q": "What is the derivative of x³?", "a": "d/dx(x³) = 3x²"},
    {"q": "What is the integral of 2x dx?", "a": "∫2x dx = x² + C"},
    {"q": "Find the limit of sin(x)/x as x→0.", "a": "lim(x→0) sin(x)/x = 1 — this is a fundamental limit"},
    {"q": "What is the derivative of e^x?", "a": "d/dx(e^x) = e^x — the exponential is its own derivative"},
    {"q": "Evaluate ∫₀¹ x² dx.", "a": "∫₀¹ x² dx = [x³/3]₀¹ = 1/3"},
]

KNOWLEDGE_BASE["geometry"] = [
    {"q": "What is the area of a circle with radius 5?", "a": "A = πr² = π(5)² = 25π ≈ 78.54"},
    {"q": "What is the Pythagorean theorem?", "a": "a² + b² = c², where c is the hypotenuse"},
    {"q": "Find the volume of a sphere with radius 3.", "a": "V = (4/3)πr³ = (4/3)π(27) = 36π ≈ 113.10"},
    {"q": "What is the sum of angles in a triangle?", "a": "The sum of interior angles in a triangle is 180°"},
    {"q": "What is the circumference of a circle with diameter 10?", "a": "C = πd = 10π ≈ 31.42"},
]

KNOWLEDGE_BASE["statistics"] = [
    {"q": "What is the mean of 2, 4, 6, 8, 10?", "a": "Mean = (2+4+6+8+10)/5 = 30/5 = 6"},
    {"q": "What is standard deviation?", "a": "Standard deviation measures the spread of data from the mean. It's the square root of variance."},
    {"q": "What is a normal distribution?", "a": "A normal (Gaussian) distribution is symmetric, bell-shaped, defined by mean μ and standard deviation σ. About 68% of data falls within 1σ of the mean."},
    {"q": "What is the median of 3, 7, 1, 9, 5?", "a": "Ordered: 1, 3, 5, 7, 9. Median = 5 (the middle value)"},
]

KNOWLEDGE_BASE["discrete_mathematics"] = [
    {"q": "What is 5! (factorial)?", "a": "5! = 5 × 4 × 3 × 2 × 1 = 120"},
    {"q": "How many ways to choose 2 from 5?", "a": "C(5,2) = 5!/(2!·3!) = 10 ways"},
    {"q": "Is 17 prime?", "a": "Yes, 17 is prime. It has no divisors other than 1 and itself."},
    {"q": "What is the GCD of 24 and 36?", "a": "GCD(24, 36) = 12. Using Euclidean algorithm: 36 = 1·24 + 12, 24 = 2·12 + 0."},
]

# --- Physics ---
KNOWLEDGE_BASE["classical_mechanics"] = [
    {"q": "State Newton's second law.", "a": "F = ma. The net force on an object equals its mass times its acceleration."},
    {"q": "A ball is dropped from 10m. How long to hit the ground?", "a": "Using h = ½gt²: 10 = ½(9.8)t² → t = √(20/9.8) ≈ 1.43 seconds"},
    {"q": "What is kinetic energy?", "a": "KE = ½mv². Kinetic energy is the energy of motion, proportional to mass and velocity squared."},
    {"q": "What is momentum?", "a": "p = mv. Momentum is mass times velocity. It's conserved in isolated systems."},
    {"q": "What is the work-energy theorem?", "a": "Work done on an object equals its change in kinetic energy: W = ΔKE = ½mv₂² - ½mv₁²"},
]

KNOWLEDGE_BASE["electromagnetism"] = [
    {"q": "State Ohm's law.", "a": "V = IR. Voltage equals current times resistance."},
    {"q": "What is the electric field of a point charge?", "a": "E = kQ/r², where k is Coulomb's constant, Q is the charge, and r is the distance."},
    {"q": "How do capacitors work?", "a": "Capacitors store energy in an electric field between two plates. Q = CV, where C is capacitance."},
    {"q": "What is electromagnetic induction?", "a": "A changing magnetic field induces an EMF: EMF = -dΦ/dt (Faraday's law)"},
]

KNOWLEDGE_BASE["thermodynamics"] = [
    {"q": "State the first law of thermodynamics.", "a": "ΔU = Q - W. The change in internal energy equals heat added minus work done."},
    {"q": "What is entropy?", "a": "Entropy (S) is a measure of disorder. The second law states entropy of an isolated system never decreases."},
    {"q": "What is the ideal gas law?", "a": "PV = nRT. Pressure × Volume = moles × gas constant × temperature."},
    {"q": "What is absolute zero?", "a": "Absolute zero is 0 Kelvin (-273.15°C), the theoretical lowest temperature where molecular motion stops."},
]

KNOWLEDGE_BASE["quantum_mechanics"] = [
    {"q": "What is the uncertainty principle?", "a": "ΔxΔp ≥ ℏ/2. You cannot simultaneously know both the exact position and momentum of a particle."},
    {"q": "What is wave-particle duality?", "a": "All quantum objects exhibit both wave and particle properties. Light can behave as both a wave and a particle (photon)."},
    {"q": "What is superposition?", "a": "A quantum system can exist in multiple states simultaneously until measured. |ψ⟩ = α|0⟩ + β|1⟩"},
    {"q": "What is quantum entanglement?", "a": "Two particles can be correlated such that measuring one instantly determines the state of the other, regardless of distance."},
]

# --- Computer Science ---
KNOWLEDGE_BASE["programming"] = [
    {"q": "What is a function in Python?", "a": "A function is a reusable block of code defined with 'def'. It takes inputs (parameters), performs operations, and optionally returns a value.\n\ndef greet(name):\n    return f'Hello, {name}!'"},
    {"q": "What is a loop?", "a": "A loop repeats a block of code. Python has 'for' loops (iterate over a sequence) and 'while' loops (repeat while condition is true).\n\nfor i in range(5):\n    print(i)"},
    {"q": "What is an if statement?", "a": "An if statement conditionally executes code based on a boolean expression.\n\nif temperature > 100:\n    print('Boiling!')\nelif temperature > 50:\n    print('Warm')\nelse:\n    print('Cold')"},
    {"q": "What is a list in Python?", "a": "A list is an ordered, mutable collection. Created with square brackets.\n\nnumbers = [1, 2, 3, 4, 5]\nnumbers.append(6)\nprint(numbers[0])  # 1"},
]

KNOWLEDGE_BASE["data_structures"] = [
    {"q": "What is a binary search tree?", "a": "A BST is a tree where each node has at most two children. Left children are smaller, right children are larger. Search time: O(log n) average."},
    {"q": "What is a hash table?", "a": "A hash table maps keys to values using a hash function. Average O(1) lookup, insert, and delete. Python's dict is a hash table."},
    {"q": "What is a linked list?", "a": "A linked list stores elements in nodes, where each node points to the next. Unlike arrays, elements aren't contiguous in memory."},
    {"q": "What is a stack?", "a": "A stack is LIFO (Last In, First Out). push() adds to top, pop() removes from top. Used for undo operations, function call stacks."},
]

KNOWLEDGE_BASE["algorithms"] = [
    {"q": "How does merge sort work?", "a": "Merge sort divides the array in half, recursively sorts each half, then merges the sorted halves. Time: O(n log n), Space: O(n)."},
    {"q": "What is dynamic programming?", "a": "DP solves complex problems by breaking them into overlapping subproblems, storing results to avoid redundant computation. Example: Fibonacci with memoization."},
    {"q": "How does binary search work?", "a": "Binary search finds a target in a sorted array by repeatedly halving the search space. Compare target to middle element, go left or right. O(log n)."},
    {"q": "What is breadth-first search?", "a": "BFS explores a graph level by level using a queue. It finds the shortest path in unweighted graphs. Time: O(V + E)."},
]

KNOWLEDGE_BASE["machine_learning"] = [
    {"q": "What is overfitting?", "a": "Overfitting occurs when a model learns noise in training data instead of the underlying pattern. It performs well on training data but poorly on new data. Prevented by regularization, cross-validation, more data."},
    {"q": "How does a neural network learn?", "a": "Through backpropagation: compute loss, calculate gradients via the chain rule, update weights using gradient descent. The network adjusts to minimize prediction error."},
    {"q": "What is a transformer?", "a": "A transformer uses self-attention to process sequences in parallel. It has encoder-decoder architecture with multi-head attention, position encodings, and feed-forward layers."},
    {"q": "What is gradient descent?", "a": "An optimization algorithm that iteratively adjusts parameters in the direction of steepest descent of the loss function. θ = θ - α·∇L(θ), where α is the learning rate."},
    {"q": "What is the attention mechanism?", "a": "Attention computes weighted sums of values based on query-key similarity. Self-attention lets each position attend to all others. Attention(Q,K,V) = softmax(QK^T/√d)V"},
]

KNOWLEDGE_BASE["web_development"] = [
    {"q": "What is REST API design?", "a": "REST uses HTTP methods (GET, POST, PUT, DELETE) on resources (URLs). Stateless, uses JSON for data. Status codes: 200 OK, 404 Not Found, 500 Error."},
    {"q": "What is the virtual DOM?", "a": "The virtual DOM is a lightweight JavaScript representation of the real DOM. React uses it to batch updates and minimize expensive DOM manipulations."},
    {"q": "What is CSS Grid?", "a": "CSS Grid is a two-dimensional layout system. Define rows/columns with grid-template-rows/columns. Place items with grid-row/column."},
]

KNOWLEDGE_BASE["cybersecurity"] = [
    {"q": "What is SQL injection?", "a": "SQL injection inserts malicious SQL into queries via user input. Prevented by parameterized queries and input validation."},
    {"q": "What is encryption?", "a": "Encryption converts plaintext to ciphertext using an algorithm and key. AES (symmetric) and RSA (asymmetric) are common algorithms."},
    {"q": "What is a man-in-the-middle attack?", "a": "An attacker intercepts communication between two parties. Prevented by TLS/HTTPS, certificate pinning, and end-to-end encryption."},
]

# --- Natural Sciences ---
KNOWLEDGE_BASE["chemistry"] = [
    {"q": "What is the periodic table?", "a": "The periodic table organizes elements by atomic number, electron configuration, and chemical properties. Elements in the same group have similar properties."},
    {"q": "What is a chemical bond?", "a": "A chemical bond is the attraction between atoms. Types: ionic (electron transfer), covalent (electron sharing), metallic (electron sea)."},
    {"q": "Balance: Fe + O₂ → Fe₂O₃.", "a": "4Fe + 3O₂ → 2Fe₂O₃ — balanced by ensuring equal atoms on both sides."},
    {"q": "What is pH?", "a": "pH = -log[H⁺]. pH < 7 is acidic, pH = 7 is neutral, pH > 7 is basic. Each unit is a 10x change in hydrogen ion concentration."},
]

KNOWLEDGE_BASE["biology"] = [
    {"q": "What is DNA?", "a": "DNA (deoxyribonucleic acid) stores genetic information as a double helix. Nucleotides (A, T, G, C) form base pairs: A-T and G-C."},
    {"q": "What is photosynthesis?", "a": "6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. Plants convert light energy into glucose using chlorophyll in chloroplasts."},
    {"q": "What is the central dogma?", "a": "DNA → RNA → Protein. DNA is transcribed to mRNA, which is translated to protein by ribosomes."},
    {"q": "What is natural selection?", "a": "Organisms with traits better suited to their environment tend to survive and reproduce more. Over generations, beneficial traits become more common."},
]

KNOWLEDGE_BASE["earth_sciences"] = [
    {"q": "What causes earthquakes?", "a": "Earthquakes occur when tectonic plates slide past each other, releasing energy as seismic waves. Measured by the Richter scale."},
    {"q": "What is the water cycle?", "a": "Water evaporates from oceans/lakes, rises, condenses into clouds, precipitates as rain/snow, and flows back to oceans through rivers and groundwater."},
    {"q": "What are the layers of the Earth?", "a": "Inner core (solid iron), outer core (liquid iron), mantle (semi-solid rock), crust (solid rock). Temperature increases with depth."},
]

KNOWLEDGE_BASE["astronomy"] = [
    {"q": "What is a light year?", "a": "A light year is the distance light travels in one year: about 9.46 trillion km (5.88 trillion miles). It's a unit of distance, not time."},
    {"q": "What is a black hole?", "a": "A black hole is a region of spacetime where gravity is so strong that nothing can escape. Formed when massive stars collapse."},
    {"q": "How do stars form?", "a": "Stars form from clouds of gas and dust (nebulae) that collapse under gravity. When the core reaches ~10 million K, hydrogen fusion begins."},
]

# --- Medicine & Health ---
KNOWLEDGE_BASE["anatomy"] = [
    {"q": "How many bones are in the human body?", "a": "Adults have 206 bones. Babies are born with about 270, which fuse as they grow."},
    {"q": "What does the heart do?", "a": "The heart pumps blood through the body. Right side sends blood to lungs (pulmonary), left side sends oxygenated blood to body (systemic)."},
    {"q": "What is the largest organ?", "a": "The skin is the largest organ. It weighs about 4 kg, covers ~2 m², and protects against infection, UV, and dehydration."},
]

KNOWLEDGE_BASE["physiology"] = [
    {"q": "How does breathing work?", "a": "The diaphragm contracts, expanding the chest cavity, creating negative pressure that draws air in. Exhalation is passive relaxation."},
    {"q": "What is blood pressure?", "a": "Blood pressure is the force of blood against artery walls. Normal: ~120/80 mmHg. Systolic (beating) / Diastolic (resting)."},
    {"q": "How does the immune system work?", "a": "Innate immunity (barriers, inflammation) provides immediate defense. Adaptive immunity (B cells, T cells) targets specific pathogens and creates memory."},
]

KNOWLEDGE_BASE["nutrition"] = [
    {"q": "What are macronutrients?", "a": "Carbohydrates (4 kcal/g), proteins (4 kcal/g), and fats (9 kcal/g). They provide energy and structural building blocks."},
    {"q": "What vitamins are essential?", "a": "Fat-soluble: A, D, E, K. Water-soluble: B-complex (B1-B12) and C. The body cannot synthesize most vitamins."},
    {"q": "How much water should I drink daily?", "a": "About 2-3 liters (8-12 cups) for adults, more with exercise or hot weather. Needs vary by body size, activity level, and climate."},
]

# --- Social Sciences ---
KNOWLEDGE_BASE["economics"] = [
    {"q": "What is supply and demand?", "a": "Supply and demand determine market prices. When demand exceeds supply, prices rise. When supply exceeds demand, prices fall. Equilibrium is where they intersect."},
    {"q": "What is GDP?", "a": "Gross Domestic Product is the total value of goods and services produced in a country. GDP = C + I + G + (X - M) where C=consumption, I=investment, G=government, X-M=net exports."},
    {"q": "What is inflation?", "a": "Inflation is the rate at which prices rise, reducing purchasing power. Measured by CPI (Consumer Price Index). Moderate inflation (2-3%) is normal."},
    {"q": "What is compound interest?", "a": "Interest calculated on both principal and accumulated interest. A = P(1 + r/n)^(nt). This is why saving early is powerful."},
]

KNOWLEDGE_BASE["psychology"] = [
    {"q": "What is cognitive bias?", "a": "Cognitive biases are systematic errors in thinking. Examples: confirmation bias (seeking confirming evidence), anchoring (over-relying on first information), availability heuristic."},
    {"q": "What is Maslow's hierarchy of needs?", "a": "A pyramid: physiological (food, water), safety, love/belonging, esteem, self-actualization. Basic needs must be met before higher ones."},
    {"q": "What is classical conditioning?", "a": "Learning by association. Pavlov's dogs: a bell (neutral stimulus) paired with food (unconditioned stimulus) eventually triggers salivation alone."},
    {"q": "What is the Stanford prison experiment?", "a": "A 1971 study where students assigned as 'guards' quickly became abusive toward 'prisoners'. It demonstrated how roles and authority affect behavior."},
]

KNOWLEDGE_BASE["philosophy"] = [
    {"q": "What is the Socratic method?", "a": "A form of cooperative argumentative dialogue through questioning. Socrates asked probing questions to expose contradictions and deepen understanding."},
    {"q": "What is utilitarianism?", "a": "An ethical theory that the best action maximizes overall happiness. 'The greatest good for the greatest number.' Proposed by Bentham and Mill."},
    {"q": "What is the trolley problem?", "a": "A thought experiment: do you redirect a trolley to kill 1 person instead of 5? It explores the tension between consequentialism and deontological ethics."},
    {"q": "What is existentialism?", "a": "A philosophy emphasizing individual existence, freedom, and choice. 'Existence precedes essence' (Sartre) — we define ourselves through actions, not predetermined nature."},
]

# --- Humanities ---
KNOWLEDGE_BASE["history"] = [
    {"q": "When was the Renaissance?", "a": "The Renaissance was a cultural movement from roughly 1300-1600, beginning in Italy. It emphasized humanism, art, science, and a revival of classical learning."},
    {"q": "What caused World War I?", "a": "WWI (1914-1918) was caused by: militarism, alliances (Triple Entente vs Triple Alliance), imperialism, nationalism, and the assassination of Archduke Franz Ferdinand."},
    {"q": "What was the Industrial Revolution?", "a": "The transition from manual labor to machine manufacturing, beginning in Britain ~1760. Key inventions: steam engine, spinning jenny, power loom. Transformed society and economy."},
    {"q": "What is the Silk Road?", "a": "A network of trade routes connecting East and West from 130 BCE to 1453 CE. Facilitated trade of silk, spices, ideas, and religions between Asia, Europe, and Africa."},
]

KNOWLEDGE_BASE["literature"] = [
    {"q": "What is a metaphor?", "a": "A figure of speech that directly compares two unlike things. 'Time is money' equates time with value. Unlike similes ('like' or 'as'), metaphors state identity."},
    {"q": "Who wrote Hamlet?", "a": "William Shakespeare wrote Hamlet around 1600. It's a tragedy about Prince Hamlet seeking revenge for his father's murder."},
    {"q": "What is a protagonist?", "a": "The main character in a story. They drive the plot and often face conflict with an antagonist (opposing character or force)."},
    {"q": "What is irony?", "a": "When the outcome contradicts expectations. Verbal irony: saying the opposite of what you mean. Situational irony: unexpected outcome. Dramatic irony: audience knows something characters don't."},
]

KNOWLEDGE_BASE["linguistics"] = [
    {"q": "What is syntax?", "a": "Syntax is the study of sentence structure and word order. In English, basic word order is SVO (Subject-Verb-Object): 'The cat sat on the mat.'"},
    {"q": "How many languages exist?", "a": "About 7,000 languages are spoken today. About 40% are endangered. The most spoken are English, Mandarin Chinese, Hindi, and Spanish."},
    {"q": "What is a phoneme?", "a": "A phoneme is the smallest unit of sound that distinguishes meaning. English has about 44 phonemes. /p/ and /b/ are different phonemes (pat vs bat)."},
]

# --- Arts ---
KNOWLEDGE_BASE["music"] = [
    {"q": "What is a chord?", "a": "A chord is three or more notes played together. Major chords (happy): root + major 3rd + perfect 5th. Minor chords (sad): root + minor 3rd + perfect 5th."},
    {"q": "What is the circle of fifths?", "a": "A diagram showing the 12 chromatic pitches arranged by perfect fifths. Moving clockwise adds sharps; counterclockwise adds flats. Shows key relationships."},
    {"q": "What is rhythm?", "a": "Rhythm is the pattern of sounds and silences in time. Measured in beats per minute (BPM). Common time signatures: 4/4 (most common), 3/4 (waltz), 6/8 (compound)."},
]

KNOWLEDGE_BASE["visual_arts"] = [
    {"q": "What is color theory?", "a": "Color theory explains how colors mix and interact. Primary colors (red, blue, yellow) mix to create secondary and tertiary colors. Complementary colors (opposite on wheel) create contrast."},
    {"q": "What is the rule of thirds?", "a": "A composition guideline: divide the frame into 3×3 grid, place key elements along lines or intersections. Creates more dynamic, interesting compositions."},
    {"q": "What was Impressionism?", "a": "An 1870s-1880s art movement emphasizing light, color, and visible brushstrokes. Monet, Renoir, Degas captured fleeting moments and atmospheric effects."},
]

KNOWLEDGE_BASE["design"] = [
    {"q": "What is UX design?", "a": "User Experience design focuses on how products feel to use. It involves research, wireframing, prototyping, and testing to create intuitive, enjoyable interfaces."},
    {"q": "What is typography?", "a": "Typography is the art of arranging type. Key concepts: font, hierarchy, spacing (leading, kerning, tracking), readability, and pairing typefaces."},
    {"q": "What are Gestalt principles?", "a": "Principles of visual perception: proximity (nearby elements grouped), similarity (similar elements grouped), closure (brain completes shapes), continuity (eye follows paths)."},
]

# --- Business ---
KNOWLEDGE_BASE["finance"] = [
    {"q": "What is net present value (NPV)?", "a": "NPV = Σ(CF_t / (1+r)^t). It sums discounted future cash flows. Positive NPV means the investment creates value."},
    {"q": "What is a stock?", "a": "A stock represents ownership in a company. Shareholders may receive dividends and benefit from price appreciation. Risk varies by company and market conditions."},
    {"q": "What is diversification?", "a": "Spreading investments across assets to reduce risk. 'Don't put all your eggs in one basket.' Different assets react differently to market events."},
    {"q": "What is P/E ratio?", "a": "Price-to-Earnings ratio = Stock Price / Earnings Per Share. A high P/E may indicate growth expectations; low P/E may indicate undervaluation or problems."},
]

KNOWLEDGE_BASE["business_management"] = [
    {"q": "What is the SWOT analysis?", "a": "Strategic planning tool examining: Strengths (internal advantages), Weaknesses (internal limitations), Opportunities (external chances), Threats (external risks)."},
    {"q": "What is the Pareto principle?", "a": "The 80/20 rule: 80% of effects come from 20% of causes. 80% of sales come from 20% of customers. Focus on the vital few."},
    {"q": "What is agile methodology?", "a": "An iterative approach to project management. Work in sprints (1-4 weeks), deliver incrementally, adapt to change. Values individuals, working software, collaboration."},
]

KNOWLEDGE_BASE["marketing"] = [
    {"q": "What is SEO?", "a": "Search Engine Optimization improves website visibility in search results. Key factors: keywords, content quality, backlinks, site speed, mobile-friendliness."},
    {"q": "What is a marketing funnel?", "a": "The customer journey: Awareness → Interest → Consideration → Intent → Purchase → Loyalty. Different strategies for each stage."},
    {"q": "What is content marketing?", "a": "Creating valuable, relevant content to attract and engage an audience. Builds trust and authority. Examples: blog posts, videos, podcasts, newsletters."},
]

# --- Practical Skills ---
KNOWLEDGE_BASE["cooking"] = [
    {"q": "How do I make scrambled eggs?", "a": "Beat 2-3 eggs with a pinch of salt. Heat butter in a pan over medium-low heat. Pour in eggs, gently stir with a spatula as they set. Remove from heat while still slightly moist."},
    {"q": "What is the Maillard reaction?", "a": "A chemical reaction between amino acids and reducing sugars at 140-165°C. Creates the brown color and complex flavors in seared meat, toasted bread, and coffee."},
    {"q": "What is mise en place?", "a": "French for 'everything in its place.' Preparing and organizing all ingredients before cooking. Essential for efficient, stress-free cooking."},
]

KNOWLEDGE_BASE["first_aid"] = [
    {"q": "How do I perform CPR?", "a": "1) Check responsiveness. 2) Call 911. 3) Place heel of hand on center of chest. 4) Push hard and fast (2 inches deep, 100-120/min). 5) Give 2 rescue breaths every 30 compressions."},
    {"q": "How do I treat a burn?", "a": "Cool under running water for 10-20 minutes. Do not use ice. Cover loosely with sterile dressing. Do not pop blisters. Seek medical help for severe burns."},
    {"q": "What is the Heimlich maneuver?", "a": "For choking: stand behind the person, place fist above navel, pull inward and upward with quick thrusts. For infants: 5 back blows, 5 chest thrusts."},
]

KNOWLEDGE_BASE["sustainability"] = [
    {"q": "How do I reduce my carbon footprint?", "a": "Key actions: eat less meat, use public transport, reduce energy consumption, buy local products, reduce waste, use renewable energy, plant trees."},
    {"q": "What is renewable energy?", "a": "Energy from sources that replenish naturally: solar, wind, hydroelectric, geothermal, tidal. Unlike fossil fuels, they don't deplete and produce minimal emissions."},
    {"q": "What is the circular economy?", "a": "An economic system aimed at minimizing waste. Products are designed for durability, reuse, and recycling. Resources are kept in use as long as possible."},
]

# --- Tool use examples ---
KNOWLEDGE_BASE["calculator_use"] = [
    {"q": "Calculate 15 * 7.", "a": "I'll use the calculator for this.\n<tool:calculator>\nexpression=15 * 7\n</tool>\nThe result is 105."},
    {"q": "What is 256 / 8?", "a": "Let me compute that.\n<tool:calculator>\nexpression=256 / 8\n</tool>\n256 divided by 8 equals 32."},
    {"q": "Compute the square root of 144.", "a": "I'll calculate that.\n<tool:calculator>\nexpression=144 ** 0.5\n</tool>\nThe square root of 144 is 12."},
    {"q": "What is 3.14159 * 7^2?", "a": "Let me compute that.\n<tool:calculator>\nexpression=3.14159 * 7 ** 2\n</tool>\n3.14159 × 49 ≈ 153.94 (approximately the area of a circle with radius 7)."},
]


def _make_instruction_response_pairs() -> list[dict]:
    """Generate instruction-response pairs from the knowledge base."""
    pairs = []
    for field, items in KNOWLEDGE_BASE.items():
        for item in items:
            pairs.append({
                "id": f"sw-{field}-{hashlib.md5(item['q'].encode()).hexdigest()[:8]}",
                "field": field,
                "instruction": item["q"],
                "response": item["a"],
                "type": "instruction_response",
            })
    return pairs


def _make_reasoning_pairs() -> list[dict]:
    """Generate chain-of-thought reasoning examples."""
    reasoning = [
        {
            "instruction": "What is 15% of 230? Show your work.",
            "response": (
                "Step 1: Convert percentage to decimal: 15% = 0.15\n"
                "Step 2: Multiply: 0.15 × 230 = 34.5\n"
                "Answer: 34.5"
            ),
        },
        {
            "instruction": "If a car travels at 60 mph for 2.5 hours, how far does it go?",
            "response": (
                "Step 1: Recall the formula: distance = speed × time\n"
                "Step 2: Substitute values: distance = 60 × 2.5\n"
                "Step 3: Calculate: 60 × 2.5 = 150\n"
                "Answer: 150 miles"
            ),
        },
        {
            "instruction": "A shirt costs $40 and is on sale for 25% off. What is the sale price?",
            "response": (
                "Step 1: Calculate discount: 25% of $40 = 0.25 × 40 = $10\n"
                "Step 2: Subtract discount: $40 - $10 = $30\n"
                "Answer: $30"
            ),
        },
        {
            "instruction": "What is the derivative of sin(x) * e^x?",
            "response": (
                "Step 1: Use the product rule: d/dx[f*g] = f'g + fg'\n"
                "Step 2: f = sin(x), g = e^x\n"
                "Step 3: f' = cos(x), g' = e^x\n"
                "Step 4: f'g + fg' = cos(x)*e^x + sin(x)*e^x\n"
                "Step 5: Factor: e^x(cos(x) + sin(x))\n"
                "Answer: e^x(cos(x) + sin(x))"
            ),
        },
        {
            "instruction": "Explain why the sky is blue.",
            "response": (
                "Step 1: Sunlight contains all colors of the visible spectrum.\n"
                "Step 2: When sunlight enters Earth's atmosphere, it collides with gas molecules.\n"
                "Step 3: Shorter wavelengths (blue/violet) scatter more than longer wavelengths (red/orange) — this is Rayleigh scattering.\n"
                "Step 4: Blue light scatters in all directions, so we see blue from every direction.\n"
                "Answer: The sky appears blue because shorter blue wavelengths scatter more than other colors when sunlight interacts with air molecules."
            ),
        },
        {
            "instruction": "How much interest do I earn on $1000 at 5% annual interest for 3 years, compounded annually?",
            "response": (
                "Step 1: Formula: A = P(1 + r)^t\n"
                "Step 2: P = 1000, r = 0.05, t = 3\n"
                "Step 3: A = 1000(1.05)^3 = 1000 × 1.157625 = 1157.63\n"
                "Step 4: Interest = A - P = 1157.63 - 1000 = 157.63\n"
                "Answer: $157.63"
            ),
        },
        {
            "instruction": "If I flip a fair coin 3 times, what is the probability of getting exactly 2 heads?",
            "response": (
                "Step 1: Total outcomes = 2^3 = 8\n"
                "Step 2: Favorable outcomes (exactly 2 heads): HHT, HTH, THH = 3\n"
                "Step 3: Probability = 3/8 = 0.375\n"
                "Answer: 3/8 or 37.5%"
            ),
        },
        {
            "instruction": "A triangle has sides 3, 4, and 5. Is it a right triangle?",
            "response": (
                "Step 1: Check Pythagorean theorem: a² + b² = c²\n"
                "Step 2: 3² + 4² = 9 + 16 = 25\n"
                "Step 3: 5² = 25\n"
                "Step 4: Since 3² + 4² = 5², the triangle satisfies the Pythagorean theorem.\n"
                "Answer: Yes, it is a right triangle."
            ),
        },
        {
            "instruction": "What is the boiling point of water at high altitude (e.g., 3000m)?",
            "response": (
                "Step 1: At sea level, water boils at 100°C (212°F) at 1 atmosphere.\n"
                "Step 2: As altitude increases, atmospheric pressure decreases.\n"
                "Step 3: Lower pressure means water boils at a lower temperature.\n"
                "Step 4: At 3000m, pressure is ~0.7 atm, so water boils at approximately 90°C (194°F).\n"
                "Answer: At 3000m altitude, water boils at approximately 90°C (194°F)."
            ),
        },
        {
            "instruction": "Explain the difference between mitosis and meiosis.",
            "response": (
                "Step 1: Both are cell division processes.\n"
                "Step 2: Mitosis produces 2 identical daughter cells (diploid, 2n). Used for growth and repair.\n"
                "Step 3: Meiosis produces 4 genetically different cells (haploid, n). Used for reproduction.\n"
                "Step 4: Mitosis has 1 division; meiosis has 2 divisions.\n"
                "Step 5: Meiosis introduces genetic variation through crossing over and independent assortment.\n"
                "Answer: Mitosis creates identical body cells; meiosis creates unique reproductive cells with half the chromosomes."
            ),
        },
    ]
    return [
        {
            "id": f"sw-reasoning-{i}",
            "field": "reasoning",
            "instruction": r["instruction"],
            "response": r["response"],
            "type": "chain_of_thought",
        }
        for i, r in enumerate(reasoning)
    ]


def _make_conversational_pairs() -> list[dict]:
    """Generate natural conversational dialogue examples."""
    conversations = [
        {
            "instruction": "Hello! How are you?",
            "response": "Hello! I'm doing well, thank you for asking. I'm here and ready to help you with whatever you need — whether it's answering questions, solving problems, or just having a conversation. What can I help you with today?",
        },
        {
            "instruction": "Can you help me with something?",
            "response": "Of course! I'd be happy to help. I can assist with math, science, programming, writing, research, analysis, and much more. Just tell me what you need and I'll do my best to help.",
        },
        {
            "instruction": "What can you do?",
            "response": "I can help with a wide range of tasks: answer questions across all fields of knowledge, solve math and science problems, write and edit text, analyze data, explain complex concepts, help with coding, provide creative writing assistance, and much more. I aim to give clear, accurate, and helpful responses. What would you like to explore?",
        },
        {
            "instruction": "Tell me something interesting.",
            "response": "Here's something fascinating: honey never spoils. Archaeologists have found 3,000-year-old honey in Egyptian tombs that was still perfectly edible. Honey's low moisture content, acidic pH, and natural hydrogen peroxide make it an inhospitable environment for bacteria. It's essentially eternal food!",
        },
        {
            "instruction": "I'm feeling stressed about work.",
            "response": "I understand that work stress can be overwhelming. Here are some strategies that might help: take short breaks every hour, prioritize tasks using the Eisenhower matrix (urgent/important), practice deep breathing, break large tasks into smaller steps, and remember to maintain work-life boundaries. Would you like to talk about specific challenges you're facing?",
        },
        {
            "instruction": "What's the meaning of life?",
            "response": "That's one of humanity's oldest questions! Different perspectives offer different answers: existentialists say we create our own meaning through choices and actions; religious traditions often point to serving a higher purpose; humanists focus on contributing to human flourishing and finding fulfillment in relationships and growth. Perhaps the meaning of life is the meaning you choose to give it.",
        },
        {
            "instruction": "Explain quantum computing to a 5-year-old.",
            "response": "Imagine you have a magical coin. A normal coin is either heads or tails. But a magical quantum coin can be BOTH heads AND tails at the same time! Quantum computers use these magical coins to try lots of answers at once, instead of one by one like regular computers. That makes them super fast at certain tricky problems!",
        },
        {
            "instruction": "Write me a poem about the ocean.",
            "response": "Beneath the sky of endless blue,\nThe ocean breathes, forever new.\nWaves that crash and退 and flow,\nWhisper secrets all must know.\n\nDepths of mystery, dark and deep,\nWhere ancient creatures softly creep.\nSalt and sun and tides that sing,\nThe ocean is a wondrous thing.",
        },
        {
            "instruction": "How do I start learning to code?",
            "response": "Great question! Here's a practical path:\n\n1. **Choose a language**: Python is beginner-friendly and versatile.\n2. **Learn basics**: variables, loops, functions, conditionals.\n3. **Practice daily**: 30 minutes a day beats 5 hours once a week.\n4. **Build projects**: Start small (calculator, todo app), then scale up.\n5. **Join communities**: Stack Overflow, Reddit, Discord.\n6. **Don't give up**: Everyone struggles at first — that's normal.\n\nWant me to help you write your first program?",
        },
        {
            "instruction": "What's the difference between machine learning and AI?",
            "response": "AI (Artificial Intelligence) is the broad goal of making machines智能. Machine Learning is a subset of AI where systems learn from data instead of being explicitly programmed.\n\nThink of it this way: AI is the destination, ML is one of the roads to get there. Other paths to AI include rule-based systems, expert systems, and symbolic reasoning.\n\nML includes techniques like neural networks, decision trees, and reinforcement learning. Deep learning is a subset of ML using deep neural networks.",
        },
        {
            "instruction": "Can you explain climate change simply?",
            "response": "Sure! Think of Earth's atmosphere like a blanket. The sun warms the Earth, and some heat escapes back to space. Greenhouse gases (like CO₂) trap some heat — this is natural and keeps Earth warm enough for life.\n\nThe problem: burning fossil fuels releases extra CO₂, making the blanket thicker. More heat gets trapped, Earth warms up. This causes: melting ice, rising sea levels, more extreme weather, and disrupted ecosystems. It's like turning up a thermostat that we can't easily turn back down.",
        },
    ]
    return [
        {
            "id": f"sw-convo-{i}",
            "field": "conversation",
            "instruction": c["instruction"],
            "response": c["response"],
            "type": "conversational",
        }
        for i, c in enumerate(conversations)
    ]


def _make_tool_use_pairs() -> list[dict]:
    """Generate tool-use training examples."""
    examples = []
    idx = 0

    for field in TOOL_USE:
        for ex in field.tool_use_examples:
            examples.append({
                "id": f"sw-tool-{idx:04d}",
                "field": field.name,
                "instruction": f"Use the {ex['tool']} tool to: {ex['args']}",
                "response": (
                    f"<tool:{ex['tool']}>\n"
                    + "\n".join(f"{k}={v}" for k, v in ex["args"].items())
                    + f"\n</tool>\n"
                    f"Result: {ex['result']}"
                ),
                "tool_name": ex["tool"],
                "tool_args": ex["args"],
                "tool_result": ex["result"],
                "type": "tool_use",
            })
            idx += 1

    return examples


def _make_code_pairs() -> list[dict]:
    """Generate code generation examples."""
    examples = [
        {
            "instruction": "Write a Python function to find the factorial of a number.",
            "response": (
                "def factorial(n):\n"
                "    \"\"\"Calculate the factorial of n recursively.\"\"\"\n"
                "    if n < 0:\n"
                "        raise ValueError('Factorial not defined for negative numbers')\n"
                "    if n <= 1:\n"
                "        return 1\n"
                "    return n * factorial(n - 1)\n\n"
                "# Iterative version:\n"
                "def factorial_iterative(n):\n"
                "    result = 1\n"
                "    for i in range(2, n + 1):\n"
                "        result *= i\n"
                "    return result"
            ),
        },
        {
            "instruction": "Write a Python function to check if a string is a palindrome.",
            "response": (
                "def is_palindrome(s):\n"
                "    \"\"\"Check if a string is a palindrome (ignoring case and spaces).\"\"\"\n"
                "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
                "    return cleaned == cleaned[::-1]"
            ),
        },
        {
            "instruction": "Write a Python function to find the largest element in a list.",
            "response": (
                "def find_largest(lst):\n"
                "    \"\"\"Find the largest element in a list.\"\"\"\n"
                "    if not lst:\n"
                "        raise ValueError('List is empty')\n"
                "    largest = lst[0]\n"
                "    for item in lst[1:]:\n"
                "        if item > largest:\n"
                "            largest = item\n"
                "    return largest\n\n"
                "# Or simply: return max(lst)"
            ),
        },
        {
            "instruction": "Write a Python function to count words in a text.",
            "response": (
                "def count_words(text):\n"
                "    \"\"\"Count the number of words in a text string.\"\"\"\n"
                "    words = text.split()\n"
                "    return len(words)\n\n"
                "# More detailed version:\n"
                "def count_words_detailed(text):\n"
                "    words = text.split()\n"
                "    word_count = {}\n"
                "    for word in words:\n"
                "        word = word.lower().strip('.,!?;:')\n"
                "        word_count[word] = word_count.get(word, 0) + 1\n"
                "    return word_count"
            ),
        },
        {
            "instruction": "Write a Python function to sort a list using bubble sort.",
            "response": (
                "def bubble_sort(arr):\n"
                "    \"\"\"Sort a list using the bubble sort algorithm.\"\"\"\n"
                "    n = len(arr)\n"
                "    for i in range(n):\n"
                "        swapped = False\n"
                "        for j in range(0, n - i - 1):\n"
                "            if arr[j] > arr[j + 1]:\n"
                "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
                "                swapped = True\n"
                "        if not swapped:\n"
                "            break  # Already sorted\n"
                "    return arr"
            ),
        },
    ]
    return [
        {
            "id": f"sw-code-{i}",
            "field": "programming",
            "instruction": e["instruction"],
            "response": e["response"],
            "type": "code_generation",
        }
        for i, e in enumerate(examples)
    ]


def generate_all_dataset(output_dir: str) -> dict:
    """Generate the complete training dataset."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_data = []

    # 1. Knowledge base instruction-response pairs
    kb_pairs = _make_instruction_response_pairs()
    all_data.extend(kb_pairs)
    print(f"  Knowledge base: {len(kb_pairs)} examples")

    # 2. Reasoning / chain-of-thought
    reasoning_pairs = _make_reasoning_pairs()
    all_data.extend(reasoning_pairs)
    print(f"  Reasoning: {len(reasoning_pairs)} examples")

    # 3. Conversational
    convo_pairs = _make_conversational_pairs()
    all_data.extend(convo_pairs)
    print(f"  Conversational: {len(convo_pairs)} examples")

    # 4. Tool use
    tool_pairs = _make_tool_use_pairs()
    all_data.extend(tool_pairs)
    print(f"  Tool use: {len(tool_pairs)} examples")

    # 5. Code generation
    code_pairs = _make_code_pairs()
    all_data.extend(code_pairs)
    print(f"  Code generation: {len(code_pairs)} examples")

    # Write combined dataset
    output_path = out / "silverwing-comprehensive-v1.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Write manifest
    manifest = {
        "version": "silverwing-comprehensive-v1",
        "total_examples": len(all_data),
        "categories": {
            "knowledge_base": len(kb_pairs),
            "reasoning": len(reasoning_pairs),
            "conversational": len(convo_pairs),
            "tool_use": len(tool_pairs),
            "code_generation": len(code_pairs),
        },
        "fields_covered": sorted({
            item.get("field", "") for item in all_data
        }),
        "output_file": str(output_path),
    }
    manifest_path = out / "silverwing-comprehensive-v1.manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n  Total: {len(all_data)} examples")
    print(f"  Output: {output_path}")
    print(f"  Manifest: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive SilverWing training data"
    )
    parser.add_argument(
        "--output", default="datasets/processed",
        help="Output directory (default: datasets/processed)",
    )
    args = parser.parse_args()

    print("\n  SilverWing Comprehensive Dataset Generator")
    print("  ==========================================\n")

    manifest = generate_all_dataset(args.output)

    print(f"\n  Fields covered: {len(manifest['fields_covered'])}")
    print(f"  Categories: {list(manifest['categories'].keys())}")
    print("\n  Done!")


if __name__ == "__main__":
    main()
