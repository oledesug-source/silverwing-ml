"""SilverWing — Comprehensive knowledge field definitions.

Every field of human knowledge, modern and traditional, organized
into categories with subfields, key concepts, and example templates.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FieldDefinition:
    name: str
    category: str
    subfields: list[str]
    key_concepts: list[str]
    prompt_templates: list[str] = field(default_factory=list)
    tool_use_examples: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────
# CATEGORY 1: MATHEMATICS & FORMAL SCIENCES
# ─────────────────────────────────────────────────────

MATHEMATICS = [
    FieldDefinition(
        name="arithmetic",
        category="mathematics",
        subfields=["addition", "subtraction", "multiplication", "division",
                    "modulo", "exponents", "order_of_operations"],
        key_concepts=["integers", "decimals", "fractions", "percentages",
                      "ratios", "proportions"],
        prompt_templates=[
            "Calculate {a} {op} {b}.",
            "What is {a} percent of {b}?",
            "Simplify the fraction {n}/{d}.",
            "What is {a} {op} {b} {op2} {c}?",
        ],
        tool_use_examples=[
            {"tool": "calculator", "args": {"expression": "2 + 2"}, "result": "4"},
            {"tool": "calculator", "args": {"expression": "15 * 7"}, "result": "105"},
            {"tool": "calculator", "args": {"expression": "144 / 12"}, "result": "12"},
            {"tool": "calculator", "args": {"expression": "2 ** 10"}, "result": "1024"},
        ],
    ),
    FieldDefinition(
        name="algebra",
        category="mathematics",
        subfields=["linear_equations", "quadratic_equations", "polynomials",
                    "inequalities", "systems_of_equations", "functions",
                    "abstract_algebra"],
        key_concepts=["variables", "factoring", "graphing", "slope",
                      "intercepts", "matrices", "determinants"],
        prompt_templates=[
            "Solve for x: {equation}",
            "Factor the expression: {expression}",
            "Find the roots of {equation}.",
            "Simplify: {expression}",
        ],
    ),
    FieldDefinition(
        name="calculus",
        category="mathematics",
        subfields=["differential_calculus", "integral_calculus",
                    "multivariable_calculus", "vector_calculus",
                    "differential_equations", "analysis"],
        key_concepts=["derivatives", "integrals", "limits", "continuity",
                      "series", "sequences", "taylor_series", "chain_rule"],
        prompt_templates=[
            "Find the derivative of {function}.",
            "Evaluate the integral of {function} dx.",
            "Find the limit of {expression} as x approaches {value}.",
            "Solve the differential equation: {equation}",
        ],
    ),
    FieldDefinition(
        name="geometry",
        category="mathematics",
        subfields=["euclidean_geometry", "trigonometry", "analytic_geometry",
                    "non_euclidean", "topology", "fractal_geometry"],
        key_concepts=["angles", "triangles", "circles", "polyhedra",
                      "area", "volume", "perimeter", "similarity"],
        prompt_templates=[
            "Find the area of a {shape} with {params}.",
            "Calculate the {measurement} of a circle with radius {r}.",
            "What is the {measurement} of a triangle with sides {a}, {b}, {c}?",
        ],
    ),
    FieldDefinition(
        name="statistics",
        category="mathematics",
        subfields=["descriptive_statistics", "inferential_statistics",
                    "probability", "bayesian_statistics", "regression",
                    "hypothesis_testing"],
        key_concepts=["mean", "median", "mode", "standard_deviation",
                      "variance", "distributions", "confidence_intervals",
                      "p_values", "correlation"],
        prompt_templates=[
            "Calculate the mean, median, and mode of: {data}",
            "What is the standard deviation of: {data}",
            "Given a {distribution} distribution with mean {mu} and std {sigma}, what is P(X < {x})?",
        ],
    ),
    FieldDefinition(
        name="discrete_mathematics",
        category="mathematics",
        subfields=["combinatorics", "graph_theory", "number_theory",
                    "logic", "set_theory", "combinatorial_optimization"],
        key_concepts=["permutations", "combinations", "factorial",
                      "prime_numbers", "modular_arithmetic", "graphs",
                      "trees", "boolean_logic"],
        prompt_templates=[
            "How many ways can you choose {k} items from {n}?",
            "Is {n} a prime number?",
            "What is {n}! (factorial)?",
            "Find the GCD of {a} and {b}.",
        ],
    ),
]

# ─────────────────────────────────────────────────────
# CATEGORY 2: PHYSICS & ENGINEERING
# ─────────────────────────────────────────────────────

PHYSICS_ENGINEERING = [
    FieldDefinition(
        name="classical_mechanics",
        category="physics",
        subfields=["kinematics", "dynamics", "statics", "fluid_mechanics",
                    "oscillations", "waves"],
        key_concepts=["force", "mass", "acceleration", "velocity",
                      "momentum", "energy", "torque", "friction",
                      "newtons_laws", "conservation"],
        prompt_templates=[
            "A {object} of mass {m} kg is {situation}. Find {target}.",
            "Calculate the {quantity} of a {object} with {params}.",
            "Using Newton's second law, find the {quantity}.",
        ],
    ),
    FieldDefinition(
        name="electromagnetism",
        category="physics",
        subfields=["electrostatics", "circuits", "magnetism",
                    "electromagnetic_waves", "maxwells_equations"],
        key_concepts=["charge", "voltage", "current", "resistance",
                      "capacitance", "inductance", "magnetic_fields",
                      "ohms_law"],
        prompt_templates=[
            "Calculate the {quantity} in a circuit with {params}.",
            "Using Ohm's law, find {target} given {params}.",
            "What is the {quantity} of a capacitor with {params}?",
        ],
    ),
    FieldDefinition(
        name="thermodynamics",
        category="physics",
        subfields=["zeroth_law", "first_law", "second_law", "third_law",
                    "heat_transfer", "statistical_mechanics"],
        key_concepts=["temperature", "entropy", "enthalpy", "heat",
                      "work", "carnot_cycle", "ideal_gas_law",
                      "phase_transitions"],
        prompt_templates=[
            "Calculate the {quantity} for a {system}.",
            "What is the entropy change when {process}?",
            "Using the ideal gas law, find {target}.",
        ],
    ),
    FieldDefinition(
        name="quantum_mechanics",
        category="physics",
        subfields=["wave_mechanics", "matrix_mechanics", "quantum_field_theory",
                    "quantum_computing", "quantum_information"],
        key_concepts=["wave_function", "superposition", "entanglement",
                      "uncertainty_principle", "schrodinger_equation",
                      "quantum_states", "observables"],
        prompt_templates=[
            "Describe the quantum state of {system}.",
            "Calculate the expectation value of {observable}.",
            "Explain the {concept} in quantum mechanics.",
        ],
    ),
    FieldDefinition(
        name="electrical_engineering",
        category="engineering",
        subfields=["circuit_design", "digital_logic", "signal_processing",
                    "power_systems", "control_systems", "embedded_systems"],
        key_concepts=["resistors", "capacitors", "transistors", "op_amps",
                      "filters", "amplifiers", "microcontrollers",
                      "pcb_design"],
        prompt_templates=[
            "Design a circuit that {requirement}.",
            "Calculate the {parameter} of a {circuit_type} circuit.",
            "What is the transfer function of {system}?",
        ],
    ),
    FieldDefinition(
        name="mechanical_engineering",
        category="engineering",
        subfields=["thermodynamics", "fluid_mechanics", "materials_science",
                    "machine_design", "manufacturing", "robotics"],
        key_concepts=["stress", "strain", "fatigue", "heat_transfer",
                      "fluid_flow", "gears", "bearings", "cnc"],
        prompt_templates=[
            "Calculate the {quantity} of a {component}.",
            "What is the {measurement} for material {material}?",
            "Design a {component} that meets {requirements}.",
        ],
    ),
    FieldDefinition(
        name="civil_engineering",
        category="engineering",
        subfields=["structural_engineering", "geotechnical", "transportation",
                    "hydraulics", "construction_management", "surveying"],
        key_concepts=["load_bearing", "concrete", "steel", "foundations",
                      "soil_mechanics", "bridges", "buildings"],
        prompt_templates=[
            "Calculate the load capacity of {structure}.",
            "What is the {parameter} for {material}?",
            "Design a {structure} for {conditions}.",
        ],
    ),
    FieldDefinition(
        name="aerospace_engineering",
        category="engineering",
        subfields=["aerodynamics", "propulsion", "structures",
                    "avionics", "orbital_mechanics", "spacecraft_design"],
        key_concepts=["lift", "drag", "thrust", "weight", "airfoil",
                      "rocket_equation", "orbit", "delta_v"],
        prompt_templates=[
            "Calculate the {quantity} for a {vehicle}.",
            "What is the {measurement} of {object}?",
            "Design a {system} for {mission}.",
        ],
    ),
]

# ─────────────────────────────────────────────────────
# CATEGORY 3: COMPUTER SCIENCE & AI
# ─────────────────────────────────────────────────────

COMPUTER_SCIENCE = [
    FieldDefinition(
        name="programming",
        category="computer_science",
        subfields=["python", "javascript", "rust", "c", "java", "go",
                    "typescript", "sql", "html_css"],
        key_concepts=["data_structures", "algorithms", "oop", "functional",
                      "async", "patterns", "debugging", "testing"],
        prompt_templates=[
            "Write a {language} function that {description}.",
            "How do I {task} in {language}?",
            "Fix the bug in this code: {code}",
            "Optimize this {language} code for {goal}.",
        ],
    ),
    FieldDefinition(
        name="data_structures",
        category="computer_science",
        subfields=["arrays", "linked_lists", "trees", "graphs", "hash_tables",
                    "stacks", "queues", "heaps", "tries"],
        key_concepts=["time_complexity", "space_complexity", "traversal",
                      "sorting", "searching", "balanced_trees"],
        prompt_templates=[
            "Implement a {data_structure} in {language}.",
            "What is the time complexity of {operation} on a {data_structure}?",
            "Explain how {data_structure} works with an example.",
        ],
    ),
    FieldDefinition(
        name="algorithms",
        category="computer_science",
        subfields=["sorting", "searching", "dynamic_programming",
                    "greedy", "graph_algorithms", "string_algorithms",
                    "numerical_algorithms", "cryptography"],
        key_concepts=["big_o", "recursion", "memoization", "backtracking",
                      "divide_and_conquer", "bfs", "dfs", "dijkstra",
                      "binary_search"],
        prompt_templates=[
            "Implement {algorithm} in {language}.",
            "What is the time complexity of {algorithm}?",
            "Solve this problem using {approach}: {problem}",
            "Explain how {algorithm} works step by step.",
        ],
    ),
    FieldDefinition(
        name="machine_learning",
        category="computer_science",
        subfields=["supervised_learning", "unsupervised_learning",
                    "reinforcement_learning", "deep_learning",
                    "transformers", "generative_ai", "nlp",
                    "computer_vision"],
        key_concepts=["regression", "classification", "clustering",
                      "neural_networks", "backpropagation", "gradient_descent",
                      "regularization", "overfitting", "cross_validation",
                      "attention_mechanism", "transformer_architecture"],
        prompt_templates=[
            "Explain how {algorithm} works.",
            "Train a {model_type} to {task}.",
            "What are the tradeoffs of {approach} vs {approach2}?",
            "Design a neural network for {task}.",
        ],
    ),
    FieldDefinition(
        name="web_development",
        category="computer_science",
        subfields=["frontend", "backend", "fullstack", "devops",
                    "databases", "api_design", "security"],
        key_concepts=["html", "css", "javascript", "react", "node",
                      "databases", "rest_api", "graphql", "authentication"],
        prompt_templates=[
            "Build a {type} application using {tech}.",
            "How do I {feature} in {framework}?",
            "Design an API endpoint for {purpose}.",
            "Fix this {issue} in my {tech} code.",
        ],
    ),
    FieldDefinition(
        name="cybersecurity",
        category="computer_science",
        subfields=["network_security", "cryptography", "penetration_testing",
                    "forensics", "malware_analysis", "incident_response"],
        key_concepts=["encryption", "hashing", "authentication", "firewalls",
                      "vulnerabilities", "owasp_top_10", "zero_trust"],
        prompt_templates=[
            "Explain the {vulnerability} and how to prevent it.",
            "How does {algorithm} work in cryptography?",
            "What are best practices for {security_topic}?",
        ],
    ),
    FieldDefinition(
        name="systems_programming",
        category="computer_science",
        subfields=["operating_systems", "compilers", "databases_internals",
                    "networking", "distributed_systems", "embedded"],
        key_concepts=["memory_management", "scheduling", "virtual_memory",
                      "tcp_ip", "consensus", "raft", "paxos", "鎖"],
        prompt_templates=[
            "How does {system} work internally?",
            "Explain the {concept} in operating systems.",
            "Design a {system} that {requirements}.",
        ],
    ),
]

# ─────────────────────────────────────────────────────
# CATEGORY 4: NATURAL SCIENCES
# ─────────────────────────────────────────────────────

NATURAL_SCIENCES = [
    FieldDefinition(
        name="chemistry",
        category="natural_sciences",
        subfields=["organic", "inorganic", "physical_chemistry",
                    "biochemistry", "analytical", "nuclear"],
        key_concepts=["elements", "compounds", "reactions", "bonds",
                      "acids_bases", "solutions", "equilibrium",
                      "stoichiometry", "periodic_table"],
        prompt_templates=[
            "Balance the equation: {equation}",
            "What is the {property} of {substance}?",
            "Explain the {concept} in chemistry.",
            "Calculate the {quantity} for reaction: {reaction}",
        ],
    ),
    FieldDefinition(
        name="biology",
        category="natural_sciences",
        subfields=["cell_biology", "genetics", "evolution", "ecology",
                    "anatomy", "physiology", "microbiology", "botany",
                    "zoology", "marine_biology"],
        key_concepts=["cells", "dna", "rna", "proteins", "evolution",
                      "natural_selection", "ecosystems", "photosynthesis",
                      "mitosis", "meiosis"],
        prompt_templates=[
            "Explain the process of {biological_process}.",
            "What is the function of {organ/structure}?",
            "Describe the {concept} in biology.",
        ],
    ),
    FieldDefinition(
        name="earth_sciences",
        category="natural_sciences",
        subfields=["geology", "meteorology", "oceanography",
                    "paleontology", "geography", "climatology"],
        key_concepts=["plate_tectonics", "minerals", "weather",
                      "climate", "erosion", "fossils", "ocean_currents"],
        prompt_templates=[
            "Explain the geological process of {process}.",
            "What causes {phenomenon}?",
            "Describe the {concept} in earth sciences.",
        ],
    ),
    FieldDefinition(
        name="astronomy",
        category="natural_sciences",
        subfields=["stellar_astronomy", "planetary_science",
                    "cosmology", "astrophysics", "radio_astronomy"],
        key_concepts=["stars", "planets", "galaxies", "black_holes",
                      "dark_matter", "dark_energy", "big_bang",
                      "light_year", "redshift"],
        prompt_templates=[
            "Explain the {celestial_phenomenon}.",
            "What is the lifecycle of a {star_type}?",
            "Describe the {concept} in astronomy.",
        ],
    ),
]

# ─────────────────────────────────────────────────────
# CATEGORY 5: MEDICINE & HEALTH
# ─────────────────────────────────────────────────────

MEDICINE_HEALTH = [
    FieldDefinition(
        name="anatomy",
        category="medicine",
        subfields=["musculoskeletal", "cardiovascular", "nervous_system",
                    "respiratory", "digestive", "endocrine", "immune"],
        key_concepts=["organs", "tissues", "bones", "muscles",
                      "blood_vessels", "nerves", "cells"],
        prompt_templates=[
            "Describe the anatomy of {organ/system}.",
            "What is the function of {structure}?",
            "Explain the {process} in the {system}.",
        ],
    ),
    FieldDefinition(
        name="physiology",
        category="medicine",
        subfields=["cell_physiology", "neurophysiology", "cardiovascular",
                    "respiratory", "renal", "endocrine"],
        key_concepts=["homeostasis", "metabolism", "action_potential",
                      "blood_pressure", "respiration", "digestion"],
        prompt_templates=[
            "Explain the physiology of {process}.",
            "How does the {system} regulate {parameter}?",
            "What happens during {physiological_event}?",
        ],
    ),
    FieldDefinition(
        name="pathology",
        category="medicine",
        subfields=["general_pathology", "oncology", "cardiovascular_pathology",
                    "neuropathology", "infectious_disease"],
        key_concepts=["inflammation", "infection", "cancer",
                      "autoimmune", "degenerative_diseases"],
        prompt_templates=[
            "Explain the pathophysiology of {disease}.",
            "What are the symptoms and treatment of {condition}?",
            "How does {disease} affect the {system}?",
        ],
    ),
    FieldDefinition(
        name="pharmacology",
        category="medicine",
        subfields=["drug_mechanisms", "pharmacokinetics",
                    "pharmacodynamics", "toxicology", "clinical_trials"],
        key_concepts=["drug_classes", "dosage", "side_effects",
                      "interactions", "bioavailability"],
        prompt_templates=[
            "How does {drug} work mechanistically?",
            "What are the side effects of {drug}?",
            "Explain the pharmacokinetics of {drug}.",
        ],
    ),
    FieldDefinition(
        name="nutrition",
        category="health",
        subfields=["macronutrients", "micronutrients", "diet_planning",
                    "sports_nutrition", "clinical_nutrition"],
        key_concepts=["proteins", "carbohydrates", "fats", "vitamins",
                      "minerals", "calories", "dietary_guidelines"],
        prompt_templates=[
            "What are the nutritional benefits of {food}?",
            "Design a meal plan for {goal}.",
            "Explain the role of {nutrient} in the body.",
        ],
    ),
    FieldDefinition(
        name="mental_health",
        category="health",
        subfields=["psychology", "psychiatry", "cognitive_science",
                    "behavioral_science", "therapeutic_approaches"],
        key_concepts=["anxiety", "depression", "cognitive_therapy",
                      "mindfulness", "neuroplasticity", "emotional_intelligence"],
        prompt_templates=[
            "Explain the {concept} in psychology.",
            "What are the symptoms of {condition}?",
            "Describe the {therapy_type} approach.",
        ],
    ),
]

# ─────────────────────────────────────────────────────
# CATEGORY 6: SOCIAL SCIENCES
# ─────────────────────────────────────────────────────

SOCIAL_SCIENCES = [
    FieldDefinition(
        name="economics",
        category="social_sciences",
        subfields=["microeconomics", "macroeconomics", "econometrics",
                    "development_economics", "behavioral_economics",
                    "international_economics"],
        key_concepts=["supply_demand", "gdp", "inflation", "interest_rates",
                      "trade", "markets", "monetary_policy", "fiscal_policy"],
        prompt_templates=[
            "Explain the economic principle of {principle}.",
            "What is the impact of {event} on {indicator}?",
            "Analyze the {economic_concept} in {context}.",
        ],
    ),
    FieldDefinition(
        name="psychology",
        category="social_sciences",
        subfields=["cognitive_psychology", "developmental_psychology",
                    "social_psychology", "industrial_organizational",
                    "clinical_psychology", "neuropsychology"],
        key_concepts=["behavior", "cognition", "development", "personality",
                      "motivation", "perception", "learning", "memory"],
        prompt_templates=[
            "Explain the psychological concept of {concept}.",
            "What does research say about {topic}?",
            "Describe the theory of {theory}.",
        ],
    ),
    FieldDefinition(
        name="sociology",
        category="social_sciences",
        subfields=["social_structures", "social_change", "inequality",
                    "culture", "deviance", "family", "education"],
        key_concepts=["socialization", "stratification", "institutions",
                      "class", "gender", "race", "globalization"],
        prompt_templates=[
            "Analyze the social phenomenon of {phenomenon}.",
            "What are the sociological implications of {event}?",
            "Explain the concept of {concept} in sociology.",
        ],
    ),
    FieldDefinition(
        name="political_science",
        category="social_sciences",
        subfields=["political_theory", "comparative_politics",
                    "international_relations", "public_policy",
                    "political_economy"],
        key_concepts=["democracy", "sovereignty", "power", "governance",
                      "elections", "diplomacy", "institutions"],
        prompt_templates=[
            "Explain the political concept of {concept}.",
            "Compare {system1} and {system2}.",
            "Analyze the policy of {policy}.",
        ],
    ),
    FieldDefinition(
        name="anthropology",
        category="social_sciences",
        subfields=["cultural_anthropology", "physical_anthropology",
                    "linguistic_anthropology", "archaeology"],
        key_concepts=["culture", "society", "ritual", "kinship",
                      "evolution", "language", "artifacts"],
        prompt_templates=[
            "Describe the cultural practice of {practice}.",
            "What does anthropology reveal about {topic}?",
            "Explain the archaeological discovery of {discovery}.",
        ],
    ),
]

# ─────────────────────────────────────────────────────
# CATEGORY 7: HUMANITIES
# ─────────────────────────────────────────────────────

HUMANITIES = [
    FieldDefinition(
        name="philosophy",
        category="humanities",
        subfields=["metaphysics", "epistemology", "ethics", "logic",
                    "aesthetics", "political_philosophy", "existentialism"],
        key_concepts=["truth", "knowledge", "morality", "consciousness",
                      "free_will", "justice", "beauty", "meaning"],
        prompt_templates=[
            "Explain the philosophical argument of {philosopher} about {topic}.",
            "What is the {branch} perspective on {issue}?",
            "Compare the views of {philosopher1} and {philosopher2} on {topic}.",
        ],
    ),
    FieldDefinition(
        name="history",
        category="humanities",
        subfields=["ancient_history", "medieval_history", "modern_history",
                    "contemporary_history", "military_history",
                    "economic_history", "cultural_history"],
        key_concepts=["civilizations", "empires", "revolutions",
                      "wars", "trade_routes", "religions", "governments"],
        prompt_templates=[
            "What happened during the {event}?",
            "Explain the causes and effects of {event}.",
            "Describe life in {era/civilization}.",
        ],
    ),
    FieldDefinition(
        name="literature",
        category="humanities",
        subfields=["fiction", "poetry", "drama", "nonfiction",
                    "literary_criticism", "comparative_literature"],
        key_concepts=["narrative", "character", "theme", "symbolism",
                      "metaphor", "allegory", "irony", "genre"],
        prompt_templates=[
            "Analyze the themes in {work}.",
            "Explain the literary device of {device} in {work}.",
            "Compare the works of {author1} and {author2}.",
        ],
    ),
    FieldDefinition(
        name="linguistics",
        category="humanities",
        subfields=["phonology", "morphology", "syntax", "semantics",
                    "pragmatics", "sociolinguistics", "psycholinguistics",
                    "computational_linguistics"],
        key_concepts=["grammar", "phonemes", "morphemes", "language_families",
                      "translation", "bilingualism", "language_evolution"],
        prompt_templates=[
            "Explain the grammatical concept of {concept}.",
            "How does {language} handle {linguistic_feature}?",
            "Compare the {feature} of {language1} and {language2}.",
        ],
    ),
    FieldDefinition(
        name="religion",
        category="humanities",
        subfields=["comparative_religion", "theology", "religious_history",
                    "spirituality", "mythology"],
        key_concepts=["belief_systems", "rituals", "sacred_texts",
                      "ethics", "afterlife", "creation_myths"],
        prompt_templates=[
            "Explain the {concept} in {religion}.",
            "Compare the views of {religion1} and {religion2} on {topic}.",
            "What are the core beliefs of {religion}?",
        ],
    ),
]

# ─────────────────────────────────────────────────────
# CATEGORY 8: ARTS & CREATIVE
# ─────────────────────────────────────────────────────

ARTS_CREATIVE = [
    FieldDefinition(
        name="visual_arts",
        category="arts",
        subfields=["painting", "sculpture", "photography", "digital_art",
                    "art_history", "art_theory", "illustration"],
        key_concepts=["composition", "color_theory", "perspective",
                      "lighting", "texture", "form", "style", "medium"],
        prompt_templates=[
            "Describe the art movement of {movement}.",
            "Explain the technique of {technique} in {art_form}.",
            "What are the principles of {concept} in art?",
        ],
    ),
    FieldDefinition(
        name="music",
        category="arts",
        subfields=["music_theory", "composition", "performance",
                    "music_history", "ethnomusicology", "production"],
        key_concepts=["scales", "chords", "rhythm", "harmony", "melody",
                      "instruments", "genres", "notation"],
        prompt_templates=[
            "Explain the music theory concept of {concept}.",
            "What makes {genre} distinctive?",
            "Describe the structure of {musical_form}.",
        ],
    ),
    FieldDefinition(
        name="film",
        category="arts",
        subfields=["cinematography", "directing", "screenwriting",
                    "editing", "sound_design", "film_theory"],
        key_concepts=["shot_composition", "narrative_structure",
                      "genre_conventions", "mise_en_scene", "montage"],
        prompt_templates=[
            "Analyze the {technique} in {film}.",
            "Explain the {concept} in filmmaking.",
            "What are the conventions of {genre}?",
        ],
    ),
    FieldDefinition(
        name="design",
        category="arts",
        subfields=["graphic_design", "ux_ui", "industrial_design",
                    "architecture", "fashion", "interior_design"],
        key_concepts=["typography", "layout", "color", "form",
                      "function", "usability", "aesthetics"],
        prompt_templates=[
            "Explain the design principle of {principle}.",
            "How does {designer} approach {problem}?",
            "What are the best practices for {design_task}?",
        ],
    ),
    FieldDefinition(
        name="creative_writing",
        category="arts",
        subfields=["fiction_writing", "poetry_writing", "screenwriting",
                    "journalism", "copywriting", "technical_writing"],
        key_concepts=["storytelling", "voice", "dialogue", "pacing",
                      "revision", "genre", "audience"],
        prompt_templates=[
            "Write a {type} about {topic}.",
            "How do I improve my {writing_skill}?",
            "Analyze the writing style of {author}.",
        ],
    ),
]

# ─────────────────────────────────────────────────────
# CATEGORY 9: BUSINESS & PROFESSIONAL
# ─────────────────────────────────────────────────────

BUSINESS_PROFESSIONAL = [
    FieldDefinition(
        name="business_management",
        category="business",
        subfields=["strategy", "operations", "hr", "project_management",
                    "leadership", "organizational_behavior"],
        key_concepts=["swot_analysis", "okrs", "kpi", "lean",
                      "agile", "scrum", "stakeholder_management"],
        prompt_templates=[
            "How do I {task} in a business context?",
            "Explain the business concept of {concept}.",
            "What are best practices for {business_area}?",
        ],
    ),
    FieldDefinition(
        name="finance",
        category="business",
        subfields=["corporate_finance", "investing", "banking",
                    "accounting", "personal_finance", "fintech"],
        key_concepts=["present_value", "irr", "npv", "portfolio",
                      "diversification", "risk_management", "valuation"],
        prompt_templates=[
            "Calculate the {financial_metric} of {investment}.",
            "Explain the financial concept of {concept}.",
            "What are the risks of {investment_type}?",
        ],
    ),
    FieldDefinition(
        name="marketing",
        category="business",
        subfields=["digital_marketing", "content_marketing", "seo",
                    "social_media", "brand_management", "analytics"],
        key_concepts=["funnel", "conversion", "seo", "sem",
                      "content_strategy", "branding", "roi"],
        prompt_templates=[
            "How do I {marketing_task}?",
            "Explain the marketing concept of {concept}.",
            "What are best practices for {marketing_area}?",
        ],
    ),
    FieldDefinition(
        name="entrepreneurship",
        category="business",
        subfields=["startup_fundamentals", "venture_capital",
                    "product_market_fit", "growth_hacking", "fundraising"],
        key_concepts=["mvp", "pivot", "unit_economics", "burn_rate",
                      "valuation", "pitch_deck"],
        prompt_templates=[
            "How do I {startup_task}?",
            "Explain the startup concept of {concept}.",
            "What are the key metrics for {stage}?",
        ],
    ),
]

# ─────────────────────────────────────────────────────
# CATEGORY 10: PRACTICAL SKILLS
# ─────────────────────────────────────────────────────

PRACTICAL_SKILLS = [
    FieldDefinition(
        name="cooking",
        category="practical",
        subfields=["baking", "grilling", "sauces", "preservation",
                    "international_cuisine", "nutrition"],
        key_concepts=["techniques", "flavor_profiles", "ingredients",
                      "food_safety", "meal_planning", "knife_skills"],
        prompt_templates=[
            "How do I make {dish}?",
            "What are the key ingredients for {cuisine}?",
            "Explain the cooking technique of {technique}.",
        ],
    ),
    FieldDefinition(
        name="home_improvement",
        category="practical",
        subfields=["plumbing", "electrical", "carpentry", "painting",
                    "landscaping", "home_automation"],
        key_concepts=["tools", "materials", "safety", "maintenance",
                      "renovation", "diy"],
        prompt_templates=[
            "How do I {task} in my home?",
            "What tools do I need for {project}?",
            "Explain the process of {home_task}.",
        ],
    ),
    FieldDefinition(
        name="gardening",
        category="practical",
        subfields=["vegetables", "flowers", "trees", "indoor_plants",
                    "soil_science", "pest_control"],
        key_concepts=["planting", "watering", "pruning", "composting",
                      "seasons", "hardiness_zones"],
        prompt_templates=[
            "How do I grow {plant}?",
            "What is the best season to {task}?",
            "Explain the gardening technique of {technique}.",
        ],
    ),
    FieldDefinition(
        name="first_aid",
        category="practical",
        subfields=["emergency_response", "trauma", "medical_emergencies",
                    "preparedness", "wilderness_medicine"],
        key_concepts=["cpr", "wound_care", "fractures", "burns",
                      "poisoning", "allergic_reactions"],
        prompt_templates=[
            "What should I do in case of {emergency}?",
            "How do I treat {injury}?",
            "Explain the first aid procedure for {condition}.",
        ],
    ),
    FieldDefinition(
        name="sustainability",
        category="practical",
        subfields=["renewable_energy", "recycling", "conservation",
                    "green_building", "sustainable_living"],
        key_concepts=["carbon_footprint", "renewable_energy",
                      "waste_reduction", "water_conservation"],
        prompt_templates=[
            "How can I reduce my {metric}?",
            "Explain the concept of {concept} in sustainability.",
            "What are sustainable alternatives to {product}?",
        ],
    ),
]

# ─────────────────────────────────────────────────────
# CATEGORY 11: COMMUNICATION & LANGUAGE
# ─────────────────────────────────────────────────────

COMMUNICATION = [
    FieldDefinition(
        name="public_speaking",
        category="communication",
        subfields=["presentation_skills", "debate", "storytelling",
                    "persuasion", "body_language"],
        key_concepts=["rhetoric", "ethos_pathos_logos", "audience_analysis",
                      "structure", "delivery", "anxiety_management"],
        prompt_templates=[
            "How do I prepare a presentation about {topic}?",
            "What are the principles of {skill}?",
            "Give me tips for {communication_situation}.",
        ],
    ),
    FieldDefinition(
        name="writing",
        category="communication",
        subfields=["academic_writing", "business_writing", "creative_writing",
                    "technical_writing", "email", "report_writing"],
        key_concepts=["clarity", "conciseness", "structure", "tone",
                      "audience", "grammar", "style"],
        prompt_templates=[
            "How do I write a {document_type} about {topic}?",
            "Rewrite this to be more {quality}: {text}",
            "What is the best structure for {document_type}?",
        ],
    ),
    FieldDefinition(
        name="negotiation",
        category="communication",
        subfields=["business_negotiation", "conflict_resolution",
                    "mediation", "diplomacy"],
        key_concepts=["batna", "anchoring", "win_win", "active_listening",
                      "framing", "concession"],
        prompt_templates=[
            "How do I negotiate {situation}?",
            "What are the principles of effective negotiation?",
            "Give me a strategy for {negotiation_scenario}.",
        ],
    ),
]

# ─────────────────────────────────────────────────────
# CATEGORY 12: LAW & GOVERNANCE
# ─────────────────────────────────────────────────────

LAW_GOVERNANCE = [
    FieldDefinition(
        name="constitutional_law",
        category="law",
        subfields=["individual_rights", "separation_of_powers",
                    "federalism", "judicial_review"],
        key_concepts=["amendments", "bill_of_rights", "due_process",
                      "equal_protection", "checks_and_balances"],
        prompt_templates=[
            "Explain the {amendment} Amendment.",
            "What are the legal principles of {topic}?",
            "How does {law} apply to {situation}?",
        ],
    ),
    FieldDefinition(
        name="international_law",
        category="law",
        subfields=["treaty_law", "human_rights", "trade_law",
                    "environmental_law", "war_crimes"],
        key_concepts=["sovereignty", "jurisdiction", "treaties",
                      "icj", "icc", "customary_law"],
        prompt_templates=[
            "Explain the international law regarding {topic}.",
            "What obligations does {country} have under {treaty}?",
            "How is {issue} regulated internationally?",
        ],
    ),
]

# ─────────────────────────────────────────────────────
# CATEGORY 13: TOOL-USE & ACTION TRAINING
# ─────────────────────────────────────────────────────

TOOL_USE = [
    FieldDefinition(
        name="calculator_use",
        category="tool_use",
        subfields=["arithmetic", "algebra", "calculus", "statistics",
                    "unit_conversion"],
        key_concepts=["expression", "evaluation", "computation"],
        prompt_templates=[
            "Calculate {expression}.",
            "What is {expression}?",
            "Compute {expression}.",
            "Solve {expression}.",
        ],
        tool_use_examples=[
            {"tool": "calculator", "args": {"expression": "2 + 2"}, "result": "4"},
            {"tool": "calculator", "args": {"expression": "15 * 7"}, "result": "105"},
            {"tool": "calculator", "args": {"expression": "100 / 3"}, "result": "33.333333333333336"},
            {"tool": "calculator", "args": {"expression": "2 ** 20"}, "result": "1048576"},
            {"tool": "calculator", "args": {"expression": "3.14 * 5 ** 2"}, "result": "78.5"},
            {"tool": "calculator", "args": {"expression": "1000 / 7"}, "result": "142.85714285714286"},
            {"tool": "calculator", "args": {"expression": "256 ** 0.5"}, "result": "16.0"},
            {"tool": "calculator", "args": {"expression": "999 + 1"}, "result": "1000"},
        ],
    ),
    FieldDefinition(
        name="file_operations",
        category="tool_use",
        subfields=["read_file", "list_directory", "write_file"],
        key_concepts=["file_path", "content", "directory"],
        prompt_templates=[
            "Read the file at {path}.",
            "What files are in {directory}?",
            "Show me the contents of {file}.",
        ],
        tool_use_examples=[
            {"tool": "list_dir", "args": {"path": "."}, "result": "d experiments\nd foundation\nd intelligence\nd serving\n  pyproject.toml"},
            {"tool": "read_file", "args": {"path": "pyproject.toml"}, "result": "[project]\nname = \"silverwing-ml\"\nversion = \"0.1.0\""},
        ],
    ),
    FieldDefinition(
        name="web_browsing",
        category="tool_use",
        subfields=["search", "fetch_url", "scrape_content"],
        key_concepts=["url", "query", "content_extraction"],
        prompt_templates=[
            "Search for {query}.",
            "What is on the page at {url}?",
            "Find information about {topic} online.",
        ],
        tool_use_examples=[
            {"tool": "web_search", "args": {"query": "latest AI news"}, "result": "Found 5 results for 'latest AI news'"},
            {"tool": "web_fetch", "args": {"url": "https://example.com"}, "result": "Page content: ..."},
        ],
    ),
]

# ─────────────────────────────────────────────────────
# ALL FIELDS COMBINED
# ─────────────────────────────────────────────────────

ALL_FIELDS: list[FieldDefinition] = (
    MATHEMATICS
    + PHYSICS_ENGINEERING
    + COMPUTER_SCIENCE
    + NATURAL_SCIENCES
    + MEDICINE_HEALTH
    + SOCIAL_SCIENCES
    + HUMANITIES
    + ARTS_CREATIVE
    + BUSINESS_PROFESSIONAL
    + PRACTICAL_SKILLS
    + COMMUNICATION
    + LAW_GOVERNANCE
    + TOOL_USE
)

ALL_CATEGORIES = sorted({f.category for f in ALL_FIELDS})
TOTAL_FIELDS = len(ALL_FIELDS)
TOTAL_SUBFIELDS = sum(len(f.subfields) for f in ALL_FIELDS)
TOTAL_CONCEPTS = sum(len(f.key_concepts) for f in ALL_FIELDS)
TOTAL_TEMPLATES = sum(len(f.prompt_templates) for f in ALL_FIELDS)
TOTAL_TOOL_EXAMPLES = sum(len(f.tool_use_examples) for f in ALL_FIELDS)
