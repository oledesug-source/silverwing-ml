# Silverwing ML
# Phase 4 - Lesson 45
# Semantic Memory with Embeddings
#
# Goal:
# Store memories and retrieve them according
# to semantic similarity rather than exact
# keyword matching.


import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sentence_transformers import (
    SentenceTransformer
)


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 45")
print("Semantic Memory with Embeddings")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_FILE = (
        BASE_DIR / "semantic_memory.db"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


print("TEST 1: Configuration")
print()

print(
    "Database:",
    DATABASE_FILE
)

print(
    "Embedding model:",
    EMBEDDING_MODEL
)

print()


# ==================================================
# 2. LOAD EMBEDDING MODEL
# ==================================================

print("TEST 2: Load Embedding Model")
print()


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


print(
    "Embedding model loaded."
)

print()


# ==================================================
# 3. MODEL INFORMATION
# ==================================================

print("TEST 3: Embedding Information")
print()


embedding_dimension = (
    embedding_model.get_sentence_embedding_dimension()
)


print(
    "Embedding dimension:",
    embedding_dimension
)

print()


# ==================================================
# 4. DATABASE CONNECTION
# ==================================================

print("TEST 4: Database Connection")
print()


connection = sqlite3.connect(
    DATABASE_FILE
)

connection.row_factory = sqlite3.Row


cursor = connection.cursor()


print(
    "SQLite database connected."
)

print()


# ==================================================
# 5. CREATE MEMORY TABLE
# ==================================================

print("TEST 5: Create Semantic Memory Table")
print()


cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS semantic_memories (
                                                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                     memory_type TEXT NOT NULL,
                                                     subject TEXT,
                                                     content TEXT NOT NULL,
                                                     importance REAL DEFAULT 0.5,
                                                     source TEXT,
                                                     embedding BLOB,
                                                     created_at TEXT NOT NULL
    )
    """
)


connection.commit()


print(
    "Semantic memory table ready."
)

print()


# ==================================================
# 6. TIMESTAMP
# ==================================================

def current_timestamp():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 7. SERIALIZE EMBEDDING
# ==================================================

def serialize_embedding(
        vector
):
    """
    Convert a NumPy vector into bytes
    for SQLite storage.
    """

    vector = np.asarray(
        vector,
        dtype=np.float32
    )


    return vector.tobytes()


# ==================================================
# 8. DESERIALIZE EMBEDDING
# ==================================================

def deserialize_embedding(
        blob
):
    """
    Convert SQLite bytes back into a vector.
    """

    return np.frombuffer(
        blob,
        dtype=np.float32
    )


# ==================================================
# 9. COSINE SIMILARITY
# ==================================================

def cosine_similarity(
        vector_a,
        vector_b
):
    """
    Calculate cosine similarity between
    two vectors.
    """

    vector_a = np.asarray(
        vector_a,
        dtype=np.float32
    )

    vector_b = np.asarray(
        vector_b,
        dtype=np.float32
    )


    denominator = (
            np.linalg.norm(vector_a)
            *
            np.linalg.norm(vector_b)
    )


    if denominator == 0:

        return 0.0


    return float(
        np.dot(
            vector_a,
            vector_b
        )
        /
        denominator
    )


# ==================================================
# 10. SEMANTIC MEMORY MANAGER
# ==================================================

class SemanticMemory:

    def __init__(
            self,
            connection,
            embedding_model
    ):

        self.connection = connection

        self.embedding_model = (
            embedding_model
        )


    # ----------------------------------------------
    # Create embedding
    # ----------------------------------------------

    def embed(
            self,
            text
    ):

        vector = (
            self.embedding_model.encode(
                text,
                normalize_embeddings=True
            )
        )


        return np.asarray(
            vector,
            dtype=np.float32
        )


    # ----------------------------------------------
    # Store memory
    # ----------------------------------------------

    def store(
            self,
            content,
            memory_type="general",
            subject=None,
            importance=0.5,
            source="system"
    ):

        vector = self.embed(
            content
        )


        cursor = self.connection.cursor()


        cursor.execute(
            """
            INSERT INTO semantic_memories (
                memory_type,
                subject,
                content,
                importance,
                source,
                embedding,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_type,
                subject,
                content,
                importance,
                source,
                serialize_embedding(
                    vector
                ),
                current_timestamp()
            )
        )


        self.connection.commit()


        return cursor.lastrowid


    # ----------------------------------------------
    # Retrieve all memories
    # ----------------------------------------------

    def all_memories(self):

        cursor = self.connection.cursor()


        cursor.execute(
            """
            SELECT *
            FROM semantic_memories
            ORDER BY id ASC
            """
        )


        return cursor.fetchall()


    # ----------------------------------------------
    # Semantic search
    # ----------------------------------------------

    def search(
            self,
            query,
            limit=5
    ):

        query_vector = self.embed(
            query
        )


        rows = self.all_memories()


        results = []


        for row in rows:

            memory_vector = (
                deserialize_embedding(
                    row["embedding"]
                )
            )


            similarity = cosine_similarity(
                query_vector,
                memory_vector
            )


            results.append(
                {
                    "id": row["id"],
                    "memory_type":
                        row["memory_type"],
                    "subject":
                        row["subject"],
                    "content":
                        row["content"],
                    "importance":
                        row["importance"],
                    "source":
                        row["source"],
                    "similarity":
                        similarity
                }
            )


        results.sort(
            key=lambda item: (
                item["similarity"],
                item["importance"]
            ),
            reverse=True
        )


        return results[:limit]


# ==================================================
# 11. CREATE MEMORY SYSTEM
# ==================================================

print("TEST 6: Create Semantic Memory")
print()


memory = SemanticMemory(
    connection,
    embedding_model
)


print(
    "Semantic memory manager created."
)

print()


# ==================================================
# 12. STORE MEMORIES
# ==================================================

print("TEST 7: Store Memories")
print()


memories = [

    {
        "content": (
            "Silverwing is being designed as "
            "a personal general-purpose AI system."
        ),
        "type": "architecture",
        "subject": "Silverwing",
        "importance": 1.0,
        "source": "user"
    },

    {
        "content": (
            "Silverwing should be able to manage "
            "multiple independent tasks."
        ),
        "type": "capability",
        "subject": "multitasking",
        "importance": 0.95,
        "source": "user"
    },

    {
        "content": (
            "The LLM should be one component of "
            "the larger AI architecture rather than "
            "the entire system."
        ),
        "type": "architecture",
        "subject": "LLM",
        "importance": 0.95,
        "source": "architecture"
    },

    {
        "content": (
            "Silverwing needs persistent memory "
            "so information can survive application "
            "restarts."
        ),
        "type": "memory",
        "subject": "persistence",
        "importance": 0.9,
        "source": "lesson"
    },

    {
        "content": (
            "Machine-learning models can provide "
            "specialized predictions to the AI."
        ),
        "type": "capability",
        "subject": "machine learning",
        "importance": 0.85,
        "source": "lesson"
    },

    {
        "content": (
            "The task orchestrator should coordinate "
            "work between independent services."
        ),
        "type": "architecture",
        "subject": "orchestration",
        "importance": 0.9,
        "source": "architecture"
    }
]


for item in memories:

    memory_id = memory.store(
        content=item["content"],
        memory_type=item["type"],
        subject=item["subject"],
        importance=item["importance"],
        source=item["source"]
    )


    print(
        "Stored memory:",
        memory_id
    )


print()


# ==================================================
# 13. INSPECT AN EMBEDDING
# ==================================================

print("TEST 8: Inspect Embedding")
print()


example_text = (
    "Silverwing needs persistent memory."
)


example_vector = memory.embed(
    example_text
)


print(
    "Text:",
    example_text
)

print()

print(
    "Vector shape:",
    example_vector.shape
)

print()

print(
    "First values:"
)

print(
    example_vector[:10]
)

print()


# ==================================================
# 14. SEMANTIC SEARCH
# ==================================================

print("TEST 9: Semantic Search")
print()


query = (
    "How should the AI remember information?"
)


results = memory.search(
    query,
    limit=5
)


print(
    "Query:"
)

print(
    query
)

print()


for result in results:

    print(
        "Similarity:",
        round(
            result["similarity"],
            4
        )
    )

    print(
        "Type:",
        result["memory_type"]
    )

    print(
        "Subject:",
        result["subject"]
    )

    print(
        "Memory:",
        result["content"]
    )

    print(
        "Importance:",
        result["importance"]
    )

    print(
        "Source:",
        result["source"]
    )

    print(
        "-" * 60
    )


print()


# ==================================================
# 15. SECOND SEMANTIC QUERY
# ==================================================

print("TEST 10: Architecture Search")
print()


query_2 = (
    "What components should exist outside "
    "the language model?"
)


results_2 = memory.search(
    query_2,
    limit=5
)


print(
    "Query:"
)

print(
    query_2
)

print()


for result in results_2:

    print(
        round(
            result["similarity"],
            4
        ),
        "->",
        result["content"]
    )

print()


# ==================================================
# 16. THIRD SEMANTIC QUERY
# ==================================================

print("TEST 11: Multitasking Search")
print()


query_3 = (
    "How should Silverwing handle several "
    "tasks at the same time?"
)


results_3 = memory.search(
    query_3,
    limit=3
)


print(
    "Query:"
)

print(
    query_3
)

print()


for result in results_3:

    print(
        "Similarity:",
        round(
            result["similarity"],
            4
        )
    )

    print(
        result["content"]
    )

    print()


# ==================================================
# 17. KEYWORD VS SEMANTIC SEARCH
# ==================================================

print("TEST 12: Retrieval Comparison")
print()


keyword = "memory"


keyword_matches = []


for result in memory.all_memories():

    if (
            keyword.lower()
            in result["content"].lower()
    ):

        keyword_matches.append(
            result
        )


print(
    "Keyword search matches:",
    len(keyword_matches)
)

print()


semantic_matches = memory.search(
    "How can Silverwing remember "
    "important information?",
    limit=5
)


print(
    "Semantic search results:"
)

print()


for result in semantic_matches:

    print(
        round(
            result["similarity"],
            4
        ),
        "->",
        result["content"]
    )


print()


# ==================================================
# 18. RELEVANCE THRESHOLD
# ==================================================

print("TEST 13: Relevance Threshold")
print()


def filter_by_relevance(
        results,
        threshold=0.30
):

    return [
        result
        for result in results
        if result["similarity"]
           >= threshold
    ]


filtered_results = (
    filter_by_relevance(
        semantic_matches,
        threshold=0.30
    )
)


print(
    "Results above threshold:",
    len(filtered_results)
)

print()


for result in filtered_results:

    print(
        round(
            result["similarity"],
            4
        ),
        "->",
        result["content"]
    )


print()


# ==================================================
# 19. HYBRID SCORE
# ==================================================

print("TEST 14: Hybrid Ranking")
print()


def hybrid_score(
        similarity,
        importance
):

    return (
            0.8 * similarity
            +
            0.2 * importance
    )


hybrid_results = []


for result in semantic_matches:

    score = hybrid_score(
        result["similarity"],
        result["importance"]
    )


    updated = dict(
        result
    )


    updated["hybrid_score"] = score


    hybrid_results.append(
        updated
    )


hybrid_results.sort(
    key=lambda item:
    item["hybrid_score"],
    reverse=True
)


for result in hybrid_results:

    print(
        "Semantic:",
        round(
            result["similarity"],
            4
        ),
        "| Importance:",
        result["importance"],
        "| Hybrid:",
        round(
            result["hybrid_score"],
            4
        )
    )

    print(
        result["content"]
    )

    print()


# ==================================================
# 20. BUILD MEMORY CONTEXT
# ==================================================

print("TEST 15: Context Construction")
print()


def build_memory_context(
        results
):

    lines = []


    for result in results:

        lines.append(
            (
                f"[Similarity "
                f"{result['similarity']:.3f}] "
                f"{result['content']}"
            )
        )


    return "\n".join(
        lines
    )


memory_context = (
    build_memory_context(
        hybrid_results[:3]
    )
)


print(
    memory_context
)

print()


# ==================================================
# 21. MEMORY → LLM PIPELINE
# ==================================================

print("TEST 16: Memory to LLM Pipeline")
print()

print(
    "User question:"
)

print(
    query
)

print()

print(
    "Retrieved semantic memories:"
)

print(
    memory_context
)

print()

print(
    "These memories can now be inserted "
    "into the LLM's context."
)

print()


# ==================================================
# 22. EMBEDDING PROPERTIES
# ==================================================

print("TEST 17: Embedding Properties")
print()


vector_norm = np.linalg.norm(
    example_vector
)


print(
    "Vector norm:",
    vector_norm
)

print()

print(
    "Because normalize_embeddings=True "
    "was used, the vector is approximately "
    "unit length."
)

print()


# ==================================================
# 23. MEMORY ARCHITECTURE
# ==================================================

print("SEMANTIC MEMORY ARCHITECTURE")
print()

print("Information")
print("      ↓")
print("Embedding Model")
print("      ↓")
print("Vector Representation")
print("      ↓")
print("Persistent Memory")
print("      ↓")
print("Semantic Search")
print("      ↓")
print("Relevant Memories")
print("      ↓")
print("Context Builder")
print("      ↓")
print("LLM / Agent")

print()


# ==================================================
# 24. FUTURE VECTOR DATABASE
# ==================================================

print("FUTURE MEMORY ARCHITECTURE")
print()

print("SQLite")
print("   +")
print("Embedding Model")
print("   ↓")
print("Vector Index")
print("   ↓")
print("Vector Database")
print("   ↓")
print("Fast Semantic Retrieval")

print()


# ==================================================
# 25. IMPORTANT LIMITATION
# ==================================================

print("IMPORTANT LIMITATION")
print()

print(
    "This lesson performs a simple linear scan "
    "over stored vectors."
)

print()

print(
    "That is acceptable for a small educational "
    "memory store but does not scale efficiently "
    "to millions of memories."
)

print()

print(
    "Large memory systems require a vector index "
    "or vector database."
)

print()


# ==================================================
# 26. PERSONAL AI MEMORY
# ==================================================

print("PERSONAL AI MEMORY")
print()

print(
    "Semantic memory allows Silverwing to retrieve "
    "information based on meaning rather than "
    "requiring an exact keyword match."
)

print()

print(
    "This is an important foundation for a personal "
    "AI that needs continuity across many projects "
    "and conversations."
)

print()


# ==================================================
# 27. SILVERWING PROGRESS
# ==================================================

print("SILVERWING PROGRESS")
print()

print("Persistent Memory")
print("       ↓")
print("Embeddings")
print("       ↓")
print("Semantic Retrieval")
print("       ↓")
print("Relevant Context")
print("       ↓")
print("LLM")
print("       ↓")
print("Reasoning")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 45 COMPLETE ===")


# ==================================================
# CLOSE DATABASE
# ==================================================

connection.close()
