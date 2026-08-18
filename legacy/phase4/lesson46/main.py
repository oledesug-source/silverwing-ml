# Silverwing ML
# Phase 4 - Lesson 46
# Vector Indexing and Scalable Semantic Retrieval


import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


print("=== SILVERWING ML ===")
print("Phase 4 - Lesson 46")
print("Vector Indexing and Semantic Retrieval")
print()


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_FILE = (
        BASE_DIR / "silverwing_vector_memory.db"
)

INDEX_FILE = (
        BASE_DIR / "silverwing_memory.index"
)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


print("TEST 1: Configuration")
print()

print(
    "Database:",
    DATABASE_FILE
)

print(
    "Index:",
    INDEX_FILE
)

print(
    "Embedding model:",
    EMBEDDING_MODEL_NAME
)

print()


# ==================================================
# 2. LOAD EMBEDDING MODEL
# ==================================================

print("TEST 2: Load Embedding Model")
print()


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


embedding_dimension = (
    embedding_model
    .get_sentence_embedding_dimension()
)


print(
    "Embedding dimension:",
    embedding_dimension
)

print()


# ==================================================
# 3. DATABASE
# ==================================================

print("TEST 3: Database Connection")
print()


connection = sqlite3.connect(
    DATABASE_FILE
)

connection.row_factory = (
    sqlite3.Row
)


cursor = connection.cursor()


cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS memories (
                                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                                            content TEXT NOT NULL,
                                            memory_type TEXT,
                                            subject TEXT,
                                            importance REAL DEFAULT 0.5,
                                            source TEXT,
                                            created_at TEXT NOT NULL
    )
    """
)


connection.commit()


print(
    "Database ready."
)

print()


# ==================================================
# 4. TIMESTAMP
# ==================================================

def current_timestamp():

    return datetime.now(
        timezone.utc
    ).isoformat()


# ==================================================
# 5. MEMORY DATA
# ==================================================

memories = [

    {
        "content": (
            "Silverwing is designed as a personal "
            "general-purpose AI system."
        ),
        "memory_type": "architecture",
        "subject": "Silverwing",
        "importance": 1.0,
        "source": "user"
    },

    {
        "content": (
            "Silverwing should support multiple "
            "independent tasks and multitasking."
        ),
        "memory_type": "capability",
        "subject": "multitasking",
        "importance": 0.95,
        "source": "user"
    },

    {
        "content": (
            "The LLM should be replaceable and "
            "should operate as one component of "
            "the larger AI system."
        ),
        "memory_type": "architecture",
        "subject": "LLM",
        "importance": 0.95,
        "source": "architecture"
    },

    {
        "content": (
            "Silverwing needs persistent memory "
            "across application restarts."
        ),
        "memory_type": "memory",
        "subject": "persistence",
        "importance": 0.9,
        "source": "lesson"
    },

    {
        "content": (
            "Semantic retrieval allows Silverwing "
            "to retrieve memories by meaning."
        ),
        "memory_type": "memory",
        "subject": "semantic retrieval",
        "importance": 0.9,
        "source": "lesson"
    },

    {
        "content": (
            "Machine-learning models can provide "
            "specialized predictions to Silverwing."
        ),
        "memory_type": "capability",
        "subject": "machine learning",
        "importance": 0.85,
        "source": "lesson"
    },

    {
        "content": (
            "The task orchestrator should coordinate "
            "independent operations and services."
        ),
        "memory_type": "architecture",
        "subject": "orchestration",
        "importance": 0.9,
        "source": "architecture"
    },

    {
        "content": (
            "Silverwing should communicate through "
            "text, APIs, services, and eventually "
            "voice and other interfaces."
        ),
        "memory_type": "capability",
        "subject": "communication",
        "importance": 0.85,
        "source": "architecture"
    }
]


# ==================================================
# 6. STORE MEMORIES
# ==================================================

print("TEST 4: Store Memories")
print()


for memory in memories:

    cursor.execute(
        """
        INSERT INTO memories (
            content,
            memory_type,
            subject,
            importance,
            source,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            memory["content"],
            memory["memory_type"],
            memory["subject"],
            memory["importance"],
            memory["source"],
            current_timestamp()
        )
    )


connection.commit()


print(
    "Stored memories:",
    len(memories)
)

print()


# ==================================================
# 7. FETCH MEMORIES
# ==================================================

print("TEST 5: Fetch Memory Records")
print()


cursor.execute(
    """
    SELECT *
    FROM memories
    ORDER BY id ASC
    """
)


rows = cursor.fetchall()


print(
    "Records:",
    len(rows)
)

print()


# ==================================================
# 8. CREATE EMBEDDINGS
# ==================================================

print("TEST 6: Create Embeddings")
print()


texts = [
    row["content"]
    for row in rows
]


embeddings = embedding_model.encode(
    texts,
    normalize_embeddings=True
)


embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)


print(
    "Embedding matrix shape:",
    embeddings.shape
)

print()


# ==================================================
# 9. BUILD FAISS INDEX
# ==================================================

print("TEST 7: Build FAISS Index")
print()


index = faiss.IndexFlatIP(
    embedding_dimension
)


index.add(
    embeddings
)


print(
    "Vectors in index:",
    index.ntotal
)

print()


# ==================================================
# 10. SAVE INDEX
# ==================================================

print("TEST 8: Save Vector Index")
print()


faiss.write_index(
    index,
    str(INDEX_FILE)
)


print(
    "Index saved:"
)

print(
    INDEX_FILE
)

print()


# ==================================================
# 11. SEMANTIC SEARCH FUNCTION
# ==================================================

def semantic_search(
        query,
        top_k=5
):

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True
        )
    )


    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )


    scores, indices = index.search(
        query_embedding,
        top_k
    )


    results = []


    for score, memory_index in zip(
            scores[0],
            indices[0]
    ):

        if memory_index < 0:

            continue


        row = rows[
            int(memory_index)
        ]


        results.append(
            {
                "id":
                    row["id"],

                "content":
                    row["content"],

                "memory_type":
                    row["memory_type"],

                "subject":
                    row["subject"],

                "importance":
                    row["importance"],

                "similarity":
                    float(score)
            }
        )


    return results


# ==================================================
# 12. SEARCH TEST 1
# ==================================================

print("TEST 9: Semantic Search")
print()


query_1 = (
    "How should Silverwing remember "
    "important information?"
)


results_1 = semantic_search(
    query_1,
    top_k=5
)


print(
    "Query:",
    query_1
)

print()


for result in results_1:

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
        "-" * 60
    )


print()


# ==================================================
# 13. SEARCH TEST 2
# ==================================================

print("TEST 10: Architecture Search")
print()


query_2 = (
    "What parts of the system should exist "
    "outside the language model?"
)


results_2 = semantic_search(
    query_2,
    top_k=5
)


print(
    "Query:",
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
# 14. SEARCH TEST 3
# ==================================================

print("TEST 11: Multitasking Search")
print()


query_3 = (
    "How can Silverwing handle many operations "
    "at the same time?"
)


results_3 = semantic_search(
    query_3,
    top_k=3
)


print(
    "Query:",
    query_3
)

print()


for result in results_3:

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
# 15. SEARCH TEST 4
# ==================================================

print("TEST 12: Communication Search")
print()


query_4 = (
    "How should the AI communicate with users "
    "and external systems?"
)


results_4 = semantic_search(
    query_4,
    top_k=4
)


for result in results_4:

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
# 16. IMPORTANCE-AWARE RANKING
# ==================================================

print("TEST 13: Hybrid Ranking")
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


for result in results_1:

    updated = dict(
        result
    )


    updated["hybrid_score"] = (
        hybrid_score(
            result["similarity"],
            result["importance"]
        )
    )


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
        "Similarity:",
        round(
            result["similarity"],
            4
        )
    )

    print(
        "Importance:",
        result["importance"]
    )

    print(
        "Hybrid:",
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
# 17. BUILD CONTEXT
# ==================================================

print("TEST 14: Context Builder")
print()


def build_context(
        results,
        maximum_results=3
):

    selected = results[
        :maximum_results
    ]


    lines = []


    for result in selected:

        lines.append(
            (
                f"[Memory "
                f"{result['similarity']:.3f}] "
                f"{result['content']}"
            )
        )


    return "\n".join(
        lines
    )


context = build_context(
    hybrid_results,
    maximum_results=3
)


print(
    context
)

print()


# ==================================================
# 18. SIMULATE LLM CONTEXT
# ==================================================

print("TEST 15: LLM Context")
print()


user_query = (
    "What architecture should Silverwing use?"
)


llm_context = (
        "RELEVANT MEMORY\n"
        +
        context
        +
        "\n\nUSER REQUEST\n"
        +
        user_query
)


print(
    llm_context
)

print()


# ==================================================
# 19. LOAD SAVED INDEX
# ==================================================

print("TEST 16: Reload Vector Index")
print()


loaded_index = faiss.read_index(
    str(INDEX_FILE)
)


print(
    "Loaded vectors:",
    loaded_index.ntotal
)

print()


# ==================================================
# 20. VERIFY INDEX
# ==================================================

print("TEST 17: Index Verification")
print()


verification_query = (
    "What is Silverwing?"
)


verification_embedding = (
    embedding_model.encode(
        [verification_query],
        normalize_embeddings=True
    )
)


verification_embedding = (
    np.asarray(
        verification_embedding,
        dtype=np.float32
    )
)


scores, indices = loaded_index.search(
    verification_embedding,
    3
)


for score, index_position in zip(
        scores[0],
        indices[0]
):

    if index_position < 0:

        continue


    row = rows[
        int(index_position)
    ]


    print(
        round(
            float(score),
            4
        ),
        "->",
        row["content"]
    )


print()


# ==================================================
# 21. INDEX STATISTICS
# ==================================================

print("TEST 18: Index Statistics")
print()


print(
    "Vectors:",
    loaded_index.ntotal
)

print(
    "Dimension:",
    loaded_index.d
)

print(
    "Metric:",
    "Inner Product"
)

print()


# ==================================================
# 22. WHY NORMALIZED EMBEDDINGS MATTER
# ==================================================

print("TEST 19: Similarity Metric")
print()

print(
    "The embeddings are normalized."
)

print()

print(
    "For normalized vectors, inner product "
    "corresponds to cosine similarity."
)

print()

print(
    "This allows FAISS IndexFlatIP to perform "
    "similarity search using the embedding vectors."
)

print()


# ==================================================
# 23. COMPLEXITY CONCEPT
# ==================================================

print("TEST 20: Scaling Concept")
print()

print(
    "The current index is still a simple "
    "exact-search index."
)

print()

print(
    "More advanced FAISS indexes can trade "
    "some exactness for much faster search "
    "on very large collections."
)

print()


# ==================================================
# 24. MEMORY RETRIEVAL PIPELINE
# ==================================================

print("SEMANTIC MEMORY PIPELINE")
print()

print("Memory")
print(" ↓")
print("Embedding Model")
print(" ↓")
print("Vector")
print(" ↓")
print("FAISS Index")
print(" ↓")
print("Similarity Search")
print(" ↓")
print("Top-K Memories")
print(" ↓")
print("Ranking")
print(" ↓")
print("Context Builder")
print(" ↓")
print("LLM")

print()


# ==================================================
# 25. FUTURE SCALABLE MEMORY
# ==================================================

print("FUTURE SCALABLE MEMORY")
print()

print("Millions of memories")
print("       ↓")
print("Embedding service")
print("       ↓")
print("Vector database")
print("       ↓")
print("Approximate nearest-neighbor index")
print("       ↓")
print("Top-K retrieval")
print("       ↓")
print("Reranking")
print("       ↓")
print("LLM context")

print()


# ==================================================
# 26. PERSONAL AI MEMORY
# ==================================================

print("PERSONAL AI MEMORY")
print()

print(
    "Semantic indexing allows Silverwing to "
    "search a growing memory collection without "
    "depending entirely on exact keywords."
)

print()

print(
    "The same retrieval architecture can later "
    "be used for project knowledge, documents, "
    "past tasks, research, and other stored data."
)

print()


# ==================================================
# 27. IMPORTANT LIMITATION
# ==================================================

print("IMPORTANT LIMITATION")
print()

print(
    "This lesson uses one process and one local "
    "FAISS index for educational purposes."
)

print()

print(
    "A production memory service needs persistent "
    "index management, updates, deletion handling, "
    "metadata filtering, concurrency controls, "
    "backup, and evaluation."
)

print()


# ==================================================
# 28. SILVERWING PROGRESS
# ==================================================

print("SILVERWING PROGRESS")
print()

print("Persistent Memory")
print("       ↓")
print("Embeddings")
print("       ↓")
print("Semantic Search")
print("       ↓")
print("Vector Index")
print("       ↓")
print("Top-K Retrieval")
print("       ↓")
print("Context")
print("       ↓")
print("LLM")

print()


# ==================================================
# LESSON COMPLETE
# ==================================================

print("=== LESSON 46 COMPLETE ===")


# ==================================================
# CLOSE DATABASE
# ==================================================

connection.close()
