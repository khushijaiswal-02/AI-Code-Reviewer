"""
This module contains the core RAG logic

Responsibilities:
1. Load the pre-built FAISS vector database (created in preprocessing.ipynb)
2. Provide a retriever that finds the most relevant guideline chunks for a query
3. Build the final prompt that combines retrieved context + user's code

WHY THIS FILE EXISTS:
The notebook (preprocessing.ipynb) is for BUILDING the knowledge base (one-time setup).
This file is for USING the knowledge base every time a user submits code.
We load the FAISS index from disk instead of rebuilding it every time the app runs,
because rebuilding embeddings for 309 chunks would be slow and wasteful.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# Embedding Model=This must be the same model used in preprocessing.ipynb to create the FAISS index
# If you change this, the vectors won't match and retrieval will break.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Language detect= We detect language from keywords/syntax so the prompt can give language-specific review guidance.

def detect_language(code: str) -> str:
    
    code_lower = code.lower()

    # Order matters: more specific checks first to avoid false positives.
    # C++ check before C since C++ files often contain C-like syntax too.
    patterns = {
        "Python":     ["def ", "import ", "print(", "elif ", "self.", "__init__", "pip "],
        "Java":       ["public class", "public static void main", "system.out.print",
                       "import java.", "throws ", "extends ", "implements "],
        "C++":        ["#include <iostream>", "std::", "cout <<", "cin >>",
                       "namespace ", "template<", "vector<"],
        "C":          ["#include <stdio.h>", "printf(", "scanf(", "malloc(",
                       "free(", "int main(", "#include <stdlib.h>"],
        "JavaScript": ["console.log(", "const ", "let ", "function(", "=>",
                       "document.", "require(", "module.exports", "async "],
    }

    scores = {lang: 0 for lang in patterns}
    for lang, keywords in patterns.items():
        for kw in keywords:
            if kw.lower() in code_lower:
                scores[lang] += 1

    best_lang = max(scores, key=scores.get)
    return best_lang if scores[best_lang] > 0 else "Unknown"



# LANGUAGE-SPECIFIC REVIEW GUIDELINES

LANGUAGE_GUIDELINES = {
    "Java": """
Language-Specific Focus (Java):
- Check for SQL injection via string concatenation in queries
- Look for improper exception handling (catching generic Exception, empty catch blocks)
- Identify resource leaks (unclosed streams, connections not in try-with-resources)
- Check for null pointer dereference risks
- Look for hardcoded credentials or sensitive data
- Identify thread-safety issues in multi-threaded contexts
- Review proper use of access modifiers (public/private/protected)
- Check for violations of Java naming conventions
""",
    "Python": """
Language-Specific Focus (Python):
- Check for use of eval() or exec() with untrusted input (code injection risk)
- Look for SQL injection in raw query strings
- Identify mutable default arguments (common Python bug: def f(x=[]))
- Check for broad exception handling (bare except: or except Exception)
- Look for hardcoded secrets or API keys
- Check for insecure use of pickle, yaml.load(), or subprocess with shell=True
- Identify missing input validation
- Review PEP 8 compliance (naming, spacing, line length)
""",
    "C": """
Language-Specific Focus (C):
- Check for buffer overflow risks (strcpy, gets, sprintf without bounds)
- Look for memory leaks (malloc without corresponding free)
- Identify use of deprecated/unsafe functions (gets, strcpy - prefer strncpy)
- Check for integer overflow and underflow
- Look for null pointer dereference without checks
- Identify use of uninitialized variables
- Check for format string vulnerabilities (printf(userInput) instead of printf("%s", userInput))
- Review proper error handling for system calls
""",
    "C++": """
Language-Specific Focus (C++):
- Check for memory management issues (new without delete, prefer smart pointers)
- Look for buffer overflows and raw array usage (prefer std::vector/std::array)
- Identify use of raw pointers where smart pointers (unique_ptr, shared_ptr) should be used
- Check for missing virtual destructors in base classes
- Look for object slicing issues
- Identify improper use of casting (prefer static_cast over C-style casts)
- Check for exception safety in constructors
- Review RAII (Resource Acquisition Is Initialization) compliance
""",
    "JavaScript": """
Language-Specific Focus (JavaScript):
- Check for Cross-Site Scripting (XSS) risks (innerHTML, document.write with user input)
- Look for prototype pollution vulnerabilities
- Identify use of eval() with untrusted data
- Check for insecure direct object references
- Look for missing input validation and sanitization
- Identify use of == instead of === (type coercion bugs)
- Check for sensitive data exposure in client-side code
- Review async/await and Promise error handling
""",
    "Unknown": """
Language-Specific Focus (General):
- Apply universal secure coding principles
- Check for injection vulnerabilities
- Look for improper error handling
- Identify hardcoded sensitive information
- Check for input validation issues
""",
}



# PROMPT TEMPLATE

REVIEW_PROMPT = PromptTemplate(
    input_variables=["context", "language", "language_guidelines", "question"],
    template="""
You are an expert AI Code Reviewer with deep knowledge of {language} and software security best practices.

Your task is to review the user's {language} source code using the provided reference context from OWASP, Clean Code Guidelines, and language best practices.

{language_guidelines}

General Review Instructions:
1. Analyze the code for:
   - Security vulnerabilities
   - Code smells and poor practices
   - Performance issues
   - Maintainability problems
   - Language-specific anti-patterns

2. For every issue found, provide:
   - Issue name
   - Severity (Low / Medium / High / Critical)
   - Explanation of why it is a problem
   - Recommended fix with corrected code example
   - Reference to the source guideline (if available in context)

3. If the provided context does not contain enough information for a specific issue:
   - Apply general secure coding knowledge for {language}
   - Clearly note when a recommendation is based on general best practices
     rather than the provided documents

Reference Context (from knowledge base):
{context}

User's {language} Code:
{question}

Generate the review in the following structured format:

## Code Review Summary
**Language Detected:** {language}

### Issue [N]:
- **Issue Type:**
- **Severity:**
- **Description:**
- **Recommended Fix:**
- **Corrected Code:**
```
(corrected code here)
```
- **Source:**

### Overall Recommendations:
-
"""
)


# ---------------------------------------------------------------------------
# 3. LOAD VECTOR DATABASE
# ---------------------------------------------------------------------------
def load_vector_db(vector_db_path: str = "../vector_db"):
    # 1. Load the same embedding model used during preprocessing
    # 2. Use it to load the saved FAISS index from disk
    # 3. Return the database object ready for searching
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    vector_db = FAISS.load_local(
        vector_db_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_db


def load_retriever(vector_db_path: str = "../vector_db", k: int = 3):
    """
    Convenience wrapper: load FAISS and return it as a retriever directly.
    Kept for backward compatibility with earlier code.
    """
    vector_db = load_vector_db(vector_db_path)
    return vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )



# RETRIEVE WITH RELEVANCE SCORES
# 1. Convert user's code into a vector using the embedding model
# 2. FAISS searches 309 chunks and finds top 3 most similar
# 3. Returns those 3 chunks + their similarity distances
# 4. Converts distance to 0-100% relevance score
#    (smaller distance = more relevant = higher score)

def retrieve_with_scores(vector_db, query: str, k: int = 3):
    
    results = vector_db.similarity_search_with_score(query, k=k)

    docs = [doc for doc, score in results]
    distances = [score for doc, score in results]

    # Convert L2 distance to a 0-100 relevance score.
    # Smaller distance -> higher relevance.
    # We use a simple inverse mapping: relevance = 100 / (1 + distance)
    # This is a heuristic transformation, not a calibrated probability.
    relevance_scores = [100 / (1 + d) for d in distances]
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

    return docs, round(avg_relevance, 1)



# BUILD FINAL PROMPT (RETRIEVE + FORMAT)

def build_review_prompt_with_score(vector_db, user_code: str, k: int = 3):
    
    #Combines language detection + retrieval + relevance scoring + prompt building in one call and returns
    #final_prompt : str = Ready to send to Groq. Includes language-specific review guidelines.
    #relevance_score : float = 0-100 heuristic relevance score (see retrieve_with_scores docstring).
    #detected_language : str = The detected programming language (for display in the UI).
    
    detected_language = detect_language(user_code)
    language_guidelines = LANGUAGE_GUIDELINES.get(
        detected_language, LANGUAGE_GUIDELINES["Unknown"]
    )

    docs, relevance_score = retrieve_with_scores(vector_db, user_code, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = REVIEW_PROMPT.format(
        context=context,
        language=detected_language,
        language_guidelines=language_guidelines,
        question=user_code
    )

    return final_prompt, relevance_score, detected_language


def build_review_prompt(retriever, user_code: str) -> str:

    #Original simple function kept for backward compatibility.
    #Detects language and builds a language-aware prompt using a retriever.
    
    detected_language = detect_language(user_code)
    language_guidelines = LANGUAGE_GUIDELINES.get(
        detected_language, LANGUAGE_GUIDELINES["Unknown"]
    )

    retrieved_docs = retriever.invoke(user_code)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    final_prompt = REVIEW_PROMPT.format(
        context=context,
        language=detected_language,
        language_guidelines=language_guidelines,
        question=user_code
    )

    return final_prompt