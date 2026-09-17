import os
import math
import json
import hashlib
from datetime import datetime
from typing import List
from pydantic import BaseModel
import streamlit as st


from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types

load_dotenv()


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


CHAT_MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"


def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [
            x.strip()
            for x in f.read().split("\n\n")
            if x.strip()
        ]


def extract_text(file):
    reader = PdfReader(file)


    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )




def chunk_text(text, size=150, overlap=30):
    words = text.split()
    chunks = []


    step = size - overlap


    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])


        if chunk:
            chunks.append(chunk)


    return chunks


def create_embeddings(texts, task_type):

    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768
        )
    )


    return [
        x.values
        for x in result.embeddings
    ]


def similarity(a, b):


    dot = sum(x * y for x, y in zip(a, b))


    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))


    if mag_a == 0 or mag_b == 0:
        return 0


    return dot / (mag_a * mag_b)


CACHE_FILE = "knowledge_cache.json"


def _file_hash(path):
    """Return MD5 hash of a file's contents."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def load_knowledge():
    """Load knowledge base embeddings, using a disk cache when possible.

    The cache (knowledge_cache.json) is invalidated automatically whenever
    resume_guidelines.txt or jokes.txt changes (detected via MD5 hash).
    """

    guidelines_path = "resume_guidelines.txt"
    jokes_path = "jokes.txt"

    current_hashes = {
        "guidelines": _file_hash(guidelines_path),
        "jokes": _file_hash(jokes_path),
    }

    # --- Try loading from disk cache ---
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)

        if cached.get("hashes") == current_hashes:
            return (
                cached["guidelines"],
                cached["guideline_vectors"],
                cached["jokes"],
                cached["joke_vectors"],
            )

    # --- Cache miss: compute embeddings and save ---
    guidelines = read_file(guidelines_path)
    jokes = read_file(jokes_path)

    guideline_vectors = create_embeddings(guidelines, "RETRIEVAL_DOCUMENT")
    joke_vectors = create_embeddings(jokes, "RETRIEVAL_DOCUMENT")

    cache_data = {
        "hashes": current_hashes,
        "guidelines": guidelines,
        "guideline_vectors": guideline_vectors,
        "jokes": jokes,
        "joke_vectors": joke_vectors,
    }
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f)

    return (
        guidelines,
        guideline_vectors,
        jokes,
        joke_vectors
    )



def retrieve(query_vectors, texts, vectors, top_k=3):
    scores = []
    for query in query_vectors:
        for text, vector in zip(texts, vectors):
            score = similarity(query, vector)
            scores.append((text, score))


    # Highest similarity first
    scores.sort(key=lambda x: x[1], reverse=True)


    # Remove duplicates
    results = []
    for text, score in scores:
        if text not in results:
            results.append(text)
        if len(results) == top_k:
            break


    return results


class BeforeAfter(BaseModel):
    before: str
    after: str


class RoastResponse(BaseModel):
    roasts: List[str]
    biggest_problem: str
    score: int
    good: List[str]
    fix: List[str]
    before_after: List[BeforeAfter]
    verdict: str
    roast_level: int

def generate_roast(resume, guidelines, jokes):
    guidelines_text = "\n".join(f"- {x}" for x in guidelines)
    jokes_text = "\n".join(f"- {x}" for x in jokes)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    system_instruction = f"""You are a funny Hinglish resume reviewer.
Current Date and Time Context: {current_datetime}


Roast the RESUME, not the person.


Tone:
- Hinglish
- Sarcastic
- Clever
- Short
- College placement humor


Do not imitate any real comedian.
Use the joke references only as inspiration. Never copy them.
Use the guidelines to identify genuine resume problems.
Never invent facts."""


    prompt = f"""================ RESUME ================
{resume}


============= GUIDELINES ===============
{guidelines_text}


============= JOKE REFERENCES ===========
{jokes_text}


================ TASK ===================
Analyze the resume and return valid JSON following the schema.


Rules:
- Exactly 4 roasts.
- Exactly 3 good points.
- Exactly 3 improvements.
- Exactly 2 before/after examples.
- score must be between 0 and 100.
- roast_level must be between 1 and 10.
- Keep every item short and punchy.
- Return valid JSON only."""


    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.9,
            max_output_tokens=4000,
            response_mime_type="application/json",
            response_schema=RoastResponse,
        ),
    )


    raw_text = response.text.strip()


    # Clean markdown json tags if present
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]


    return json.loads(raw_text.strip())


st.set_page_config(
    page_title="Resume Roast AI",
    page_icon="🔥",
)


st.title("🔥 Resume Roast AI")
st.write("Upload your resume. Let AI judge your career decisions.")


resume_file = st.file_uploader("Upload Resume", type=["pdf"])




# ============================================================
# ROAST
# ============================================================


if st.button("🔥 Roast My Resume", type="primary"):
    if not resume_file:
        st.warning("Please upload your resume.")
        st.stop()


    # 1. PDF → TEXT
    with st.spinner("📖 Reading your resume..."):
        resume = extract_text(resume_file)
        resume = " ".join(resume.split())


    if not resume:
        st.error("Could not read text from this PDF.")
        st.stop()


    # 2. CHUNK RESUME
    chunks = chunk_text(resume)
    st.caption(f"Resume → {len(chunks)} chunks")


    # 3. CREATE RESUME EMBEDDINGS
    with st.spinner("🧠 Understanding your resume..."):
        chunk_vectors = create_embeddings(chunks, "RETRIEVAL_QUERY")


    # 4. LOAD KNOWLEDGE (served from disk cache if unchanged)
    with st.spinner("📚 Loading resume knowledge..."):
        (
            guidelines,
            guideline_vectors,
            jokes,
            joke_vectors,
        ) = load_knowledge()


    # 5. RETRIEVAL
    with st.spinner("🔎 Finding relevant advice and jokes..."):
        relevant_guidelines = retrieve(
            chunk_vectors, guidelines, guideline_vectors, top_k=4
        )
        relevant_jokes = retrieve(
            chunk_vectors, jokes, joke_vectors, top_k=3
        )


    # SHOW RETRIEVED CONTEXT
    with st.expander("🔎 See what RAG retrieved"):
        st.write("### Resume Guidelines")
        for item in relevant_guidelines:
            st.write("•", item)


        st.write("### Joke References")
        for item in relevant_jokes:
            st.write("•", item)


    # 6. GENERATION
    with st.spinner("🔥 Preparing your roast..."):
        answer = generate_roast(
            resume, relevant_guidelines, relevant_jokes
        )


    # 7. RESULT
    st.divider()


    st.header("🔥 ROAST")
    for i, roast in enumerate(answer["roasts"], 1):
        st.write(f"**{i}.** {roast}")


    st.header("💀 BIGGEST CRIME")
    st.write(answer["biggest_problem"])


    st.header("💯 SCORE")
    st.metric("Resume Score", f"{answer['score']}/100")


    st.header("🟢 GOOD THINGS")
    for item in answer["good"]:
        st.write("✅", item)


    st.header("🔴 FIX THESE")
    for item in answer["fix"]:
        st.write("❌", item)


    st.header("🚀 BEFORE → AFTER")
    for item in answer["before_after"]:
        st.markdown(f"**Before:** {item['before']}")
        st.markdown(f"**After:** {item['after']}")
        st.divider()


    st.header("🎤 FINAL VERDICT")
    st.write(answer["verdict"])
    st.caption(f"🔥 Roast Level: {answer['roast_level']}/10")


    # 8. EXPLAIN RAG
    st.divider()
    st.subheader("🧠 How RAG worked")
    st.markdown(
        """
```text
Resume PDF
    ↓
Extract Text
    ↓
Chunking
    ↓
Embeddings
    ↓
Similarity Search
    ↓
Retrieve Guidelines + Jokes
    ↓
Add Context to Prompt
    ↓
Gemini
    ↓
🔥 Roast


R — Retrieval
Find relevant resume guidelines and joke references.


A — Augmentation
Add the retrieved information to Gemini's prompt.


G — Generation
Gemini generates the final roast.
```
"""
    )
