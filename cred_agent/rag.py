"""Two-strategy local RAG with deterministic MOCK_LLM retrieval and optional Chroma."""
from __future__ import annotations
import math, re, hashlib
import os
from pathlib import Path
from collections import Counter
from functools import lru_cache

KB_DIR = Path(__file__).resolve().parents[1] / "knowledge_base"
THRESHOLD = 0.4358  # midpoint from current mock probes: (0.2124 + 0.6591) / 2
TOP_K = 3
_VECTOR_THRESHOLD: float | None = None

def documents() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(KB_DIR.glob("*.md"))}

def fixed_chunks(docs: dict[str,str], size: int = 500, overlap: int = 80) -> list[dict]:
    out=[]
    for parent, text in docs.items():
        body="\n".join(text.splitlines()[1:]).strip()
        for start in range(0, max(len(body),1), size-overlap):
            out.append({"parent":parent,"text":body[start:start+size]})
            if start+size>=len(body): break
    return out

def sentence_chunks(docs: dict[str,str], group_size: int = 3) -> list[dict]:
    out=[]
    for parent,text in docs.items():
        body=" ".join(text.splitlines()[1:]).strip()
        sentences=[x.strip() for x in re.split(r"(?<=[.!?])\s+",body) if x.strip()]
        for i in range(0,len(sentences),group_size): out.append({"parent":parent,"text":" ".join(sentences[i:i+group_size])})
    return out

_STOP={"what","are","is","the","a","an","of","for","to","and","how","do","does","can","i","my","it","in","on","by","with","be","from","or","who","when","which","this","that","will","you","your"}
def _tokens(s: str) -> list[str]: return [t for t in re.findall(r"[a-z0-9]+", s.casefold()) if t not in _STOP]
def _vec(s: str) -> Counter:
    # Stable hashed token vectors avoid randomness/hash-seed dependency in mock mode.
    v=Counter()
    for token in _tokens(s): v[int(hashlib.sha1(token.encode()).hexdigest()[:8],16)%4096]+=1
    return v
def cosine(a: str,b: str) -> float:
    x,y=_vec(a),_vec(b); dot=sum(v*y.get(k,0) for k,v in x.items())
    nx=math.sqrt(sum(v*v for v in x.values())); ny=math.sqrt(sum(v*v for v in y.values()))
    return dot/(nx*ny) if nx and ny else 0.0

def similarity(query: str, text: str) -> float:
    """Blend cosine with query-term coverage so concise natural questions match policy text."""
    q=set(_tokens(query)); d=set(_tokens(text))
    coverage=len(q & d)/len(q) if q else 0.0
    return .5*cosine(query,text)+.5*coverage

def retrieve(query: str, strategy: str = "sentence", top_k: int = TOP_K) -> list[dict]:
    if os.getenv("USE_VECTOR_STORE","0")=="1":
        return vector_retrieve(query,strategy,top_k)
    chunks=sentence_chunks(documents()) if strategy=="sentence" else fixed_chunks(documents())
    ranked=sorted(({**c,"score":similarity(query,c["text"])} for c in chunks),key=lambda x:x["score"],reverse=True)
    return ranked[:top_k]

def vector_retrieve(query: str, strategy: str="sentence", top_k: int=TOP_K) -> list[dict]:
    model,client=_vector_resources()
    name="sentence_based" if strategy=="sentence" else "fixed_window"
    collection=client.get_collection(name)
    vector=model.encode([query],normalize_embeddings=True).tolist()[0]
    raw=collection.query(query_embeddings=[vector],n_results=top_k,include=["documents","metadatas","distances"])
    hits=[]
    for doc,meta,distance in zip(raw["documents"][0],raw["metadatas"][0],raw["distances"][0]):
        hits.append({"text":doc,"parent":meta["parent"],"score":round(max(0.0,min(1.0,1-distance)),6)})
    return hits

@lru_cache(maxsize=1)
def _vector_resources():
    import chromadb
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",local_files_only=True)
    client=chromadb.PersistentClient(path=str(Path(__file__).resolve().parents[1]/"data"/"chroma"))
    return model,client

def active_threshold() -> float:
    """Use a threshold measured from the active retrieval backend's own score clusters."""
    global _VECTOR_THRESHOLD
    if os.getenv("USE_VECTOR_STORE","0")!="1": return THRESHOLD
    if _VECTOR_THRESHOLD is None:
        measured=calibration()
        if not measured["separated"]:
            raise RuntimeError("Vector retrieval calibration did not separate in-scope and out-of-scope probes")
        _VECTOR_THRESHOLD=measured["threshold"]
    return _VECTOR_THRESHOLD

def invalidate_vector_calibration() -> None:
    global _VECTOR_THRESHOLD
    _VECTOR_THRESHOLD=None

def grounded_answer(query: str, strategy: str = "sentence") -> dict:
    hits=retrieve(query,strategy)
    score=hits[0]["score"] if hits else 0
    if score < active_threshold(): return {"answer":"I don't know based on the available policy documents. Please ask an authorized support representative.","sources":[],"top1_similarity":round(score,4),"abstained":True}
    parents=[]
    for h in hits:
        if h["parent"] not in parents: parents.append(h["parent"])
    # A concise answer grounded directly in the retrieved source text.
    return {"answer":hits[0]["text"],"sources":parents,"top1_similarity":round(score,4),"abstained":False,"context":"\n".join(h["text"] for h in hits)}

def calibration() -> dict:
    in_scope=["KYC identity address verification required documents PAN income evidence","reducing balance EMI principal monthly interest rate instalments calculation","credit score repayment history utilization account age recent enquiries","fraud dispute investigation transaction evidence complaint reference"]
    out_scope=["weather forecast tomorrow rain temperature","cricket match score batsman tournament result","photosynthesis plants chlorophyll sunlight biology"]
    positives=[max(x["score"] for x in retrieve(q,"sentence",1)) for q in in_scope]
    negatives=[max(x["score"] for x in retrieve(q,"sentence",1)) for q in out_scope]
    threshold=(max(negatives)+min(positives))/2
    return {"in_scope_top1": [round(x,4) for x in positives],"out_of_scope_top1":[round(x,4) for x in negatives],"threshold":round(threshold,4),"separated":max(negatives)<min(positives)}

def compare(query_labels: list[tuple[str,set[str]]], strategy: str) -> list[dict]:
    rows=[]
    for q,relevant in query_labels:
        hits=retrieve(q,strategy,5); got=list(dict.fromkeys(h["parent"] for h in hits))
        true=len(set(got)&relevant); precision=true/len(got) if got else 0; recall=true/len(relevant) if relevant else 0
        rows.append({"query":q,"relevant":sorted(relevant),"retrieved":got,"arithmetic":f"P={true}/{len(got)}={precision:.3f}; R={true}/{len(relevant)}={recall:.3f}","precision":round(precision,3),"recall":round(recall,3)})
    return rows

def build_chroma_indexes():
    """Create distinct persisted Chroma collections with local ST embeddings."""
    model,client=_vector_resources()
    for name,chunks in (("fixed_window",fixed_chunks(documents())),("sentence_based",sentence_chunks(documents()))):
        collection=client.get_or_create_collection(name,metadata={"hnsw:space":"cosine"})
        if chunks:
            vectors=model.encode([x["text"] for x in chunks],normalize_embeddings=True).tolist()
            collection.upsert(ids=[f"{x['parent']}:{i}" for i,x in enumerate(chunks)],documents=[x["text"] for x in chunks],embeddings=vectors,metadatas=[{"parent":x["parent"]} for x in chunks])
    invalidate_vector_calibration()
    return ["fixed_window","sentence_based"]
