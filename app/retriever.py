"""Dependency-free BM25 retriever. Small enough to read in one sitting."""
import math
import re
from collections import Counter

STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "at", "by", "for",
    "and", "or", "if", "it", "that", "this", "with", "as", "up", "do", "does", "did", "can", "may",
    "i", "me", "my", "we", "our", "you", "your", "what", "how", "many", "much", "which", "who",
    "when", "there", "than", "then", "from", "have", "has", "get", "gets", "must", "will", "should",
}


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "ly", "s"):
        if w.endswith(suf) and not w.endswith("ss") and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class Retriever:
    def __init__(self, docs: list[dict], filter_deprecated: bool = False, k1=1.5, b=0.75):
        self.docs = [d for d in docs if not (filter_deprecated and d["status"] == "deprecated")]
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d["title"] + " " + d["text"]) for d in self.docs]
        self.avg_len = sum(len(t) for t in self.tokens) / max(len(self.tokens), 1)
        df = Counter(w for toks in self.tokens for w in set(toks))
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query: str, k: int) -> list[dict]:
        q = tokenize(query)
        scored = []
        for doc, toks in zip(self.docs, self.tokens):
            tf = Counter(toks)
            score = 0.0
            for w in q:
                if w in tf:
                    denom = tf[w] + self.k1 * (1 - self.b + self.b * len(toks) / self.avg_len)
                    score += self.idf[w] * tf[w] * (self.k1 + 1) / denom
            if score > 0:
                scored.append({"id": doc["id"], "score": round(score, 3),
                               "status": doc["status"], "text": doc["text"]})
        scored.sort(key=lambda x: -x["score"])
        return scored[:k]
