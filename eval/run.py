"""Golden-question measurement of the chat retrieval path.

Each golden item pairs a question with hand-picked reference urls from the
committed snapshot. The script reports how often those urls surface in the
top results and how well the ordering agrees — numbers to compare before
and after touching the ranking.

Usage: python -m eval.run
"""

import json
from pathlib import Path

import chat

GOLDEN = Path(__file__).resolve().parent / "golden.json"


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def hit_at_k(system: list[str], reference: list[str], k: int) -> bool:
    """Did any reference url make the top k — the coarsest useful signal."""
    return any(url in reference for url in system[:k])


def order_agreement(system: list[str], reference: list[str]) -> float | None:
    """Pairwise concordance over the urls both lists share: 1.0 for identical
    order, 0.0 for fully reversed. Under 2 shared urls there is no order to
    measure, hence None."""
    common = [url for url in system if url in reference]
    common = list(dict.fromkeys(common))  # a duplicate url would pair with itself as discordant
    if len(common) < 2:
        return None
    rank = {url: pos for pos, url in enumerate(reference)}
    concordant = total = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            total += 1
            if rank[common[i]] < rank[common[j]]:
                concordant += 1
    return concordant / total


def story_url(story_id: str, url: str) -> str:
    """Golden sources live in url space, so system rows must map into it the
    same way the chat renders links: stories without a url resolve to their
    item page, or they could never match a reference."""
    if isinstance(url, str) and url:  # missing urls arrive as NaN from read_csv
        return url
    return f"https://news.ycombinator.com/item?id={story_id.removeprefix('s:')}"


def run_question(question: str) -> list[str]:
    """The exact retrieval path the chat tab takes, minus the LLM."""
    hits = chat.retrieve(question)
    if hits.empty:
        return []
    hit_ids = [int(t) for t in hits["id"]]
    stories = chat.top_stories(hit_ids, question)  # default limit, as the tab calls it
    return [story_url(s.id, s.url) for s in stories.itertuples()]


def main() -> None:
    results = []
    for item in load_golden():
        system = run_question(item["question"])
        reference = item["sources"]
        results.append(
            (
                item["id"],
                hit_at_k(system, reference, 1),
                hit_at_k(system, reference, 3),
                hit_at_k(system, reference, 8),
                sum(1 for url in system if url in reference),
                order_agreement(system, reference),
            )
        )
    print(f"{'id':<6}{'hit@1':>7}{'hit@3':>7}{'hit@8':>7}{'common':>8}{'agree':>7}")
    for qid, h1, h3, h8, common, agree in results:
        agree_text = "n/a" if agree is None else f"{agree:.2f}"
        print(f"{qid:<6}{int(h1):>7}{int(h3):>7}{int(h8):>7}{common:>8}{agree_text:>7}")
    n = len(results)
    agrees = [r[5] for r in results if r[5] is not None]
    agree_mean = f"{sum(agrees) / len(agrees):.2f} (over {len(agrees)})" if agrees else "n/a"
    print(
        f"\nmeans: hit@1 {sum(r[1] for r in results) / n:.2f}"
        f", hit@3 {sum(r[2] for r in results) / n:.2f}"
        f", hit@8 {sum(r[3] for r in results) / n:.2f}"
        f", agreement {agree_mean}"
    )


if __name__ == "__main__":
    main()  # exit code stays 0 whatever the scores; nothing here is meant to fail CI
