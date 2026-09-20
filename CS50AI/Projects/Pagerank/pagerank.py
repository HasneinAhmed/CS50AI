import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML files and return a dictionary mapping each page
    to a set of all other pages in the corpus that it links to.
    """
    pages = dict()
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]?)href=\"([^\"])\"", contents)
            pages[filename] = set(links) - {filename}
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )
    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    """
    n = len(corpus)
    links = corpus[page]

    # If the page has no outgoing links, return a uniform distribution over all pages
    if not links:
        return {p: 1.0 / n for p in corpus}

    # Otherwise, distribute non-damping probability uniformly, and damping probability over outgoing links
    base_prob = (1.0 - damping_factor) / n
    link_prob = damping_factor / len(links)

    return {p: base_prob + (link_prob if p in links else 0.0) for p in corpus}


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling n pages
    according to transition model, starting with a random page.
    """
    counts = {page: 0 for page in corpus}
    current_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        counts[current_page] += 1
        model = transition_model(corpus, current_page, damping_factor)
        pages, probabilities = zip(*model.items())
        current_page = random.choices(pages, weights=probabilities, k=1)[0]

    return {page: count / n for page, count in counts.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    """
    n = len(corpus)
    pagerank = {page: 1.0 / n for page in corpus}

    while True:
        new_pagerank = {}
        for page in corpus:
            total_sum = 0.0
            for possible_linker, outgoing_links in corpus.items():
                # If possible_linker links to page, or has no links at all (treat as linking to all)
                if page in outgoing_links:
                    total_sum += pagerank[possible_linker] / len(outgoing_links)
                elif not outgoing_links:
                    total_sum += pagerank[possible_linker] / n

            new_pagerank[page] = ((1.0 - damping_factor) / n) + (damping_factor * total_sum)

        # Check convergence threshold (0.001) across all pages
        if all(abs(new_pagerank[p] - pagerank[p]) < 0.001 for p in corpus):
            break

        pagerank = new_pagerank

    return pagerank


if __name__ == "_main_":
    main()
