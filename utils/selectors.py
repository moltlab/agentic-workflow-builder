def tokenize(text: str) -> set[str]:
    return {t for t in ''.join(c if c.isalnum() else ' ' for c in text.lower()).split() if t}

def rank_files_by_query(filenames: list[str], query: str) -> list[str]:
    """Simple relevance ranking by token overlap with special-casing common intents."""
    q_tokens = tokenize(query.lower())
    if not filenames:
        return []

    # Calculate scores for each filename based on token overlap
    scored_files: list[tuple[int, str]] = []
    for name in filenames:
        name_tokens = tokenize(name)
        score = len(name_tokens & q_tokens)
        if score > 0:
            scored_files.append((score, name))

    sorted_files = sorted(scored_files, key=lambda item: (-item[0], item[1]))

    # Extract just the filenames and return the top 3 relevant files
    ranked_filenames = [name for score, name in sorted_files[:3]]
    return ranked_filenames
