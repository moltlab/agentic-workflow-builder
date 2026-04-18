def extract_markdown_points(markdown_text: str, section_header: str) -> list[str]:
    """Extract bullet points under a '## {section_header}' in markdown.

    Stops at the next '## ' header. Returns a list of point strings without leading dashes.
    """
    lines = markdown_text.splitlines()
    in_section = False
    points: list[str] = []
    for line in lines:
        if line.strip().startswith("## "):
            in_section = line.strip().lower() == f"## {section_header}".lower()
            continue
        if in_section:
            stripd = line.strip()
            if stripd.startswith("- ") or stripd.startswith("* "):
                points.append(stripd[2:].strip())
            elif stripd.startswith("1.") or stripd.startswith("2."):
                points.append(stripd.split('.', 1)[1].strip())
            elif stripd.startswith("## "):
                break
    return points
