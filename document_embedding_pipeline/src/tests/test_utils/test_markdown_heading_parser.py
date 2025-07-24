from utils.markdown_heading_parser import MarkdownHeadingParser

def test_single_h1():
    content = "# Title\nContent"
    sections = MarkdownHeadingParser.parse(content)
    assert len(sections) == 1
    assert sections[0].title == "Title"
    assert sections[0].subsections == []

def test_h1_h2():
    content = "# Main\n## Sub1\nContent\n## Sub2\nMore content"
    sections = MarkdownHeadingParser.parse(content)
    assert len(sections) == 1
    assert sections[0].title == "Main"
    assert len(sections[0].subsections) == 2
    assert sections[0].subsections[0].title == "Sub1"
    assert sections[0].subsections[1].title == "Sub2"

def test_orphaned_h2():
    content = "## Orphan\nContent"
    sections = MarkdownHeadingParser.parse(content)
    assert len(sections) == 1
    assert sections[0].title == "Orphan"
    assert sections[0].subsections is None

def test_mixed_headings():
    content = "# H1\n## H2a\n### Ignored\n## H2b\n# Another H1"
    sections = MarkdownHeadingParser.parse(content)
    assert len(sections) == 2
    assert sections[0].title == "H1"
    assert len(sections[0].subsections) == 2
    assert sections[0].subsections[0].title == "H2a"
    assert sections[0].subsections[1].title == "H2b"
    assert sections[1].title == "Another H1"