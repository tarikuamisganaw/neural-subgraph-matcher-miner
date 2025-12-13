"""
Utility functions
"""
import re


def extract_first_sentences(text: str, n: int = 4) -> str:
    """
    Extract first n sentences from text
    """
    if not text:
        return ""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(parts[:n]).strip()


def format_large_number(num):
    """Format large numbers with commas"""
    try:
        return f"{int(num):,}"
    except (ValueError, TypeError):
        return str(num)