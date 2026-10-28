import re

def extract_answer(task, text):
    """Return normalized answer or None. Uses the last 'Answer:' line; falls back to task-specific rules."""
    if not text:
        return None
    m = re.findall(r"answer\s*:\s*(.+)", text, flags=re.I)
    raw = m[-1].strip() if m else None
    if task == "formal_fallacies":
        src = raw if raw else text[-200:]
        s = src.lower()
        if re.search(r"\binvalid\b", s): return "invalid"
        if re.search(r"\bvalid\b", s): return "valid"
        return None
    if task == "logical_deduction":
        if raw:
            mm = re.search(r"\(?([A-G])\)?", raw.upper())
            return mm.group(1) if mm else None
        mm = re.findall(r"\(([A-G])\)", text.upper())
        return mm[-1] if mm else None
    if task == "word_sorting":
        if not raw: return None
        return " ".join(raw.strip().strip(".").split()).lower()
    raise ValueError(task)

def normalize_target(task, target):
    t = target.strip()
    if task == "logical_deduction":
        return re.sub(r"[()]", "", t).upper()
    if task == "word_sorting":
        return " ".join(t.split()).lower()
    return t.lower()

def is_correct(task, response, target):
    a = extract_answer(task, response)
    return a is not None and a == normalize_target(task, target)
