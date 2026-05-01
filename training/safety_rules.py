def emergency_keyword_score(text: str) -> int:
    text = text.lower()

    emergency_keywords = [
        "chest pain",
        "shortness of breath",
        "difficulty breathing",
        "trouble breathing",
        "vomiting blood",
        "blood in stool",
        "fainting",
        "seizure",
        "confusion",
        "unconscious",
        "high fever",
        "jaundice"
    ]

    score = 0
    for keyword in emergency_keywords:
        if keyword in text:
            score += 1

    return score