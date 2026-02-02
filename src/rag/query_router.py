def route_query(query: str):
    q = query.lower()

    if any(word in q for word in ["violate", "non-compliant", "fire", "ei"]):
        return ["mismatches"]

    if any(word in q for word in ["explain", "overview", "project", "system"]):
        return ["documents", "rules"]

    if any(word in q for word in ["regulation", "code", "standard"]):
        return ["regulations"]

    if any(word in q for word in ["product", "solution", "manufacturer"]):
        return ["products"]

    # fallback: search everything
    return ["mismatches", "rules", "documents", "regulations", "products"]
