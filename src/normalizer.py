import requests

def brand_to_generic(drug_name: str) -> str:
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}&search=1"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        rxcui = data["idGroup"]["rxnormId"][0]

        # Get the generic ingredient using the related endpoint
        related_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json?tty=IN"
        related = requests.get(related_url, timeout=5).json()
        concepts = related["relatedGroup"]["conceptGroup"]
        for group in concepts:
            if "conceptProperties" in group:
                return group["conceptProperties"][0]["name"].lower()

        # Fallback: return the name from properties
        props_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json"
        props = requests.get(props_url, timeout=5).json()
        return props["properties"]["name"].lower()
    except Exception:
        return drug_name.lower()