"""Persona generation and prompt building for Bell test experiments."""

import random
from typing import Dict, Any, List, Optional

DEFAULT_LOCATIONS = {
    "bloomington, in": "en",
    "detroit, mi": "en",
    "chicago, il": "en",
    "austin, tx": "en",
}

OCCUPATIONS = [
    "teacher", "doctor", "engineer", "artist", "writer", "chef", "musician",
    "farmer", "accountant", "nurse", "programmer", "mechanic", "designer",
    "lawyer", "student", "retiree", "entrepreneur", "scientist", "driver",
    "retail worker", "police officer", "firefighter", "architect", "journalist",
]

HOBBIES = [
    "reading", "gardening", "hiking", "painting", "cycling", "cooking",
    "photography", "gaming", "fishing", "knitting", "woodworking", "dancing",
    "swimming", "yoga", "bird watching", "collecting stamps", "playing guitar",
    "watching movies", "traveling",
]


def generate_location(locations: Optional[Dict[str, str]] = None) -> str:
    """Pick a random location from the location map."""
    loc_map = locations or DEFAULT_LOCATIONS
    return random.choice(list(loc_map.keys()))


def generate_age(min_age: int = 25, max_age: int = 70) -> int:
    """Generate a random age within the given range."""
    return random.randint(min_age, max_age)


def generate_occupation() -> str:
    """Pick a random occupation."""
    return random.choice(OCCUPATIONS)


def generate_hobby() -> str:
    """Pick a random hobby."""
    return random.choice(HOBBIES)


def generate_persona(
    location: Optional[str] = None,
    locations: Optional[Dict[str, str]] = None,
    include_occupation: bool = True,
    include_hobby: bool = True,
    min_age: int = 25,
    max_age: int = 70,
) -> Dict[str, Any]:
    """Generate a random persona dictionary.

    Parameters
    ----------
    location : str, optional
        Force a specific location. If None, one is picked randomly.
    locations : dict, optional
        Location-to-language mapping. Defaults to DEFAULT_LOCATIONS.
    include_occupation : bool
        Whether to include an occupation field.
    include_hobby : bool
        Whether to include a hobby field.
    min_age, max_age : int
        Age range.

    Returns
    -------
    dict with keys: location, language, age, and optionally occupation, hobby.
    """
    loc_map = locations or DEFAULT_LOCATIONS
    if location is None:
        location = random.choice(list(loc_map.keys()))
    language = loc_map.get(location, "en")

    persona = {
        "location": location,
        "language": language,
        "age": generate_age(min_age, max_age),
    }
    if include_occupation:
        persona["occupation"] = generate_occupation()
    if include_hobby:
        persona["hobby"] = generate_hobby()
    return persona


def get_persona_prompt(persona: Dict[str, Any]) -> str:
    """Build a system prompt string from a persona dictionary."""
    return f"You are a {persona['age']}-year-old from {persona['location']}."


def create_personas_pool(
    count: int = 100,
    locations: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Create a pool of diverse personas.

    Ensures at least one persona from each location, then fills
    up to ``count`` with random personas.
    """
    loc_map = locations or DEFAULT_LOCATIONS
    personas = [generate_persona(location=loc, locations=loc_map) for loc in loc_map]
    while len(personas) < count:
        personas.append(generate_persona(locations=loc_map))
    return personas
