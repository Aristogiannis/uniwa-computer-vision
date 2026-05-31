"""Caption templates for the disaster categories.

Kept in its own module so that consumers (CLI scripts, generation pipeline)
that only need text prompts do not pay the cost of importing PyTorch via
``dataset.py``.
"""

from __future__ import annotations

DISASTER_PROMPT_TEMPLATES: dict[str, list[str]] = {
    "pre_disaster": [
        "a satellite image of an intact landscape, high resolution remote sensing",
        "a clean overhead satellite photo of an undamaged area",
    ],
    "flood": [
        "a satellite image of severe flooding, brown water covering buildings and roads, post-disaster remote sensing",
        "an aerial overhead view of a flooded urban area after a major storm",
    ],
    "wildfire": [
        "a satellite image of an active wildfire, smoke plumes and burnt vegetation, post-disaster remote sensing",
        "an aerial overhead view of a forest after a wildfire, blackened ground and scorched trees",
    ],
    "post_disaster": [
        "a satellite image of disaster aftermath, damaged buildings and disturbed ground, remote sensing",
    ],
    # EuroSAT land-cover classes — used as a proxy when xBD is not available.
    "forest": [
        "a satellite image of a dense forest, green canopy, overhead remote sensing photograph",
        "an aerial overhead view of a deciduous forest, continuous green tree cover",
    ],
    "river": [
        "a satellite image of a river winding through land, water and vegetation, overhead remote sensing photograph",
        "an aerial overhead view of a river with curved banks and surrounding terrain",
    ],
    "residential_buildings": [
        "a satellite image of a residential neighbourhood, rows of houses and gardens, overhead remote sensing photograph",
        "an aerial overhead view of a suburban residential area with streets and rooftops",
    ],
}


def build_text_prompt(category: str, *, index: int = 0) -> str:
    """Return a deterministic caption for ``category``.

    A deterministic mapping makes evaluation reproducible — the same image
    always lands with the same caption. We rotate through a small bank of
    templates by ``index`` so the model sees light prompt variety.
    """

    key = category.lower()
    templates = DISASTER_PROMPT_TEMPLATES.get(key)
    if templates is None:
        return f"a satellite image of a {key} scene, remote sensing photograph"
    return templates[index % len(templates)]
