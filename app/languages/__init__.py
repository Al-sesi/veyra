"""Per-language vocabulary packs for the rule-based parser and intent matcher.

Each pack (`ha.py`, `ig.py`, `yo.py`) holds native-speaker-confirmed phrases
for one language, stored diacritic-folded and lowercased at import time, so
matching code can compare them directly against folded ASR text.

Matching policy: the English/Pidgin baseline living in app.parser and
app.intents is ALWAYS active, and the Hausa/Igbo/Yoruba packs are merged in
unconditionally too. Code-switching inside a single voice note
("Na sayi rice 5k" — Hausa verb, English item) is the normal case, so
vocabulary is never gated on one selected language.
"""

from app.languages.ha import HAUSA_PACK
from app.languages.ig import IGBO_PACK
from app.languages.yo import YORUBA_PACK

PACKS = {
    "ha": HAUSA_PACK,
    "ig": IGBO_PACK,
    "yo": YORUBA_PACK,
}
