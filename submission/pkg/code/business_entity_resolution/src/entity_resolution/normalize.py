"""Text normalization for business names and addresses.

All functions are pure and country-agnostic: every rule is applied to every Record,
so an unseen country (France in test) goes through the same path.
"""
import re

from anyascii import anyascii

# Legal-form and filler words, including romanized Indic spellings (anyascii drops vowels).
NAME_DROP = set(
    """
    pvt private privet praivet praibhet piraivet praivett prayivet pravet prayvet
    ltd limited limitet limitedd limitad limited. llp elelpi elelpee llc inc incorporated
    corp corporation co company cie sarl sas sasu eurl sa sci snc selarl plc lp pllc ltda gmbh
    the and et of formerly dba aka
    """.split()
)

ADDRESS_WORDS = {
    "street": "st", "str": "st", "saint": "st",
    "road": "rd",
    "avenue": "ave", "av": "ave",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "bvd": "blvd",
    "highway": "hwy",
    "parkway": "pkwy",
    "suite": "ste",
    "apartment": "apt",
    "circle": "cir",
    "terrace": "ter",
    "r": "rue",
    "chem": "chemin",
    "imp": "impasse",
    "all": "allee",
    "rte": "route",
    "fbg": "faubourg",
    "number": "no", "nr": "no",
    "near": "nr_",
    "opp": "opposite", "opposite": "opposite",
    "keralam": "kl", "orissa": "od", "ts": "tg",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}

INDIA_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "telangana": "tg", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "west bengal": "wb", "delhi": "dl", "jammu and kashmir": "jk", "chandigarh": "ch",
    "puducherry": "py",
}

# State names in Indic scripts, as they appear in the Mentions.
SCRIPT_STATES = {
    "महाराष्ट्र": "mh", "दिल्ली": "dl", "उत्तर प्रदेश": "up", "ಕರ್ನಾಟಕ": "ka", "தமிழ்நாடு": "tn",
    "পশ্চিমবঙ্গ": "wb", "ગુજરાત": "gj", "తెలంగాణ": "tg", "हरियाणा": "hr", "राजस्थान": "rj",
    "കേരളം": "kl", "बिहार": "br", "मध्य प्रदेश": "mp", "ఆంధ్రప్రదేశ్": "ap", "ਪੰਜਾਬ": "pb",
    "ଓଡ଼ିଶା": "od", "झारखंड": "jh", "छत्तीसगढ़": "cg", "असम": "as", "অসম": "as", "उत्तराखंड": "uk",
    "गोवा": "ga", "हिमाचल प्रदेश": "hp",
}

_STATE_PHRASES = {**US_STATES, **INDIA_STATES}
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _STATE_PHRASES), key=len, reverse=True)) + r")\b")
_SCRIPT_RE = re.compile("|".join(sorted(map(re.escape, SCRIPT_STATES), key=len, reverse=True)))
_WEB_RE = re.compile(r"^(https?://)?(www\.)?|\.(com|net|org|in|co\.in|fr|co|biz|info)$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LEADING_ZEROS = re.compile(r"(?<![0-9a-z])0+(?=[0-9])")


def _ascii_lower(text):
    return anyascii(text or "").lower()


def normalize_name(text):
    """Core business name: ASCII, lowercase, no punctuation, legal forms and fillers removed."""
    s = _WEB_RE.sub(" ", _ascii_lower(text).strip())
    tokens = _NON_ALNUM.sub(" ", s).split()
    return " ".join(t for t in tokens if t not in NAME_DROP)


# Legal forms in one canonical spelling (romanized Indic spellings included), for `name_key`.
LEGAL_CANON = {
    **dict.fromkeys("pvt private privet praivet praibhet piraivet praivett prayivet pravet prayvet".split(), "pvt"),
    **dict.fromkeys("ltd limited limitet limitedd limitad".split(), "ltd"),
    **dict.fromkeys("llp elelpi elelpee".split(), "llp"),
    "inc": "inc", "incorporated": "inc", "llc": "llc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co",
}
_KEY_DROP = set("the and et of formerly dba aka mr dr shri smt sri".split())


def name_key(text):
    """Order-free name identity that keeps legal forms and extra words: "Up Chemicals Public Limited" and
    "Limited Up Chemicals Public" share a key; "... Inc" and "... LLC" do not."""
    s = _WEB_RE.sub(" ", _ascii_lower(text).replace(".", "").strip())
    tokens = {LEGAL_CANON.get(t, t) for t in _NON_ALNUM.sub(" ", s).split() if t not in _KEY_DROP}
    return " ".join(sorted(tokens))


_SKELETON_MAP =str.maketrans({"b": "p", "d": "t", "g": "k", "c": "k", "q": "k", "z": "j", "w": "v", "x": "k"})
_VOWELS = re.compile(r"[aeiouyh]")
_REPEATS = re.compile(r"(.)\1+")


def name_skeleton(text):
    """Consonant skeleton of the core name, so romanized Indic script lines up with Latin spellings."""
    out = []
    for token in normalize_name(text).split():
        token = token.replace("x", "ks").translate(_SKELETON_MAP)
        token = _REPEATS.sub(r"\1", _VOWELS.sub("", token))
        if token:
            out.append(token)
    return " ".join(out)


def normalize_address(text):
    """Address as one canonical token string: ASCII, lowercase, abbreviations and state names unified."""
    s = _SCRIPT_RE.sub(lambda m: " " + SCRIPT_STATES[m.group(0)] + " ", text or "")
    s = _NON_ALNUM.sub(" ", _ascii_lower(s))
    s = _STATE_RE.sub(lambda m: _STATE_PHRASES[m.group(0)], s)
    return " ".join(ADDRESS_WORDS.get(t, t) for t in _LEADING_ZEROS.sub("", s).split())


# Numbers that belong to a unit, suite or mail box, not to the building ("unit b 47", "po box 7616").
UNIT_NUMBER_RE = r"\b(unit|apt|apartment|ste|pmb|box|fl|floor|room|rm|bldg|flat)( [a-z]+)? \d+\b"
HOUSE_NUMBER_RE = r"\b(\d+)\b"
_UNIT_NUMBER = re.compile(UNIT_NUMBER_RE)
_HOUSE_NUMBER = re.compile(HOUSE_NUMBER_RE)


def house_number(addr_n):
    """First street number of a normalized address, ignoring unit and box numbers (None if absent)."""
    m = _HOUSE_NUMBER.search(_UNIT_NUMBER.sub(" ", addr_n or ""))
    return int(m.group(1)) if m else None


# Countries whose state-level unit is written as a multi-word name: French regions, and the départements
# that Mentions use in their place ("Nord" for Hauts-de-France, "Gironde" for Nouvelle-Aquitaine).
REGION_PHRASES = {
    "France": {
        "hauts de france": "hdf", "nord": "hdf", "pas de calais": "hdf",
        "nouvelle aquitaine": "naq", "gironde": "naq",
        "pays de la loire": "pdl", "loire atlantique": "pdl",
    },
}
_REGION_RE = {c: re.compile(r"\b(" + "|".join(sorted(map(re.escape, p), key=len, reverse=True)) + r")\b")
              for c, p in REGION_PHRASES.items()}

STATE_CODES = {"US": set(US_STATES.values()), "India": set(INDIA_STATES.values()),
               **{c: set(p.values()) for c, p in REGION_PHRASES.items()}}


def address_state(addr_n, country):
    """State code found in a normalized address, or "" (always "" for a country without a lexicon)."""
    if country in _REGION_RE:
        found = [REGION_PHRASES[country][m] for m in _REGION_RE[country].findall(addr_n)]
        return found[-1] if found else ""
    codes = STATE_CODES.get(country)
    if not codes:
        return ""
    found = [t for t in addr_n.split() if t in codes]
    return found[-1] if found else ""
