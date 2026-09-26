import re
import jellyfish
from rapidfuzz import fuzz
from src.normalize import normalize


def jaccard(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def numeric_tokens(s):
    return set(re.findall(r'\b\d+\b', s))


def first_token(s):
    tokens = s.split()
    return tokens[0] if tokens else ''


def featurize_pair(s1_row, cand_row):
    """Given two pandas Series, return a dict of numeric features."""
    na = normalize(s1_row['business_name'])
    nb = normalize(cand_row['business_name'])
    aa = normalize(s1_row['business_address'])
    ab = normalize(cand_row['business_address'])

    f = {
        'name_ratio':     fuzz.ratio(na, nb) / 100,
        'name_token_set': fuzz.token_set_ratio(na, nb) / 100,
        'name_partial':   fuzz.partial_ratio(na, nb) / 100,
        'name_jaccard':   jaccard(na, nb),
        'addr_ratio':     fuzz.ratio(aa, ab) / 100,
        'addr_token_set': fuzz.token_set_ratio(aa, ab) / 100,
        'addr_partial':   fuzz.partial_ratio(aa, ab) / 100,
        'addr_jaccard':   jaccard(aa, ab),
        'same_country':   int(s1_row['country'] == cand_row['country']),
        'first_token_match': int(first_token(na) == first_token(nb)),
        'name_length_ratio': max(0.5, min(2.0, len(na) / len(nb))) if nb else 0.5,
        'address_length_ratio': max(0.5, min(2.0, len(aa) / len(ab))) if ab else 0.5,
    }

    # pincode
    za = set(re.findall(r'\b\d{5,6}\b', aa))
    zb = set(re.findall(r'\b\d{5,6}\b', ab))
    f['same_zip'] = int(bool(za & zb)) if (za or zb) else 0

    # numeric token match
    nums_a = numeric_tokens(aa)
    nums_b = numeric_tokens(ab)
    f['numeric_token_match'] = int(bool(nums_a & nums_b))

    # phonetic
    f['name_phonetic'] = int(
        jellyfish.metaphone(na.split()[0] if na.split() else '') ==
        jellyfish.metaphone(nb.split()[0] if nb.split() else '')
    )

    return f
