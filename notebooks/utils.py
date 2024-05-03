def count_unique_with_none(s):
    all_vals = []
    for lst in s:
        if lst is None:
            pass
        else:
            all_vals += lst
    
    return len(set(all_vals))


def color_license_classes(s):
    ret = []

    if 'All' in s:
        ret += [r'\CommercialDataCircle']
    else:
        ret += [r'\TransparentCircle']
    
    if 'Unspecified' in s or len(s) == 0:
        ret += [r'\UnspecifiedDataCircle']
    else:
        ret += [r'\TransparentCircle']
    
    if 'Acad' in s or 'NC' in s:
        ret += [r'\NCDataCircle']
    else:
        ret += [r'\TransparentCircle']
    
    return ' '.join(ret)


FORMATS_MAP = {
    'Zero-shot': 'ZS',
    'zero-shot': 'ZS',
    'Few-shot': 'FS',

    'Single-turn Dialog': 'SD',
    'Multi-turn Dialog': 'MD',
    
    'Chain-of-Thought': 'CT',
    'Program of Thoughts': 'PT',
    'Program-of-Thought': 'PT',
    
    'Response Ranking': 'RR',
    'Evaluation': 'EV',
}