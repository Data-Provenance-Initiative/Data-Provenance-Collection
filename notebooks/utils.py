def count_unique_with_none(s):
    all_vals = []
    for lst in s:
        if lst is None:
            pass
        else:
            all_vals += lst
    
    return len(set(all_vals))