def flat_dict(_dict):
    keys = list(_dict.keys())

    if len(keys) > 0:
        k = keys[0]
        v = _dict[k]
        del _dict[k]

        if isinstance(v, dict):
            r = {f"{k}__{sk}": sv for sk, sv in v.items()}

        else:
            r = {k: v}

        r.update(flat_dict(_dict))
        return r
    else:
        return {}
