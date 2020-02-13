import numpy as np
import pandas as pd
import preprocessor as p


def clean_text(t):
    p.set_options(
        p.OPT.URL,
        p.OPT.MENTION,
        p.OPT.HASHTAG,
        p.OPT.RESERVED,
        p.OPT.EMOJI,
        p.OPT.SMILEY,
    )
    t = p.clean(t)
    t = re.sub(r"[\\//_,;.:*+-=><)($|~&%^`'\"\[\]]+", '', t)
    t = re.sub(r'[^\x00-\x7F]+',' ', t)
    return t