from typing import List
from pypinyin import Style, lazy_pinyin

PUNCTUARIONS = set([
    '，', '。', '？', '！', '：', '；', '、', '、'
])


def length_match(src: List[str], res: List[str]) -> float:
    """[UNK] will be treated as one character.
    Characters are seperated by space.
    """
    assert len(src) == len(res)
    match = 0
    for s, r in zip(src, res):
        s = s.replace('[UNK]', 'U').replace(' ', '')
        r = r.replace('[UNK]', 'U').replace(' ', '')
        if len(s) == len(r):
            match += 1
    return match / len(src)

def is_character_match(src: str, res: str) -> bool:
    """Whether charater at the coresponding position
     in the couplet is different (punctuation is also considered).
    """
    if len(src) != len(res):
        return False
    for s, r in zip(src, res):
        if s in PUNCTUARIONS and r not in PUNCTUARIONS:
            return False
        if s not in PUNCTUARIONS and r in PUNCTUARIONS:
            return False
        if s not in PUNCTUARIONS and s == r:
            return False
    return True


def character_match(src: List[str], res: List[str]) -> float:
    """[UNK] will be treated as one character"""
    assert len(src) == len(res)
    match = 0
    for s, r in zip(src, res):
        s = s.replace('[UNK]', 'U').replace(' ', '')
        r = r.replace('[UNK]', 'U').replace(' ', '')
        match += is_character_match(s, r)
    return match / len(src)

def get_tone(char):
    tone = lazy_pinyin(char, style=Style.TONE3)
    if len(tone) == 0:
        tone = ''
    else:
        tone = tone[0][-1]
    if tone in ['1', '2']:
        return 0
    elif tone in ['3', '4']:
        return 1
    else:
        return 2

    
def tone_match(src: List[str], res: List[str], strict: bool = True) -> float:
    """If ```strict``` is ```True```, only those antecedent clause with 3rd or 4th
    tone will be evaluated.
    If ```strict``` is ```False```, all clauses will be evaluated as correct as long as
    the tone of the antecedent clause and the subsequent clause is opposed."""
    assert len(src) == len(res)
    match = 0
    for s, r in zip(src, res):
        s = s.replace('[UNK]', 'U').replace(' ', '')
        r = r.replace('[UNK]', 'U').replace(' ', '')
        if s == '':
            s_tone = get_tone(s)
        else:
            s_tone = get_tone(s[-1])
        if r == '':
            r_tone = get_tone(r)
        else:
            r_tone = get_tone(r[-1])
        #s_tone = get_tone(s[-1])
        #r_tone = get_tone(r[-1])
        if s_tone != r_tone and r_tone != 2:
            match += 1
    return match / len(src)

def compute_format_metrics(src: List[str], res: List[str]) -> None:
    print(f'Length Matched {length_match(src, res):.4f}')
    print(f'Character Matched {character_match(src, res):.4f}')
    print(f'Tone Matched (strict) {tone_match(src, res):.4f}')
    print(f'Tone Matched (not strict) {tone_match(src, res, False):.4f}')