from preprocessing.diacritics import strip_diacritics


def post_process(tokens):
    """
    tokens: list of predicted words like ["الشَّهَادَةُ", "ظَاهِرَةٌ", ...]
    returns: corrected tokens
    """

    prepositions = {"من","في","إلى","الى","عن","على","ب","ك","ل"}
    kana_family = {"كان","ليس","صار","أصبح","أمسى","ظل","بات","مازال","مازال"}
    
    result = tokens.copy()

    for i, word in enumerate(tokens):

        bare = strip_diacritics(word)

        # RULE 1 — PREPOSITIONS
        if bare in prepositions and i + 1 < len(tokens):
            next_word = tokens[i+1]
            # replace final diacritic with kasra
            result[i+1] = force_case(next_word, "genitive")

        # RULE 2 — INNA (إنّ)
        if bare == "إن" and "ّ" in word:   # detect إنَّ
            if i + 1 < len(tokens):
                result[i+1] = force_case(result[i+1], "accusative")

        # RULE 3 — Kana and sisters
        if bare in kana_family:
            if i + 1 < len(tokens):
                result[i+1] = force_case(result[i+1], "nominative")

        # RULE 4 — pronouns:
        if word.endswith("هُ") or word.endswith("هِ"):
            if i > 0 and strip_diacritics(tokens[i-1]) in prepositions:
                result[i] = force_pronoun(result[i], "kasra")
            else:
                result[i] = force_pronoun(result[i], "damma")

    return result


def force_case(word, case_type):
    """
    Replace last diacritic of word based on grammatical case.
    """
    base = strip_diacritics(word)
    last = base[-1]

    if case_type == "genitive":
        return base + "ِ"
    if case_type == "accusative":
        return base + "َ"
    if case_type == "nominative":
        return base + "ُ"
    
    return word


def force_pronoun(word, mode):
    base = strip_diacritics(word)
    if not base.endswith("ه"):
        return word
    
    if mode == "kasra":
        return base + "هِ"
    if mode == "damma":
        return base + "هُ"
    
    return word
