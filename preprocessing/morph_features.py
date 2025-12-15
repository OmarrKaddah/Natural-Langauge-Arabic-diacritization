from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
def extract_word_morphology(sentence):
    

    db = MorphologyDB.builtin_db()
    analyzer = Analyzer(db)

    words = sentence.split()
    word_feats = []

    for word in words:
        analyses = analyzer.analyze(word)
        # Take first analysis (or implement disambiguation)
        if analyses:
            feats = analyses[0]
            word_feats.append({
                'pos': feats.get('pos', 'UNK'),
                'gender': feats.get('gen', 'UNK'),
                'number': feats.get('num', 'UNK'),
                'aspect': feats.get('asp', 'UNK')
            })
        else:
            word_feats.append({
                'pos': 'UNK',
                'gender': 'UNK',
                'number': 'UNK',
                'aspect': 'UNK'
            })

    return words, word_feats
