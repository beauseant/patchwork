from gensim import corpora

def file_lines(fname):
    """
    Count number of lines in file

    Parameters
    ----------
    fname: Path
        the file whose number of lines is calculated

    Returns
    -------
    number of lines
    """
    with fname.open('r', encoding='utf8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1

##########################################
# Carry out specific preprocessing steps #
##########################################
def tkz_clean_str(rawtext, stopwords, equivalents):
    if not rawtext:
        return ''
    
    # Lowercasing and tokenization
    cleantext = rawtext.lower().split()

    # Remove stopwords and apply equivalences in one pass
    cleantext = [equivalents.get(word, word) for word in cleantext if word not in stopwords]

    # Second stopword removal (in case equivalences introduced new stopwords)
    cleantext = [word for word in cleantext if word not in stopwords]
    
    return ' '.join(cleantext)

##########################################
# Carry out specific preprocessing steps #
##########################################
def tkz_clean_str(rawtext, stopwords, equivalents):
    if not rawtext:
        return ''
    
    # Lowercasing and tokenization
    cleantext = rawtext.lower().split()

    # Remove stopwords and apply equivalences in one pass
    cleantext = [equivalents.get(word, word) for word in cleantext if word not in stopwords]

    # Second stopword removal (in case equivalences introduced new stopwords)
    cleantext = [word for word in cleantext if word not in stopwords]
    
    return ' '.join(cleantext)

def preprocBOW(data_col, min_lemas=15, no_below=10, no_above=0.6, keep_n=100000):

    # filter out documents (rows) with less than minimum number of lemmas
    data_col = data_col[data_col.apply(lambda x: len(x.split())) >= min_lemas]
    
    final_tokens = [doc.split() for doc in data_col.values.tolist()]
    gensimDict = corpora.Dictionary(final_tokens)

    # Remove words that appear in less than no_below documents, or in more than no_above, and keep at most keep_n most frequent terms
    gensimDict.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    
    # Remove words not in dictionary, and return a string
    vocabulary = set([gensimDict[idx] for idx in range(len(gensimDict))])
    
    return vocabulary