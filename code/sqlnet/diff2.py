import numpy as np
import difflib
import heapq

s = "导游（见习）"
w = "见习导游"
def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


#target_str, list(candidates.keys()),
def search_abbr(w, s, ngram =10):
    sl = []
    wl = []
    for idx in range(len(s)):
        if w.startswith(s[idx]):
            i = 0
            while i<ngram:
                if idx+i>len(s):
                    break
                word = s[idx:idx+i]
                i+=1
                sc = string_similar(word,w)
                wl.append(word)
                sl.append(sc)
    return wl[np.argmax(sl)]

def extact_sort(target,candlist,limit =10):
    wl = []

    for item in candlist:
        score = string_similar(target, item) * 100
        wl.append((item,score))
    return heapq.nlargest(limit, wl, key=lambda i: i[1]) if limit is not None else sorted(wl, key=lambda i: i[1], reverse=True)

