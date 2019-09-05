import numpy as np
import difflib
import heapq

s = "导游（见习）"
w = "见习导游"
def string_similar(s1, s2):
    s2_low = s2.lower()
    score1 = difflib.SequenceMatcher(None, s1, s2).quick_ratio()
    score2 = difflib.SequenceMatcher(None, s1, s2_low).quick_ratio()
    val_max = max(score1,score2)

    return val_max


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
        wl.append((item, score))
    return heapq.nlargest(limit, wl, key=lambda i: i[1]) if limit is not None else sorted(wl, key=lambda i: i[1], reverse=True)

def digit_distance_search(target, candidates,limit =10):
    target = abs(float(target))
    if target ==0:
        target = target+0.01
    candlist = list(candidates.keys())

    wl = []
    wls=[]
    score = 0
    for item in candlist:
        try:
            float(item)
            wl.append(item)
        except ValueError:
            pass

    if len(wl) >1:
        for i in range(len(wl) -2):
            for j in range(i+1,len(wl)-1):
                try:
                    if candidates[wl[i]][0] == candidates[wl[j]][0]:
                        wls.append(wl[i])
                except:
                    print('keyerror',candidates[wl[i]][0],candidates[wl[j]][0])
                    pass
    wl = [x for x in wl if x not in wls]
    wls.clear()
    wlt=[]
    for item in wl :
        if float(item) !=0:
            score = min(target,abs(float(item)))/max(target,abs(float(item)))
        else:
            score = (float(item) +0.01)/target
        wlt.append((score*100,item))
    return heapq.nlargest(limit, wlt, key=lambda i: i[1]) if limit is not None else sorted(wlt, key=lambda i: i[1], reverse=True)