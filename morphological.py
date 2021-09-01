# -*- coding: utf-8 -*-
'''
Created on 2010/07/21

@author: iori_o
'''
import urllib
import urllib2
from BeautifulSoup import BeautifulSoup

appid = 'Yahoo!デベロッパーズネットワークのアプリケーションIDを入力して下さい'
pageurl = "http://jlp.yahooapis.jp/MAService/V1/parse"

# Yahoo!形態素解析の結果をリストで返す
def split(sentence, appid=appid, results="ma", filter="1|2|3|4|5|9|10"):
    sentence = sentence.encode("utf-8")
    params = urllib.urlencode({'appid':appid, 'results':results, 'filter':filter, 'sentence':sentence})
    results = urllib2.urlopen(pageurl, params)
    soup = BeautifulSoup(results.read())

    return [w.surface.string for w in soup.ma_result.word_list]


