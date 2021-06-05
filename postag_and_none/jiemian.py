#import nltk
#from nltk.tree import Tree
#from nltk.tokenize import word_tokenize
#from nltk.tag import pos_tag
#from stanfordcorenlp import StanfordCoreNLP
#nlp = StanfordCoreNLP(r'/home/xue/stanford-corenlp-full-2018-10-05')
#doc = "He derives great joy and happiness from cycling"
#temp = doc
#doc = nltk.word_tokenize(doc)
#doc = nltk.pos_tag(doc)
#grammar = "NP: {<DT>?<JJ>*<NN>}"
#cp = nltk.RegexpParser(grammar)
#print(doc)
#tag = nlp.pos_tag(temp)
#print(tag)
#ans = nlp.dependency_parse(temp)
#print(ans)
#print(nlp.parse(temp))
#Tree.fromstring(nlp.parse(temp)).draw()
from spacy import displacy
import en_core_web_sm
nlp = en_core_web_sm.load()

doc = nlp('I just bought 2 shares at 9 a.m. because the stock went up 30% in just 2 days according to the WSJ')
for item in doc:
    print("("+str(item) + "," + str(item.head)+")")
#displacy.serve(doc, style='dep')
