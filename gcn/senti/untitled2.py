import spacy
import nltk

nlp = spacy.load("en_core_web_sm")
doc = nlp("for those that go once and don ' t enjoy it , all i can say is that they just don ' t get it . ")
spacy_text = []
for item in doc:
    spacy_text.append(item.text)
print(spacy_text)
#for item in doc:
from spacy.pipeline import DependencyParser
#    print(item.text)
#from nltk.parse.dependencygraph import DependencyGraph, conll_data2
#def whitespace_tokenize(text):
#    """Runs basic whitespace cleaning and splitting on a peice of text."""
#    text = text.strip()
#    if not text:
#        return []
#    tokens = text.split()
#    return tokens
for token in doc:
    print(token.head.i)
for token in doc:
    print("{2}({3}-{6}, {0}-{5})".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_, token.i+1, token.head.i+1))
#a = "I shot an elephant in my sleep  ###aa . ."
#
##a= nltk.parse.dependencygraph.DependencyGraph('John N 2\n'
## 'loves V 0\n'
##  'Mary N 2')
##print(a.to_dot())
###dep_parser = nltk.parse.corenlp.CoreNLPDependencyParser(url='http://localhost:9000')
###parse, = dep_parser.raw_parse(
###'The quick brown fox jumps over the lazy dog.' )
#nltk.parse()
#tokens_a = nextnlp.word_tokenize(a)
#print(tokens_a)
##print(temp)
#sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
##grammar = "NP: {<DT>?<JJ>*<NN>}"
##cp = nltk.RegexpParser()
##result = cp.parse(sentence)
#parser = nlp.create_pipe("parser")
#
## Construction from class
#
#parser = DependencyParser(nlp.vocab)
#doc = nlp("This is a sentence.")
## This usually happens under the hood
#processed = parser(doc)
#from spacy.tokens import Doc
#a= "the food is uniformly exceptional , with a very capable kitchen which will proudly whip up whatever you feel like eating , whether it \' s on the menu or not . ##a"
#words = a.split()
#print(words)
#spaces = len(words) * [True]
#doc = Doc(nlp.vocab, words=words,spaces = spaces)
#doc = Doc(nlp.vocab, orths_and_spaces=[(u'Some', True), (u'text', True)])
##doc = nlp(a)
#vocab = []
#for word in words:
#    vocab.append(word)
#ans = nlp(doc)
#print(ans)
from spacy.gold import align

bert_tokens = "for those that go once and don ' t enjoy it , all i can say is that they just don ' t get it . ".split()
print(bert_tokens)
spacy_tokens =spacy_text
print(spacy_text)
alignment = align(bert_tokens, spacy_tokens)
cost, a2b, b2a, a2b_multi, b2a_multi = alignment
print(a2b)
print(b2a)
print(b2a_multi)
temphead = len(bert_tokens) * [-1]
k = 0;
for item in doc:
    temphead[b2a[k]] = b2a[item.head.i]
    k += 1
print(temphead)
head = []
head.append(-1)
for item in temphead:
    head.append(item + 1)
print(head)