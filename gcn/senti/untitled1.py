from stanfordcorenlp import StanfordCoreNLP
#import en_core_web_sm
#nlp = en_core_web_sm.load()
nlp = StanfordCoreNLP(r'/home/xue/stanford-corenlp-full-2018-10-05')
a = ['my', 'friend', 'got', 'mugged', 'on', 'the', 'kings', 'road', 'in', 'location', '-', '1', 'at', '8pm', 'at', 'night', ',', 'on', 'a', 'summers', 'evening', 'on', 'a', 'friday', 'night', 'will', 'lots', 'of', 'people', 'on', 'the', 'streets']
print(len(a))
temp = "my friend got mugged on the kings road in location - 1 at 8pm at night , on a summers evening on a friday night will lots of people on the streets"
ans = temp.split()
print(len(ans))
docs = nlp.dependency_parse(temp)
print(docs)
toke = nlp.word_tokenize(temp)
print(toke)
print(a)