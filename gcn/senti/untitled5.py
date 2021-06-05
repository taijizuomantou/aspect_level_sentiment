#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:36:37 2020

@author: xue
#"""
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'/home/xue/stanford-corenlp-full-2018-10-05')
a =['transitlocation','location', '-', '1', ',', '[UNK]', 'it', 'is', 'a', 'beautiful', 'late', '-', 'victorian', '/', '[UNK]', 'suburb', 'of', 'south', '-', 'east', 'london']
char_tokens = " ".join(a)
pos_tag = nlp.pos_tag(char_tokens)
print(char_tokens)
k =  nlp.word_tokenize(char_tokens)
print(k)
print(len(k))
i = 0
for pos in pos_tag:
    i += 1
#    print(pos[1])
#print(i)
def tokenize(self, text,tag,head):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer.

        Returns:
          A list of wordpiece tokens.
        """


        #output_tags.append(0)
        output_tokens = []
        for i,token in enumerate((text)):
            
            chars = list(token)
           # print(token)
          #  doc = nlp(token)

           # print(pos_tag)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
          #  is_bad = False
            start = 0
            sub_tags = []
            sub_tokens = []
            sub_head = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    
                    if substr in self.vocab:
                        cur_substr = substr
                    if start > 0:
                        substr = "##" + substr
                    end -= 1
                if cur_substr is None:
                    break
                sub_tags.append(tag)
                #print(tag[i])
                sub_tokens.append(cur_substr)
                sub_head.append(head)
                start = end

                output_tokens.append(sub_tokens)
            
        assert(len(output_tokens) == len(sub_tags))
        
        return output_tokens,sub_tags
temp = WordpieceTokenizer("uncased_L-12_H-768_A-12/vocab.txt")
ans,as2 = temp.tokenize("price","efw","efwf")