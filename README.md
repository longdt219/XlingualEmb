# Speech Translation/Alignment
This is the implementation of our EMNLP 2016 paper titled : 
[Learning CrosslingualWord Embeddings without Bilingual Corpora] (https://arxiv.org/abs/1606.09403)

If you use  this code, please cite the paper 

```
@InProceedings{duong-EtAl:2016:EMNLP,
  author    = {Duong, Long  and  Kanayama, Hiroshi  and  Ma, Tengfei  and  Bird, Steven  and  Cohn, Trevor},
  title     = {Learning Crosslingual Word Embeddings without Bilingual Corpora},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016)},
  month     = {November},
  year      = {2016},
  address   = {Texas, USA},
  publisher = {Association for Computational Linguistics},
}
```
### Getting started
The implementation is basically the extension of C version [Word2Vec] (https://code.google.com/archive/p/word2vec/). You just need to do the `make`

### How to run  
We included the extracted dictionaries from [Panlex] (http://panlex.org/) for several languages including (German, Dutch, Spanish, Italian, Greek, Finish, Japanse, Serbian) in folder `/data/dicts`. We also included a tiny mixed English-Italian monolingual data `/data/mono/en_it.shuf.10k`
for demo purposes. The full monolingual data can be downloaded from [Polyglot website] (https://sites.google.com/site/rmyeid/projects/polyglot).

Note that both dictionary and monolingual data are pre-processed with 
- lowercased 
- adding language prefix 

The following will build the crosslingual word embeddings for English and Italian. 
```
./xlingemb -train data/mono/en_it.shuf.10k -output en.it.word.emb -size 200 -window 48 -iter 15 
-negative 25 -sample 0.0001 -alpha 0.025 -cbow 1 -threads 5 -dict data/dicts/en.it.panlex.all.processed 
-outputn en.it.context.emb -reg 0.01
```
Some options :
- train : the training file which is the combination of English and Italian monolingual data. 
- output: the usual word embedding output file which is for reference purpose only.  
- <b>outputn</b> : the context word embedding file which is the <b> final output </b>. 
- size, window, iter, negative, sample, alpha, cbow, threads : the same as Word2Vec
- <b> dict </b>: the dictionary 
- <b> reg </b> : the regulariser sensitivity for combining word and context embeddings. 

