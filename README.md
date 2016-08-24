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
#### Getting started
The implementation is basically the extension of C version Word2Vec. You just need to do the make 
    make

### How to run  
We included the extracted dictionaries from Panlex for several languages including (German, Dutch, Spanish, Italian, Greek, Finish, Japanse, Serbian) in folder 
    /data/dicts 
We also included a tiny mixed English-Italian monolingual data for demo purposes. 
   /data/mono/en_it.shuf.10k
The following will build the crosslingual word embeddings for English and Italian. 
```
./xlingemb -train data/mono/en_it.shuf.10k -output en.it.word.emb -size 200 -window 48 -iter 15 -negative 25 -sample 0.0001 -alpha 0.025 -cbow 1 -threads 5 -dict data/dicts/en.it.panlex.all.processed -outputn en.it.context.emb -reg 0.01
```
Some options :
- output: the usual word embeddings 
- outputn : the context word embeddings which is the final output.
- size, window, iter, negative, sample, alpha, cbow, threads : the same as Word2Vec
- dict: the dictionary 
- reg: the regulariser sensitivity for combining word and context embeddings. 

