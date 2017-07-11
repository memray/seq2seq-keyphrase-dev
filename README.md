# seq2seq-keyphrase
**Note: This is a development branch. Please check out this [[repo]](https://github.com/memray/seq2seq-keyphrase) for the release version of seq2seq-keyphrase.**

Introduction
==========
This is an implementation of Deep Keyphrase Generation [[PDF]](http://memray.me/uploads/acl17-keyphrase-generation.pdf) [[arXiv]](https://arxiv.org/abs/1704.06879).


Data
==========
The KE20k dataset is released in JSON format. Please download [here](http://crystal.exp.sis.pitt.edu:8080/iris/data/ke20k.zip). Each data point contains the title, abstract and keywords of a paper.

Part | #(data) 
--- | --- 
Training | 530,803 
Validation | 20,000
Test | 20,000

The raw dataset (without filtering noisy data) is also provided. Please download [here](http://crystal.exp.sis.pitt.edu:8080/iris/data/all_title_abstract_keyword_clean.json.zip). 

Well-trained model and other datasets will be released soon.

Cite
==========

If you use the code or datasets, please cite the following paper:

> Rui Meng, Sanqiang Zhao, Shuguang Han, Daqing He, Peter Brusilovsky and Yu Chi. Deep Keyphrase Generation. 55th Annual Meeting of Association for Computational Linguistics. [[PDF]](http://memray.me/uploads/acl17-keyphrase-generation.pdf) [[arXiv]](https://arxiv.org/abs/1704.06879)
