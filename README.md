# Higher-order Coreference Resolution with Coarse-to-fine Inference: PyTorch Implementation

Pytorch implementation of the c2f-coref model proposed by [Kenton Lee](http://kentonl.com/), 
[Luheng He](https://homes.cs.washington.edu/~luheng), and 
[Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz) in their paper 
__[Higher-order Coreference Resolution with Coarse-to-fine Inference](https://arxiv.org/abs/1804.05392)__.
 
This implementation is based upon the [original implementation](https://github.com/kentonl/e2e-coref) by the 
 authors above. The __model__ package is written from scratch, whereas the scripts in both packages __eval__ and 
__setup__ are borrowed with almost no changes from the original implementation. Optimization for mention pruning 
inspired by [Ethan Yang](https://github.com/YangXuanyue/pytorch-e2e-coref). 


## Requirements
This project was written with Python 3.8.5 and PyTorch 1.7.1. For installation details regarding PyTorch please visit the 
official [website](https://pytorch.org/). Further requirements are listed in the __requirements.txt__ and can be 
installed via pip: <code>pip install -r requirements.txt</code> 


## Setup
> __Hint:__ Run setup.sh in an environment with Python 2.7 so the CoNLL-2012 scripts are executed by the correct interpreter 

First of all download the [pretrained model](https://drive.google.com/file/d/1fkifqZzdzsOEo0DXMzCFjiNXqsKG_cHi) of the
original implementation which contains the glove data for the head word embeddings. Unpack it and place the 
__glove_50_300_2.txt__ file in the __data/embs__ folder. (The other files are not needed an can be deleted)

To obtain all necessary data for training and evaluation run __setup.sh__ with the path to the 
[OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) folder (often named ontonotes-release-5.0).

e.g. <code>$ ./setup.sh /path/to/ontonotes-release-5.0</code>

Run __setup/cache_elmo.py__ afterwards to build up the ELMo embeddings for training, validation and test data. The cache 
files need about 20 GB of disk space.



## Training
Run the training with <code>python train.py -c \<config\> -f \<folder\> --cpu --amp</code>. Select a configuration 
from [coref.conf](coref.conf) with the optional parameter <code>config</code>. The parameter <code>folder</code> 
names the folder the snapshots, taken during the training, are saved to. If the given folder already exists and contains 
at least one snapshot the training is restarted loading the latest snapshot. The optional flags <code>cpu</code> and 
<code>amp</code> can be set to train exclusively on the CPU or to use the automatic mixed precision training.


## Evaluation
Run the evaluation with <code>python evaluate.py -c \<conf\> -p \<pattern\> --cpu --amp</code>. All snapshots in the 
__data/ckpt__ folder that match the given <code>pattern</code> are evaluated. This works with simple snapshots (pt) as 
well as with snapshots with additional metadata (pt.tar). See [Training](#Training) for details on the remaining 
parameters.