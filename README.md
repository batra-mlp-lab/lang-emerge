# Language Emergence

Code for the paper

**[Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog][1]**  
[Satwik Kottur][3], [José M. F. Moura][5], [Stefan Lee][4], [Dhruv Batra][6]  
[Arxiv][1]  


This repository contains code to **train**, **evaluate**, and **visualize**
dialogs between conversational agents (Abot and QBot) that talk about
instances in an abstract world.  

If you find this code useful, consider citing our work ([ACL Anthology](https://www.aclweb.org/anthology/D17-1321/)):

```
@inproceedings{kottur-etal-2017-natural,
    title = "Natural Language Does Not Emerge {`}Naturally{'} in Multi-Agent Dialog",
    author = "Kottur, Satwik  and
      Moura, Jos{\'e}  and
      Lee, Stefan  and
      Batra, Dhruv",
    booktitle = "Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing",
    month = Sep,
    year = "2017",
    address = "Copenhagen, Denmark",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D17-1321",
    doi = "10.18653/v1/D17-1321",
    pages = "2962--2967",
}
```

## Setup

All our code is implemented in [PyTorch][2]. Current version has been tested in Python 3.6 and PyTorch 1.4.

Additionally, our code also uses some famous python packages that can be installed as follows:

```
pip install json
pip install tqdm
pip install pickle
pip install json
```

## Contents

* `options.py` - Read the options from the commandline
* `dataloader.py` - Create and handle data for toy instances
* `chatbots.py` - Conversational agents - Abot and Qbot
* `learnChart.py` - Obtain evolution of language chart from checkpoints
* `html.py` - Easy creation of html tables
* `utilities.py` -  Helper functions
* `train.py` - Script to train conversational agents
* `test.py` - Script to test agents

## Usage
Checkout `run_me.sh` to see how train our model.

Pretrained models and detailed documentation coming soon!

## Contributors

* [Satwik Kottur][3]

## License

BSD-3


[1]: https://arxiv.org/abs/1706.08502
[2]: http://pytorch.org/
[3]: https://satwikkottur.github.io
[4]: https://computing.ece.vt.edu/~steflee/
[5]: http://users.ece.cmu.edu/~moura/
[6]: https://computing.ece.vt.edu/~batra/
