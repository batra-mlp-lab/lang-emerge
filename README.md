# Language Emergence

Code for the paper

**[Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog][1]**  
[Satwik Kottur][3], [José M. F. Moura][5], [Stefan Lee][4], [Dhruv Batra][6]  
[Arxiv][1]  


This repository contains code to **train**, **evaluate** and **visualize**
dialogs between conversational agents (Abot and QBot), which talk about
instances in an abstract world.  

If you find this code useful, consider citing our work:
**replace with actual citation**

```
@inproceedings{visdial,
  title = {{N}atural {L}anguage {D}oes {N}ot {E}merge '{N}aturally' in {M}ulti-{A}gent {D}ialog},
  author = {Satwik Kottur and Jos\'e M.F. Moura and Stefan Lee and Dhruv Batra},
  journal = {CoRR},
  volume = {abs/1706.08502},
  year = {2017}
}
```

## Setup

All our code is implemented in [Pytorch][2].

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

Pretrained models coming soon!
Detailed documentation coming soon!

## Contributors

* [Satwik Kottur][3]

## License

BSD


[1]: https://arxiv.org/abs/1706.08502
[2]: http://pytorch.org/
[3]: https://satwikkottur.github.io
[4]: https://computing.ece.vt.edu/~steflee/
[5]: http://users.ece.cmu.edu/~moura/
[6]: https://computing.ece.vt.edu/~batra/
