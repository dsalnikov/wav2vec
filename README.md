# wav2vec 

This is a minimal [wav2vec 2.0](https://arxiv.org/abs/2006.11477) implementation in plain [NumPy](https://numpy.org) inspired by [picoGPT](https://github.com/jaymody/picoGPT).

This implementation:
* 130 lines of code + utils to load and convert parameters. 
* Contains only fwd path.
* Slow. Doesn't use GPU, multithreading etc. 
* Written in education purpose and may contain bugs.


#### Install
```bash
pip install -r requirements.txt
```
Tested on `Python 3.9`.

#### Usage
```bash
python wav2vec.py
```

#### Output
```bash
transcript:  I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|
```

#### Files
* wav2vec.py - model layers implementation 
* utils.py - helper functions to download and convert model parameters and example wav file

#### TODO
* Add option to load user specified wav
