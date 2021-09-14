# CIRT.5<sub>embed</sub>
Contextual Information Representation Technique based on 5 steps and using word embedding based models (CIRT.5<sub>embed</sub>) is a new text representation technique based on vector space model. This technique assumes that the frequency relationship between terms is dependent, considering the reliance of a set of correlated terms (context) directly proportional to the frequency with his terms occurs in a text document. Thus, this CIRT.5<sub>embed</sub> based script generate a unique vector representation to each document, calculating the frequency of all contexts in all documents. In contrast with classics text representation techniques such as bag of words, the CIRT.5<sub>embed</sub> technique allows to consider the similarities between the terms that compose the different contexts, moderated according to a predefined threshold. The output is a matrix where rows are the documents ids and columns are the frequencies to each document.

> Generating a CIRT.5<sub>embed</sub> based text representation matrix:
```
python3 CIRT.5_embed.py --language EN --contexts 3 --thresholds 0.05 --model models/model --input in/db/ --output out/CIRT.5_embed/txt/
```
> Example of converting a Doc-Context matrix to Arff file (Weka):
```
python3 Bag2Arff.py --input out/CIRT.5_embed/txt/ --output out/CIRT.5_embed/arff/
```

![](https://joao8tunes.github.io/hello/wp-content/uploads/photo-gallery/LABIC_image_8_1538169499.png?bwg=1542306976)

# Related scripts
* [CIRT.5_embed.py](https://github.com/joao8tunes/CIRT.5_embed/blob/master/CIRT.5_embed.py)
* [Bag2Arff.py](https://github.com/joao8tunes/Bag2Arff/blob/master/Bag2Arff.py)


# Assumptions
These scripts expect a database folder following an specific hierarchy like shown below:
```
in/db/                 (main directory)
---> class_1/          (class_1's directory)
---------> file_1      (text file)
---------> file_2      (text file)
---------> ...
---> class_2/          (class_2's directory)
---------> file_1      (text file)
---------> file_2      (text file)
---------> ...
---> ...
```


# Observations
All generated Doc-Context matrices use *TAB* character as separator.


# Requirements installation (Linux)
> Python 3 + PIP installation as super user:
```
apt-get install python3 python3-pip
```
> Gensim installation as normal user:
```
pip3 install --upgrade gensim
```
> NLTK + Scipy + Numpy installation as normal user:
```
pip3 install -U nltk scipy numpy
```


# See more
Project page on LABIC website: http://sites.labic.icmc.usp.br/MSc-Thesis_Antunes_2018
