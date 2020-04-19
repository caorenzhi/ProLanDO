# ProLanDO

ProLanDO is a novel machine learning method with help of natural language processing to predict Disorder Ontology (DO) terms for protein sequence.

# Citation
--------------------------------------------------------------------------------------


# Test Environment
--------------------------------------------------------------------------------------
Ubuntu, Centos

# Requirements
--------------------------------------------------------------------------------------
(1). Python3+

(2). Pytorch
```
https://pytorch.org/

```
GPU is NOT required.

# Test example
--------------------------------------------------------------------------------------
You could provide one fasta format file or a folder with several fasta format files for this software. Here are examples to test:

#cd script

#python ProLanDO_oneFasta_model1.py ../model/training_DO_RNN_512_0.009000000000000001.model ../test/test3.fasta ../test/ProLanDO_test3_do.txt

#python ProLanDO_fastaFolder_model1.py ProLanDO_oneFasta_model1.py ../test/fastaFolder ../model/training_DO_RNN_512_0.009000000000000001.model ../test/ProLanDO_testFolder_DO

You should be able to find the output file named TopQAScores.txt in the output folder.


--------------------------------------------------------------------------------------
Developed by John Smith and Dr. Renzhi Cao at Pacific Lutheran University:

Please contact Dr. Cao for any questions: caora@plu.edu (PI)
