# seqgen
a retrieval-enhanced sequence generation framework 

Source code for our EMNLP19 paper "Retrieval-guided Dialogue Response Generation via a Matching-to-Generation Framework"

Our dataset is released [here](https://ai.tencent.com/ailab/nlp/dialogue/) under the name ***Retrieval Generation Chat***.(direct download link:[Retrieval_Generation_Chat.zip](https://ai.tencent.com/ailab/nlp/dialogue/datasets/Retrieval_Generation_Chat.zip))

code is tested with **python==3.6.8** and **torch==1.2.0**
## Run Demo

`run_demo.sh` with details explained in `deploy.py`. you can query the demo in the following ways:

1. query + retrievals => skeletons + responses
   http://0.0.0.0:8080/query_retrievals?query=q&retrievals=r1;;;r2;;;r3
   where q is a single query, r1, r2 and r3 are multiple retrieval responses (also support single retrieval response), seperated by `;;;`.

   return format: {'skeletons':[s1, s2, s3], "responses":[re1, re2, re3]}

2. query + responses => ranks
   http://0.0.0.0:8080//query_responses?query=q&responses=r1;;;r2;;;r3
   where q is a single query, r1, r2 and r3 are multiple response candidates seperated by `;;;`.

   return format: {'rank':[ra1, ra2, ra3]} correspond to teh rank of r1, r2, r3 (start from 0)

3. query + skeleton => response
   http://0.0.0.0:8080/query_skeleton?query=q&skeleton=s
   where q is a single query, s is a single skeleton, the blanks in the skeleton is indicated by `;;;` , e.g., 今天(today);;;上课(in class);;;表扬(praise);;;.

   return format:  {'respones':r}

Note the demo requires pretrianed neural masker, generator, and ranker. To obtain your own models, please refer to the following instructions.

## Train and Test 

1. Masker
`ranker/train.sh` and `ranker/another_work.sh`
2. Generator
`./tain.sh` and `./work.sh` 

3. Ranker
`ranker/train.sh` and `ranker/work.sh`

Detailed instructions will be added soon.

## Citation

If you find the code useful, please cite our paper.
```
@inproceedings{cai-etal-2019-retrieval,
    title = "Retrieval-guided Dialogue Response Generation via a Matching-to-Generation Framework",
    author = "Cai, Deng  and
      Wang, Yan  and
      Bi, Wei  and
      Tu, Zhaopeng  and
      Liu, Xiaojiang  and
      Shi, Shuming",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1195",
    doi = "10.18653/v1/D19-1195",
    pages = "1866--1875",
}
```
## Contact
For any questions, please drop an email to [Deng Cai](https://jcyk.github.io/).
