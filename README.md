# ChiCoupletGen

This is the implementation for [Chinese Couplet Generation with Syntactic Information](https://aclanthology.org/2022.coling-1.560/) at COLING2022

## Citation

```
@inproceedings{song-2022-chinese,
    title = "Chinese Couplet Generation with Syntactic Information",
    author = "Song, Yan",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    pages = "6436--6446",
}
```

## Requirements

```pytorch```

```transformers```

## How to Train the Models

Download the [pretrained ```bert-base-chinese``` model](https://huggingface.co/bert-base-chinese/tree/main) and put it in the ```pretrained``` directory.

Put the corpus data under the ```data_new``` directory.

The final directory should look like this:

```
root
    data_new
        dev
        ...
    measure
    ...
    pretrained
        bert-base-chinese
    utils
    ...
    train_bert_seg.py
```

The script to train the models in the paper is the ```train_xxx.py```, where ```xxx``` corresponds to the variants of models in the paper. To train a model, please run the corresponding script.
