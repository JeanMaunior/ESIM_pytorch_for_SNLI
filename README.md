# ESIM_pytorch_for_SNLI

This is a toy project for implementing ESIM in pytorch. Only ESIM is modeled, without Tree-LSTM.

Pretrained GloVe vocabulary needs to be download and its path must be included in `main.py` .

As for SNLI 1.0 dataset, I just drop out the "others" category simply.The unzipped dataset also needs to be download to `./data/`.It's a good choice to customize the path in `data_convert.py` and `data_helper.py` .

## Results

I got 80.9% on test set after training for 17 epoch, taking 8 hours on GTX 1050Ti.

## Reference

[Reasoning about Entailment with Neural Attention]( https://arxiv.org/pdf/1609.06038v3.pdf)