# LAE-LS
# An LLM-Enhanced Adversarial Editing System for Lexical Simplification

The repository for **An LLM-Enhanced Adversarial Editing System for Lexical Simplification**

If you find this code useful in your research, please cite:
> @article{tan2024llm,  
>   title={An LLM-Enhanced Adversarial Editing System for Lexical Simplification},  
>   author={Tan, Keren and Luo, Kangyang and Lan, Yunshi and Yuan, Zheng and Shu, Jinlong},  
>   journal={arXiv preprint arXiv:2402.14704},  
>   year={2024}  
> }

## Installation
```
pip install -r requirements.txt
```

## Datasets
- Lexcival Simplification datasets: 
```
data-LS
```
- Text Simplification datasets: 
```
data-TS
```

## Pretrained Models
bert-base-uncased
[[BERT]]([https://huggingface.co/bert-base-chinese](https://huggingface.co/google-bert/bert-base-uncased))

## Train model
- Train Discriminator:
```
python train_discriminator.py
```
- Train LAE-LS model:
```
python train_adv.py
```

## Model Inference
```
python predict.py
```

## Evaluation method
```
python evaluation.py
```
