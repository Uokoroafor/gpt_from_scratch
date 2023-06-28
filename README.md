# transformer_from_scratch

This is a PyTorch implementation of a smaller version of the GPT model. It has been inspired by the work of Andrej Karpathy and his nanoGPT [repo](https://github.com/karpathy/nanoGPT/tree/master) and  video [tutorial](https://github.com/karpathy/ng-video-lecture/).

It enacts the components of the transformer architecture in the pre-Norm style, which is the style used in the original paper.

The key components are:
- *Positional Encoding*: While in the original paper, the positional encoding is a sine and cosine function that is added to the input embeddings to give the model a sense of position in the sequence, I have used a learned positional encoding. This is a vector of the same dimension as the input embeddings that is added to the input embeddings. This is a vector that is learned during training.

- *Scaled Dot Product Attention*: This is the attention mechanism used in the Transformer. It is a dot product between the query and key vectors, scaled by the square root of the dimension of the key vectors. The output is a weighted sum of the value vectors.

- *Multi-Head Attention*: This is a concatenation of multiple attention heads. Each head is a scaled dot product attention mechanism. The output of each head is concatenated and then projected to the output dimension.

- *Feed Forward Network*: This is a two layer fully connected network with a ReLU activation function in between the layers.

- *Post-Norm Architecture*: This is a residual connection that is added to the output of each sub-layer and then normalised by layer normalisation. In the original paper, the residual connection is added to the input of each sub-layer and then normalised by layer normalisation.
## Installation
```
git clone https://github.com/Uokoroafor/gpt_from_scratch
cd gpt_from_scratch
pip install -r requirements.txt
```
## Usage
There is a sample training loop in train_gpt.py. Populate training parameters as below:
```python
import torch
training_hyperparams = {
    'batch_size': 32,
    'epochs': 1000,
    'learning_rate': 5e-5,
    'max_seq_len': 64,
    'num_heads': 8,
    'num_layers': 4,
    'd_model': 128,
    'd_ff': 128 * 4,
    'dropout_prob': 0.2,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'eval_every': 100,
    'eval_iters': 10,
    'save_every': 100,
}
``` 

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Andrej Karpathy's github repo](https://github.com/karpathy)

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


