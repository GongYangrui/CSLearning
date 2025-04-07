# miniGPT
miniGPT是根据教学视频[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6793s)而来，帮助我理清楚了Transformer结构的基础细节，该miniGPT仅仅包含了multi-head attention以及Feed Forward两个模块，Cross Attention还没有包含。

## 小细节
### LayerNorm
`LayerNorm`的主要作用是让每一层的输出分布更稳定，一个标准的Transformer Block通常包含两个子结构，分别是自注意力模块和前馈网络，现在的主流做法是**Pre-LN**，也就是在input输入到自注意力模块或者是前馈神经网络之前，进行`LayerNorm`。
```python
x = x + sa(LayerNorm(x))
x = x + ff(LayerNorm(x))
```
另外，在最后的`output_projection`之前，也就是从隐藏层维度映射到词汇表维度的之前，从block模块中输出的output也需要经过`LayerNorm`。

### Dropout
Dropout是一种正则化的方法，用于防止模型过拟合，它会在训练过程中丢弃一部分神经元的输出，防止模型对某些特征过度依赖，从而提高泛化能力。

Dropout通常用于：
- Linear之后，作用时防止MLP过拟合；
- Attention权重之后，防止过度依赖单个token；
- 在残差连接之前

在[miniGPT.ipynb](/miniGPT/miniGPT.ipynb)中在skip connection之前，以及计算attention score(softmax之后)使用了dropout技术。
  
```python
attn_weights = softmax(score)
attn_weights = dropout(attn_weights)
attn_output = attn_weights @ V

# 注意力模块的输出之后,在residual之前
x = x + dropout(attn_output)
# 前馈神经网络的输出之后,在residual之前
x = x + dropout(ffn_output)
```