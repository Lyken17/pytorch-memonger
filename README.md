# pytorch-memonger

This is a re-implementation of [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174). 
You may also want to have a look at the [original mxnet implementation](https://github.com/dmlc/mxnet-memonger) and 
[OpenAI's tensorflow implementation](https://github.com/openai/gradient-checkpointing).

## How to use 

Different from TensorFlow and mxnet where the computation graph is static and known before actual computing,
pytorch's philosophy is **define-by-run** and the graph details are not known until forward is finished. This implemention
only supports `Sequential` models. By replacing `nn.Sequential` with `memonger.SublinearSequential`, 
the memory required for backward is reduced from `O(N)` to `O(sqrt(N))`.

```python
# previous, O(N) memory footprint
import torch.nn as nn
net1 = nn.Sequential(
    nn.Conv2d(3, 16, kernel=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    ...
)

# optimized, O(sqrt(N)) memory footprint
from momonger import SublinearSequential
net2 = SublinearSequential(
    *list(net1.children())  
)
```    
 
## Speed / Memory Comparision

Model (Batch size 16) | Memory | Speed 
--- | --- | ---
original resnet152	| 5459MiB | 2.9258 iter/s
Checkpoint (Sublinear) | 2455MiB | 2.6273 iter/s

## Caution

Since sublinear memory optimization requires re-forwarding, if your model contains layer with non-derministic behavior 
(e.g, BatchNorm, Dropout), you need to be careful when using the module. I have supported BatchNorm by [re-scaling momentum 
](momonger/memonger.py#L24). Support for dropout is still under construction.
