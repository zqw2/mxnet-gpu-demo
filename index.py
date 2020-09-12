import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

print(mx.cpu(), mx.gpu())
print(mx.context.num_gpus())