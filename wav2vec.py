import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def group_norm(x, weight, bias):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return weight[None, :, None] * (x - mean) / np.sqrt(variance + 1e-5) + bias[None, :, None]

def conv1d(x, weight, bias=None, stride=2, padding=0, dilation=1, groups=1): 
    N, Cin, Lin = x.shape
    Cout = weight.shape[0]
    kernel_size = weight.shape[-1]
    Lout = (Lin + 2*padding - dilation * (kernel_size - 1) - 1) // stride + 1

    x = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
    x = x.reshape((N, groups, Cin//groups, Lin + 2*padding))
    w = weight.reshape((groups, Cout//groups, Cin//groups, kernel_size))

    y = np.zeros((N, groups, Cout//groups, Lout))
    for n in range(N):
        for g in range(groups):
            for l in range(Lout):
                win = x[n, g, :, l*stride : l*stride + kernel_size] 
                acc = win * w[g, ...]
                res = np.sum(acc, axis=(1, 2))
                y[n, g, :, l] = res

    y = y.reshape((N, Cout, Lout)) 
    y = y + bias[:, None] if bias is not None else y
    return y

def softmax(x, dim):
    x = np.exp(x - x.max(axis=dim, keepdims=True))
    return x / x.sum(axis=dim, keepdims=True)
    
def ConvBlock(x, conv_stride, conv, layer_norm=None):
    x = conv1d(x, stride=conv_stride, **conv)
    if layer_norm is not None:
        x = group_norm(x, **layer_norm)
    return gelu(x)

def FeatureExtractor(x, params):
    x = x[None, :]
    for i in range(len(params["conv_layers"])):
        x = ConvBlock(x, conv_stride=5 if i==0 else 2, **params["conv_layers"][f"{i}"])
    return np.transpose(x, (0, 2, 1))

def layer_norm(x, weight, bias):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return weight * (x - mean) / np.sqrt(variance + 1e-5) + bias

def linear(x, weight, bias):
    return x @ weight.T + bias

def FeatureProjection(x, params):
    x = layer_norm(x, **params["layer_norm"])
    x = linear(x, **params["projection"])
    return x

def WeightNorm(g, v):
    """ Computes weights from v and g. PT uses norm_except_dim with dim=2"""
    return g / np.sqrt(np.sum(v*v, axis=(0, 1), keepdims=True)) * v

def ConvolutionalPositionalEmbedding(x, params):
    x = x.transpose((0, 2, 1))
    w = WeightNorm(params["conv"]["weight_g"], params["conv"]["weight_v"])
    x = conv1d(x, weight=w, bias=params["conv"]["bias"], stride=1, padding=64, groups=16)
    x = x[..., : -1]
    x = gelu(x)
    return x.transpose(0, 2, 1)

def FeedForward(x, params):
    x = linear(x, **params["intermediate_dense"])
    x = gelu(x)
    x = linear(x, **params["output_dense"])
    return x

def SelfAttention(x, params):
    q = linear(x, **params["q_proj"])
    k = linear(x, **params["k_proj"])
    v = linear(x, **params["v_proj"])

    batch_size, length, embed_dim = x.shape
    shape = (batch_size, length, 12, 64) # num_heads=12, head_dim=64
    q = np.reshape(q, shape).transpose((0, 2, 1, 3))
    k = np.reshape(k, shape).transpose((0, 2, 3, 1))
    v = np.reshape(v, shape).transpose((0, 2, 1, 3))

    weights = softmax((0.125 * q) @ k, dim=-1) # scaling = 0.125

    output = weights @ v  
    output = output.transpose(0, 2, 1, 3).reshape((batch_size, length, embed_dim))
    return linear(output, **params["out_proj"])

def EncoderLayer(x, params):
    x = x + SelfAttention(x, params["attention"])
    x = layer_norm(x, **params["layer_norm"])
    x = x + FeedForward(x, params["feed_forward"])
    x = layer_norm(x, **params["final_layer_norm"])
    return x

def Transformer(x, params):
    x = x + ConvolutionalPositionalEmbedding(x, params["pos_conv_embed"])
    x = layer_norm(x, **params["layer_norm"])
    for layer in range(len(params["layers"])):
        x = EncoderLayer(x, params["layers"][f"{layer}"])
    return x

def Encoder(x, params):
    x = FeatureProjection(x, params["feature_projection"])
    x = Transformer(x, params["transformer"])
    return x

def Wav2Vec(x, params):
    x = FeatureExtractor(x, params["feature_extractor"]) # params["feature_extractor"])
    x = Encoder(x, params["encoder"])
    return linear(x, **params["aux"])

def decode(emission, labels, blank=0):
    from itertools import groupby
    indices = np.argmax(emission, axis=-1)  # [num_seq,]
    indices = [i[0] for i in groupby(indices)]
    indices = [i for i in indices if i != blank]
    return "".join([labels[i] for i in indices])

def main():
    from utils import load_data
    x, params, labels = load_data()
    y = Wav2Vec(x, params)
    transcript = decode(y[0], labels)
    print("transcript: ", transcript)

if __name__=="__main__":
    main()

