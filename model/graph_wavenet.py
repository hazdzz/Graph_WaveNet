import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedTemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out):
        super(GatedTemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.conv = nn.Conv2d(c_in, 2 * c_out, (Kt, 1), 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x_conv = self.conv(x)
        P = x_conv[:, : self.c_out, :, :]
        Q = x_conv[:, -self.c_out:, :, :]
        x_gated_temp_conv = torch.matmul(self.tanh(P), self.sigmoid(Q))
        return x_gated_temp_conv

class SelfAdaptiveConv(nn.Module):
    def __init__(self, Ks, c_in, c_out, sac_kernel, enable_bias):
        super(SelfAdaptiveConv, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.sac_kernel = sac_kernel
        self.enable_bias = enable_bias
        
class GraphDiffusionConv(nn.Module):
    def __init__(self, Ks, c_in, c_out, gdc_kernel, enable_bias):
        super(GraphDiffusionConv, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.gdc_kernel = gdc_kernel
        self.enable_bias = enable_bias

class GraphDiffusionConvSAA(nn.Module):
    def __init__(self, Ks, c_in, c_out, gdc_kernel, enable_bias):
        super(GraphDiffusionConvSAA, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.gdc_kernel = gdc_kernel
        self.enable_bias = enable_bias
        self.node_embedding_1 = nn.Parameter()

class GraphConv_LWL(nn.Module):
    def __init__(self, Ks, c_in, c_out, gc_lwl_kernel, enable_bias):
        super(GraphConv_LWL, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.gc_lwl_kernel = gc_lwl_kernel
        self.enable_bias = enable_bias

class GraphConvLayer(nn.Module):
    def __init__(self, Ks, c_in, c_out, gc, graph_conv_kernel):
        super(GatedTemporalConvLayer, self).__init__()
        self.Ks = Ks
        self.c_in = c_in
        self.c_out = c_out
        self.gc = gc
        self.graph_conv_kernel = graph_conv_kernel
        self.enable_bias = True
        if self.gc == "sac":
            self.sac = SelfAdaptiveConv(self.Ks, c_in, c_out, self.graph_conv_kernel, self.enable_bias)
        elif self.gc == "gdc":
            self.gdc = GraphDiffusionConv(self.Ks, c_in, c_out, self.graph_conv_kernel, self.enable_bias)
        elif self.gc == "gdc_saa":
            self.gdc_saa = GraphDiffusionConvSAA(self.Ks, c_in, c_out, self.graph_conv_kernel, self.enable_bias)
        elif self.gc == "gc_lwl":
            self.gc_lwl = GraphConv_LWL(self.Ks, c_in, c_out, self.graph_conv_kernel, self.enable_bias)

    def forward(self, x):

class Graph_WaveNet(nn.Module):
    def __init__(self, Kt, Ks, T, n_vertex, gc, gdc_saa, drop_prob):
        super(Graph_WaveNet, self).__init__()

    def forward(self, x):


