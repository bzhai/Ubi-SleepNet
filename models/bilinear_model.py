import torch


class BilinearFusion(torch.nn.Module):
    """
    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.
    Original github: https://github.com/HaoMood/bilinear-cnn/blob/master/src/bilinear_cnn_all.py
    Adapted by: Bing Zhai
    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 200.
    """
    def __init__(self, feature_map_dim=512, linear_dim=1024):
        """Declare all needed layers."""
        super(BilinearFusion, self).__init__()
        self.feature_map_dim = feature_map_dim
        self.linear_dim = linear_dim
        # Linear classifier.
        self.fc = torch.nn.Linear(self.feature_map_dim**2, self.linear_dim)

    def forward(self, x_1, x_2):
        """Forward pass of the network.
        Args:
            x_1, the first input matrix shape= [N, feature_map_dim, temporal_steps] , e.g., [32, 512, 10].
            x_2, the second input matrix shape= [N, feature_map_dim, temporal_steps]
        Returns:
            x, fused bilinear results shape = [N, feature_map_dim, temporal_steps].
        """
        assert x_1.size() == x_2.size()
        N = x_1.size()[0]
        num_time_step = x_1.size()[2]
        x = torch.bmm(x_1, torch.transpose(x_2, 1, 2)) / num_time_step  # Bilinear
        assert x.size() == (N, self.feature_map_dim, self.feature_map_dim)
        x = x.view(N, self.feature_map_dim ** 2)  # reduce the dimensions
        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)
        x = self.fc(x)
        assert x.size() == (N, self.linear_dim)
        return x
