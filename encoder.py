import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, obs_shape, feature_dim, action_dim, num_layers=2, num_filters=32):
        super().__init__()

        # self.encoder1 = PixelEncoder(obs_shape, feature_dim, num_layers, num_filters)
        # self.encoder2 = PixelEncoder(obs_shape, feature_dim, num_layers, num_filters)

        self.action_map = mlp([feature_dim*2, 512, 512, action_dim], nn.Tanh)

    def forward(self, encoding1, encoding2):
        # encoding1 = self.encoder1(img1)
        # encoding2 = self.encoder2(img2)
        predict = self.action_map(torch.cat((encoding1, encoding2), dim=1))
        return predict

class SplitEncoder(nn.Module):
    """Split Encoder: encode observation to a controllable feature and an uncontrollable feature
    """
    def __init__(self, encoder_1, encoder_2):
        super().__init__()

        self.controllable = encoder_1
        self.uncontrollable = encoder_2 
    
    def forward(self, obs, detach=False):
        return self.controllable(obs, detach), self.uncontrollable(obs, detach)

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        self.controllable.copy_conv_weights_from(source.controllable)
        self.uncontrollable.copy_conv_weights_from(source.uncontrollable)
    
    def log(self, L, step, log_freq):
        self.controllable.log(L, step, log_freq)
        self.uncontrollable.log(L, step, log_freq)

class HierarchicalEncoder(nn.Module):
    """Hierarchical Encoder: encode observation to a normal (reconstructable) representation, 
    followed by a predictable representation and a controllable representation
    """
    def __init__(self, base_encoder, in_feature_dim, out_feature_dim, num_layers=1, hidden_dim=256):
        super().__init__()

        self.base_encoder = base_encoder
        sizes = [in_feature_dim] + [hidden_dim]*num_layers + [out_feature_dim]
        self.predictable = mlp(sizes, nn.Tanh)
        self.controllable = mlp(sizes, nn.Tanh)
        

    def forward(self, obs, detach=False):
        base_output = self.base_encoder(obs, detach)
        predictable_output = self.predictable(base_output)
        controllable_output = self.controllable(base_output)
        return base_output, predictable_output, controllable_output

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        self.base_encoder.copy_conv_weights_from(source.base_encoder)
    
    def log(self, L, step, log_freq):
        self.base_encoder.log(L, step, log_freq)

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = torch.tanh(h_norm)
        self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {
    'pixel': PixelEncoder, 
    'identity': IdentityEncoder, 
}



def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, 
    hierarchical=False, hier_layers=1, split=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    encoder = _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )
    if hierarchical:
        return HierarchicalEncoder(encoder, feature_dim, feature_dim, num_layers=hier_layers)
    elif split:
        encoder_2 = _AVAILABLE_ENCODERS[encoder_type](
            obs_shape, feature_dim, num_layers, num_filters
        )
        return SplitEncoder(encoder, encoder_2)
    else:
        return encoder
