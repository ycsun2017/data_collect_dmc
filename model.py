import torch
import torch.nn as nn


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class PredictModel(nn.Module):
    """Latent Transition Model."""
    def __init__(self, feature_dim, action_shape, hidden_dim, num_layers):
        super().__init__()

        sizes = [feature_dim]
        for _ in range(num_layers-1):
            sizes.append(hidden_dim)
        sizes.append(feature_dim)

        self.fc = mlp(sizes, activation=nn.Tanh)


    def forward(self, encoding):
        out = self.fc(encoding)
        return out

class ControlModel(nn.Module):
    """Latent Transition Model."""
    def __init__(self, feature_dim, action_shape, hidden_dim, num_layers):
        super().__init__()

        sizes = [feature_dim+action_shape[0]]
        for _ in range(num_layers-1):
            sizes.append(hidden_dim)
        sizes.append(feature_dim)

        self.fc = mlp(sizes, activation=nn.Tanh)


    def forward(self, encoding, action):
        inp = torch.cat((encoding, action), 1)
        out = self.fc(inp)
        return out



_AVAILABLE_MODELS = {'predict': PredictModel, 'control': ControlModel}


def make_model(
    model_type, feature_dim, action_shape, hidden_dim, num_layers
):
    assert model_type in _AVAILABLE_MODELS
    return _AVAILABLE_MODELS[model_type](
        feature_dim, action_shape, hidden_dim, num_layers
    )
