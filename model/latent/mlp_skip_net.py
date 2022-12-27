from ..module import *

class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers

    default MLP for the latent DPM in the paper!
    """
    def __init__(
        self,
        input_channel, # latent_channel
        model_channel,
        num_layers,
        time_emb_channel,
        use_norm,
        dropout,
        **kwargs
    ):
        super().__init__()

        self.input_channel = input_channel
        self.skip_layers = list(range(1,num_layers))
        self.time_emb_channel = time_emb_channel

        self.time_embed = nn.Sequential(
            nn.Linear(self.time_emb_channel, input_channel),
            nn.SiLU(),
            nn.Linear(input_channel, input_channel),
        )

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                act = "silu"
                norm = use_norm
                cond = True
                a, b = input_channel, model_channel
                dropout = dropout
            elif i == num_layers - 1:
                act = "none"
                norm = False
                cond = False
                a, b = model_channel, input_channel
                dropout = 0
            else:
                act = "silu"
                norm = use_norm
                cond = True
                a, b = model_channel, model_channel
                dropout = dropout

            if i in self.skip_layers:
                a += input_channel

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=input_channel,
                    use_cond=cond,
                    dropout=dropout,
                ))

    def forward(self, x, t):
        t = timestep_embedding(t, self.time_emb_channel)
        cond = self.time_embed(t)
        h = x
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond)
        output = h
        return output


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm,
        use_cond,
        activation,
        cond_channels,
        dropout,
    ):
        super().__init__()
        self.activation = activation
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.SiLU() if self.activation == "silu" else nn.Identity()
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == "silu":
                    nn.init.kaiming_normal_(module.weight, a=0, nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (1.0 + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x