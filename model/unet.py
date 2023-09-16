import torch
import torch.nn as nn

from model.module import linear, conv_nd, normalization, zero_module, timestep_embedding
from model.module import ResBlock, AttentionBlock, TimestepSequential

class UNet(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param input_channel: channels in the input Tensor.
    :param base_channel: base channel count for the model.
    :param num_residual_blocks_of_a_block: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_multiplier: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_class: if specified (as an int), then this model will be
        class-conditional with `num_class` classes.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        input_channel,
        base_channel,
        channel_multiplier,
        num_residual_blocks_of_a_block,
        attention_resolutions,
        num_heads,
        head_channel,
        use_new_attention_order,
        dropout,
        num_class=None,
        dims=2,
        learn_sigma=False,
        **kwargs
    ):
        super().__init__()
        self.num_class = num_class
        self.base_channel = base_channel
        output_channel = input_channel * 2 if learn_sigma else input_channel
        time_embed_dim = base_channel * 4
        self.time_embed = nn.Sequential(
            linear(base_channel, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_class is not None:
            self.label_emb = nn.Embedding(num_class, time_embed_dim)

        ch = input_ch = int(channel_multiplier[0] * base_channel)
        self.input_blocks = nn.ModuleList(
            [TimestepSequential(conv_nd(dims, input_channel, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_multiplier):
            for _ in range(num_residual_blocks_of_a_block):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * base_channel),
                        dims=dims
                    )
                ]
                ch = int(mult * base_channel)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=head_channel,
                            use_new_attention_order=use_new_attention_order
                        )
                    )
                self.input_blocks.append(TimestepSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_multiplier) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            down=True
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=head_channel,
                use_new_attention_order=use_new_attention_order
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_multiplier))[::-1]:
            for i in range(num_residual_blocks_of_a_block + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(base_channel * mult),
                        dims=dims
                    )
                ]
                ch = int(base_channel * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=head_channel,
                            use_new_attention_order=use_new_attention_order
                        )
                    )
                if level and i == num_residual_blocks_of_a_block:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            up=True
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, output_channel, 3, padding=1)),
        )

    def forward(self, x, time, condition=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param time: a 1-D batch of timesteps.
        :param condition: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        hs = []
        emb = self.time_embed(timestep_embedding(time, self.base_channel))

        if self.num_class is not None:
            assert condition is not None
            emb = emb + self.label_emb(condition)

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)