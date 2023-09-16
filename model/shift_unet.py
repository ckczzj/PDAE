import torch
import torch.nn as nn

from model.module import linear, conv_nd, normalization, zero_module, timestep_embedding
from model.module import ResBlock, ResBlockShift, AttentionBlock, TimestepSequential

class ShiftUNet(nn.Module):
    """
    ShiftUNet based on UNet with additive trainable label_emb, shift_middle_block, shift_output_blocks and shift_out.

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
    :param latent_dim: latent dim
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
        latent_dim,
        dims=2,
        learn_sigma=False,
        **kwargs
    ):
        super().__init__()
        self.base_channel = base_channel
        output_channel = input_channel * 2 if learn_sigma else input_channel

        time_embed_dim = base_channel * 4

        # this layer is freeze, which is trained in diffusion model
        self.time_embed = nn.Sequential(
            linear(base_channel, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # original class label
        # self.label_emb = nn.Embedding(latent_dim, time_embed_dim)

        # free representation learning
        # this layer is trainable
        self.label_emb = nn.Linear(latent_dim, time_embed_dim)

        ch = input_ch = int(channel_multiplier[0] * base_channel)
        self.input_blocks = nn.ModuleList(
            [TimestepSequential(conv_nd(dims, input_channel, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        shift_input_block_chans = [ch]
        ds = 1
        shift_ds = 1
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
                input_block_chans.append(ch)
                shift_input_block_chans.append(ch)
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
                shift_input_block_chans.append(ch)
                ds *= 2
                shift_ds *= 2

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

        self.shift_middle_block = TimestepSequential(
            ResBlockShift(
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
            ResBlockShift(
                ch,
                time_embed_dim,
                dropout,
                dims=dims
            ),
        )

        memory_ch = ch

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

        ch = memory_ch
        self.shift_output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_multiplier))[::-1]:
            for i in range(num_residual_blocks_of_a_block + 1):
                ich = shift_input_block_chans.pop()
                layers = [
                    ResBlockShift(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(base_channel * mult),
                        dims=dims
                    )
                ]
                ch = int(base_channel * mult)
                if shift_ds in attention_resolutions:
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
                        ResBlockShift(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            up=True
                        )
                    )
                    shift_ds //= 2
                self.shift_output_blocks.append(TimestepSequential(*layers))


        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, output_channel, 3, padding=1)),
        )

        self.shift_out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, input_channel, 3, padding=1)),
        )

        self.freeze()

    def forward(self, x, time, condition):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param time: a 1-D batch of timesteps.
        :param condition: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        emb = self.time_embed(timestep_embedding(time, self.base_channel))
        shift_emb = self.label_emb(condition)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        epsilon_h = self.middle_block(h, emb)
        shift_h = self.shift_middle_block(h, emb, shift_emb)

        for module, shift_module in zip(self.output_blocks, self.shift_output_blocks):
            h_previous = hs.pop()

            epsilon_h = torch.cat([epsilon_h, h_previous], dim=1)
            epsilon_h = module(epsilon_h, emb)

            shift_h = torch.cat([shift_h, h_previous], dim=1)
            shift_h = shift_module(shift_h, emb, shift_emb)

        return self.out(epsilon_h), self.shift_out(shift_h)


    def set_train_mode(self):
        self.label_emb.train()
        self.shift_middle_block.train()
        self.shift_output_blocks.train()
        self.shift_out.train()

    def set_eval_mode(self):
        self.label_emb.eval()
        self.shift_middle_block.eval()
        self.shift_output_blocks.eval()
        self.shift_out.eval()

    def freeze(self):
        self.time_embed.eval()
        self.input_blocks.eval()
        self.middle_block.eval()
        self.output_blocks.eval()
        self.out.eval()

        self.time_embed.requires_grad_(requires_grad=False)
        self.input_blocks.requires_grad_(requires_grad=False)
        self.middle_block.requires_grad_(requires_grad=False)
        self.output_blocks.requires_grad_(requires_grad=False)
        self.out.requires_grad_(requires_grad=False)
