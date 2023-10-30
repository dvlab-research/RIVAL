from einops import rearrange
import torch

def new_forward(
    self,
    hidden_states,
    encoder_hidden_states=None,
    attention_mask=None,
    temb=None,
    **cross_attention_kwargs
):
    # Implement your new forward function here.
    # You can access the parameters of the original module as self.weight, etc.
    if not hasattr(self, "editing_early_steps"):
        self.editing_early_steps = 1000
        
    if (hidden_states.shape[0] <= 2) or encoder_hidden_states is not None:
        # if hidden_states is with batch_size = 1, then we can use the original forward function:
        return self.ori_forward(
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            **cross_attention_kwargs
        )
    else:
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        self.init_step -= self.step_size

        if self.init_step > self.editing_early_steps:
            return self.ori_forward(
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                **cross_attention_kwargs
            )

        frame_length = 1
        if frame_length == 1 and hidden_states.shape[0] == 4:
            frame_length = 2

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        encoder_hidden_states = encoder_hidden_states

        query = self.to_q(hidden_states)

        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(frame_length).long()
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=frame_length)  # .transpose(0, 1)
        if self.cfg["atten_frames"] != -1:
            if self.init_step > self.t_align:
                self.cfg["atten_frames"] = 1
            else:
                self.cfg["atten_frames"] = 2

        # ic(key.shape) b,f,2N,c
        if self.cfg["atten_frames"] == 1:
            key = torch.cat([key[:, [0] * int(frame_length)]], dim=2)
        elif self.cfg["atten_frames"] == 2:
            key = torch.cat(
                [key[:, [0] * int(frame_length)], key[:, former_frame_index]], dim=2
            )

        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(
            value, "(b f) d c -> b f d c", f=frame_length
        )  # .transpose(0, 1)

        if self.cfg["atten_frames"] == 1:
            value = torch.cat([value[:, [0] * int(frame_length)]], dim=2)
        elif self.cfg["atten_frames"] == 2:
            value = torch.cat(
                [value[:, [0] * int(frame_length)], value[:, former_frame_index]], dim=2
            )
        value = rearrange(value, "b f d c -> (b f) d c")

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        if self.init_step <= 0:
            self.init_step = 1000
            # ic("reset")

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (0, target_length), value=0.0
                )
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
            # attention, what we cannot get enough of
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        return hidden_states