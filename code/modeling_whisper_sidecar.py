import math
import itertools
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .sidecar import Sidecar
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoderLayer, 
    WhisperEncoder, 
    WhisperDecoder,
    shift_tokens_right
)
from transformers import (
    WhisperConfig,
    WhisperModel, 
    WhisperForConditionalGeneration
)
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor
from transformers.utils import logging
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE

logger = logging.get_logger(__name__)

class WhisperSidecarEncoder(WhisperEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig, sidecar_loc: int = 1, num_spks: int = 2, target_asr: bool = False, for_target_asr_eval: bool = False,
                 sidecar_residual: bool = True, sidecar_activate: str = "sigmoid",
                 mask_norm: str = "none", sidecar_gate_init: float = 0.1):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)

        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.num_spks = num_spks
        self.num_perms = math.perm(self.num_spks, self.num_spks)
        # FIX(v2): store the separator wiring options so they can be forwarded
        # from finetune.py / evaluation.py via from_pretrained kwargs.
        self.sidecar_residual = sidecar_residual
        if sidecar_loc and sidecar_loc >= -1:
            self.sidecar_loc = sidecar_loc
            self.sep_enc = nn.Conv1d(embed_dim, embed_dim, 3, stride=1, padding='same')
            self.sidecar = Sidecar(N=embed_dim, B=128, H=embed_dim, num_spks=self.num_spks,
                                   activate=sidecar_activate, mask_norm=mask_norm)
            self.sep_dec = nn.Conv1d(embed_dim, embed_dim, 3, stride=1, padding='same')
            # FIX(v2): learnable residual gate.  With the original code the
            # sidecar output REPLACED the encoder hidden states; if the
            # separator produced near-zero (or all-zero) output the decoder
            # had nothing left to attend to and collapsed to the LM prior.
            # A gated residual keeps the pretrained Whisper pathway alive:
            #     h <- h + alpha * sep_dec(sidecar(sep_enc(h)))
            # `alpha` is a scalar initialised to a small value.
            # NOTE: keep this a 1-element tensor of shape (1,), NOT a 0-dim
            # scalar.  transformers' `from_pretrained` builds meta tensors via
            # `torch.empty(*param.size(), dtype=...)`; for a 0-dim parameter
            # `param.size() == ()` that call raises
            # `TypeError: empty() missing 1 required positional argument: "size"`.
            # Shape (1,) broadcasts against (N, T, C) exactly like a scalar.
            self.sidecar_alpha = nn.Parameter(torch.tensor([float(sidecar_gate_init)]))
            self.target_asr = target_asr
        else:
            assert self.num_spks == 1, "The num_spks should be 1 when sidecar is disabled."
            self.sidecar_loc = None
            self.target_asr = None

        if self.target_asr:
            self.proj_target = nn.Linear(embed_dim, 1)
            self.proj_target_2  = nn.Linear(150, 1)
            self.for_target_asr_eval = for_target_asr_eval

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def _apply_sidecar(self, hidden_states):
        """Run the separator on ``hidden_states`` of shape (B, T, C).

        FIX(v2): the original code did

            hidden_states = self.sep_dec(self.sidecar(self.sep_enc(hidden_states.T))).T

        i.e. it *replaced* the encoder stream by the separator output.  There
        was no skip connection around ``sep_enc``/``sidecar``/``sep_dec`` (the
        Conv1D_Block residuals are internal to the separator and live inside
        the low-dimensional bottleneck).  If the separator collapsed to zero
        (see ``param_init``), the whole encoder output became zero and the
        frozen decoder had nothing to attend to.

        The fixed version keeps the pretrained Whisper representation::

            h_out = repeat_interleave(h, num_spks) + alpha * sep(h)

        ``alpha`` is a learnable scalar initialised to ``sidecar_gate_init``
        (0.1) so training starts close to the frozen-Whisper behaviour and the
        separator has to *earn* its contribution.  The batch dimension still
        grows by ``num_spks`` (required by the PIT loss / label layout).
        """
        b = hidden_states.shape[0]
        x = hidden_states.transpose(1, 2)                  # (B, C, T)
        x = self.sep_enc(x)
        sidecar_out, masks = self.sidecar(x)
        self.last_sidecar_masks = masks.detach()           # for diagnostics
        sidecar_out = self.sep_dec(sidecar_out)            # (B*num_spks, C, T)
        self.last_sidecar_streams = sidecar_out                               # LOSS_V3
        _need_rec = (float(__import__('os').environ.get("SIDECAR_REC_LAM", "0") or 0) > 0
                     or (float(__import__('os').environ.get("SIDECAR_P3_BETA", "0") or 0) > 0 or float(__import__('os').environ.get("SIDECAR_KEEP_FEATS", "0") or 0) > 0))
        if _need_rec:
            self.last_streams_rec = sidecar_out                                # LOSS_V3
            self.last_mix_feats = hidden_states.repeat_interleave(self.num_spks, dim=0)  # LOSS_V3
        sidecar_out = sidecar_out.transpose(1, 2)          # (B*num_spks, T, C)

        if self.sidecar_residual and sidecar_out.shape[0] == b * self.num_spks:
            residual = hidden_states.repeat_interleave(self.num_spks, dim=0)
            _full = residual + self.sidecar_alpha * sidecar_out
            if (float(__import__('os').environ.get("SIDECAR_SEP_LAM", "0") or 0) > 0
                    or (float(__import__('os').environ.get("SIDECAR_P3_BETA", "0") or 0) > 0 or float(__import__('os').environ.get("SIDECAR_KEEP_FEATS", "0") or 0) > 0)):
                self.last_full_streams = _full.transpose(1, 2)                # LOSS_V3  (B*S, C, T)
            return _full
        return sidecar_out

    # ---- LOSS_V3 -----------------------------------------------------------
    def set_mel_filters(self, mel_filters):
        """Store Whisper's exact mel filterbank for the GPU mel front-end."""
        import numpy as _np
        mf = torch.tensor(_np.asarray(mel_filters), dtype=torch.float32)
        self.register_buffer("_mel_filters", mf, persistent=False)
        self.register_buffer("_hann_400", torch.hann_window(400), persistent=False)
        return mf.shape

    def _clean_features(self, wavs):
        """clean waveforms (B*S, L) -> encoder features at the Sidecar depth (no grad)."""
        mf = getattr(self, "_mel_filters", None)
        if mf is None or wavs is None or wavs.numel() == 0:
            return None
        win = self._hann_400.to(wavs.device)
        mf = mf.to(wavs.device)
        if mf.dim() == 2 and mf.shape[0] == 201 and mf.shape[1] == 80:
            mf = mf.t()
        if wavs.dim() == 3:
            wavs = wavs.reshape(-1, wavs.shape[-1])
        t = wavs.shape[-1]
        if t < 480000:
            wavs = torch.nn.functional.pad(wavs, (0, 480000 - t))
        elif t > 480000:
            wavs = wavs[..., :480000]
        spec = torch.stft(wavs.float(), n_fft=400, hop_length=160, window=win,
                          return_complex=True)
        mag = spec[..., :-1].abs() ** 2
        mel = torch.matmul(mf, mag)
        log = torch.clamp(mel, min=1e-10).log10()
        log = torch.maximum(log, log.amax(dim=(1, 2), keepdim=True) - 8.0)
        log = (log + 4.0) / 4.0
        h = torch.nn.functional.gelu(self.conv1(log))
        h = torch.nn.functional.gelu(self.conv2(h))
        h = h.permute(0, 2, 1) + self.embed_positions.weight
        n = 0 if self.sidecar_loc == -1 else self.sidecar_loc + 1
        for layer in self.layers[:n]:
            h = layer(h, None, layer_head_mask=None)[0]
        return h
    # ---- end LOSS_V3 -------------------------------------------------------

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        target_speaker=None,
        pit: bool = False,
        clean_audios=None,          # LOSS_V3
    ):
        r"""
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        # In order to maintain a consistent batch_size during eval, 
        # num_spks was duplicated in the collator. 
        # Therefore, it needs to be removed here.
        input_features = input_features[::self.num_spks]
        if attention_mask is not None:
            attention_mask = attention_mask[::self.num_spks]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # LOSS_V3: clean-feature targets (frozen path, no grad)
        self.last_clean_feats = None
        self.last_streams_rec = None
        self.last_mix_feats = None
        self.last_full_streams = None
        self.last_full_streams = None
        self.last_clean_feats = None
        self.last_p3_audit = None
        _rec_lam = float(__import__('os').environ.get("SIDECAR_REC_LAM", "0") or 0)
        if clean_audios is not None and (_rec_lam > 0 or
                                         (float(__import__('os').environ.get("SIDECAR_P3_BETA", "0") or 0) > 0 or float(__import__('os').environ.get("SIDECAR_KEEP_FEATS", "0") or 0) > 0)):
            with torch.no_grad():
                self.last_clean_feats = self._clean_features(clean_audios)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        if self.sidecar_loc == -1:
            hidden_states = self._apply_sidecar(hidden_states)

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if idx == self.sidecar_loc:
                hidden_states = self._apply_sidecar(hidden_states)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # target talker identifier
        if self.target_asr and (target_speaker is not None or self.for_target_asr_eval):
            target_spk_logits = self.proj_target(hidden_states[:, :150, :]).reshape(-1,self.num_spks, 150, 1).squeeze(-1)    # pooling, (B, self.num_spks)
            target_spk_logits = torch.relu(target_spk_logits)
            target_spk_logits = self.proj_target_2(target_spk_logits).squeeze(-1)    # pooling, (B, self.num_spks)
            hidden_states = hidden_states[:, 155:, :]

        if pit:
            _, t, c = hidden_states.shape
            hidden_states_unbind = hidden_states.view(-1, self.num_spks, t, c).unbind(1)
            hidden_states_perm = list(itertools.permutations(hidden_states_unbind))
            hidden_states = torch.stack([torch.stack(t, dim=1) for t in hidden_states_perm], dim=1).view(-1, t, c)


        elif self.target_asr and self.for_target_asr_eval:
            # Only select the hidden_states of target spk
            target_spk_idx = torch.argmax(target_spk_logits, dim=1)

            # mask those not target spk
            mask = torch.arange(self.num_spks, device=target_spk_idx.device) != target_spk_idx.unsqueeze(1)     # (B, num_spks)
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            hidden_states = hidden_states.view(-1, self.num_spks, hidden_states.shape[-2], hidden_states.shape[-1])
            hidden_states = hidden_states.masked_fill(mask, -1e9).view(-1, hidden_states.shape[-2], hidden_states.shape[-1])    # Block other spks


        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        else:
            encoder_outputs = BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            )

            if self.target_asr and target_speaker is not None and pit:
                encoder_outputs['target_spk_logits'] = target_spk_logits
            return encoder_outputs

class WhisperSidecarModel(WhisperModel):
    def __init__(self, config: WhisperConfig, sidecar_loc: int = 1, num_spks: int = 2, target_asr: bool = False, for_target_asr_eval: bool = False,
                 sidecar_residual: bool = True, sidecar_activate: str = "sigmoid",
                 mask_norm: str = "none", sidecar_gate_init: float = 0.1):
        super().__init__(config)
        self.num_spks = num_spks
        self.num_perms = math.perm(self.num_spks, self.num_spks)
        self.target_asr = target_asr

        self.encoder = WhisperSidecarEncoder(config, sidecar_loc, num_spks, target_asr, for_target_asr_eval,
                                             sidecar_residual=sidecar_residual,
                                             sidecar_activate=sidecar_activate,
                                             mask_norm=mask_norm,
                                             sidecar_gate_init=sidecar_gate_init)
        self.decoder = WhisperDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        target_speaker: Optional[torch.LongTensor] = None,
        pit: bool = False,
        clean_audios=None,          # LOSS_V3
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, WhisperModel
         >>> from datasets import load_dataset

         >>> model = WhisperModel.from_pretrained("openai/whisper-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            input_features = self._mask_input_features(input_features, attention_mask=attention_mask)

            encoder_outputs = self.encoder(
                input_features,
                clean_audios=clean_audios,        # LOSS_V3
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                target_speaker=target_speaker,
                pit=pit,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        outputs = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        if self.target_asr and "target_spk_logits" in encoder_outputs:
            outputs['target_spk_logits'] = encoder_outputs['target_spk_logits']
        return outputs


from p3_center_state import P3CenterMixin

class WhisperSidecarForConditionalGeneration(P3CenterMixin, WhisperForConditionalGeneration):
    def __init__(self, config: WhisperConfig, 
                 sidecar_loc: int = 1, 
                 num_spks: int = 2, 
                 soft_prompt_len: int = 0,
                 target_asr: bool = False,
                 for_target_asr_eval: bool = False,
                 sidecar_residual: bool = True,
                 sidecar_activate: str = "sigmoid",
                 mask_norm: str = "none",
                 sidecar_gate_init: float = 0.1):
        super().__init__(config)
        self.num_spks = num_spks
        self.num_perms = math.perm(self.num_spks, self.num_spks)
        self.soft_prompt_len = soft_prompt_len
        self.model = WhisperSidecarModel(config, sidecar_loc, self.num_spks, target_asr, for_target_asr_eval,
                                         sidecar_residual=sidecar_residual,
                                         sidecar_activate=sidecar_activate,
                                         mask_norm=mask_norm,
                                         sidecar_gate_init=sidecar_gate_init)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.target_asr = target_asr

        if self.soft_prompt_len > 0:
            self.soft_prompt_embeds = nn.Embedding(self.soft_prompt_len, config.d_model)
            self.register_soft_prompt_hook()

        # Initialize weights and apply final processing
        self.post_init()
        self.init_p3_center()

    def set_mel_filters(self, mel_filters):   # LOSS_V3
        return self.model.encoder.set_mel_filters(mel_filters)

    def param_init(self, modules_to_save):
        """Re-initialise *only the newly added* modules.

        FIX(v2)  <-- THIS IS THE MAIN BUG FIX.

        The original implementation was::

            for name, params in self.named_parameters():
                if any(e in name for e in modules_to_save):
                    if 'bias' in name: torch.nn.init.constant_(params, 0.0)
                    if 'weight' in name or 'soft_prompt_embeds' in name:
                        torch.nn.init.normal_(params, mean=0.0, std=0.02)

        which matched parameters by *substring of the parameter name*.  Because
        the module names are ``sidecar``, ``sep_enc``, ``sep_dec`` this also
        matched every LayerNorm / GlobalLayerNorm / CumulativeLayerNorm weight
        (``sidecar.LayerN_S.weight``, ``sidecar.separation.*.norm_1.weight``,
        ...) and every PReLU slope.

        Consequences of the old behaviour:
          * LayerNorm gamma, which must be 1.0, became N(0, 0.02) ~ 0.02.
          * PReLU slope, which must be 0.25, became N(0, 0.02) ~ 0.02.
          * The separator has 24 convolutional blocks; every block is
            ``x + Sc_conv(PReLU(norm(dwconv(PReLU(norm(conv1x1(x)))))))``.
            With gamma ~ 0.02 the signal is attenuated by ~0.02 at each norm,
            and the residual branch (which carries the *only* useful signal)
            is also scaled down.  In fp16 the mask path underflows to 0.
          * With mask == 0 the sidecar output is exactly 0; the encoder then
            produced all-zero hidden states, the frozen decoder cross-attended
            to all-zero keys (uniform attention) and fell back to the language
            model prior -> the endless "THE PART OF THE PART OF ..." loops.
          * Gradient through a hard zero is zero, so the separator never
            recovered, no matter how many steps were trained.

        The fixed version is type-aware and only touches parameters that
        actually belong to the newly created modules (it exempts the frozen
        Whisper weights).
        """
        import math as _math
        new_module_prefixes = ('sidecar', 'sep_enc', 'sep_dec',
                               'soft_prompt_embeds', 'proj_target')

        def _is_new_module(module_name):
            parts = [p for p in module_name.split('.') if p]
            return any(part == p or part.startswith(p + '_') or part.startswith(p)
                       for part in parts for p in new_module_prefixes)

        for module_name, module in self.named_modules():
            if module_name == '' or not _is_new_module(module_name):
                continue
            # never touch modules that belong to the pretrained backbone
            if any(part in ('layers', 'encoder', 'decoder') for part in module_name.split('.')[:-1]
                   ) and not any(part in new_module_prefixes for part in module_name.split('.')):
                continue

            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, 'padding_idx', None) is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
            elif module_name.endswith('gen_masks'):
                # FIX(v2): the mask generator must start *neutral*.  With the
                # default sigmoid activation a zero output means every speaker
                # gets a mask of exactly 0.5 (the masks sum to 1), i.e. the
                # separator starts as a copy of its input instead of one
                # speaker winning / the other being muted at random.
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm,)):
                # CumulativeLayerNorm subclasses nn.LayerNorm
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif module.__class__.__name__ == 'GlobalLayerNorm':
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.PReLU):
                nn.init.constant_(module.weight, 0.25)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # anything else (bare nn.Parameter holders etc.) keeps the
            # explicit initialisation performed in __init__.

    def init_sidecar_weights(self, gate_init: float = 0.1):
        """Explicit, readable entry point for the corrected initialisation.

        Kept as a separate method so that ``finetune.py`` can call it without
        having to know about module names.  Safe to call multiple times.
        """
        self.param_init(['sidecar', 'sep_enc', 'sep_dec', 'soft_prompt_embeds', 'proj_target'])
        enc = getattr(self.model, 'encoder', None)
        if enc is not None and getattr(enc, 'sidecar_alpha', None) is not None:
            with torch.no_grad():
                enc.sidecar_alpha.fill_(float(gate_init))
        return self

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        target_speaker: Optional[torch.LongTensor] = None,
        clean_audios=None,          # LOSS_V3
        clean_lengths=None,         # post-preprocessing, pre-padding (B,S)
        mix_length=None,            # post-preprocessing, pre-padding (B,)
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and self.training and float(__import__('os').environ.get('SIDECAR_P3_BETA', '0') or 0) > 0:
            if clean_audios is None or clean_lengths is None or mix_length is None:
                raise ValueError('P3 training requires clean_audios and explicit valid lengths')
            if encoder_outputs is not None:
                raise ValueError('P3 requires a fresh encoder forward, not cached encoder_outputs')
            if self.target_asr or not self.model.encoder.sidecar_residual:
                raise ValueError('P3 requires residual separation without target-ASR prefixes')
        self.model.encoder.last_p3_audit = None

        if labels is not None:
            labels_input = labels.view(-1, self.num_spks, labels.shape[-1]).repeat(1, self.num_perms, 1).view(-1, labels.shape[-1])

            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels_input, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        outputs = self.model(
            input_features,
            clean_audios=clean_audios,        # LOSS_V3
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            target_speaker=target_speaker,
            pit=labels is not None,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            targets = labels_input.to(lm_logits.device)

            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size).to(torch.float32), targets.reshape(-1))
            # PIT loss
            loss, loss_min_idx= loss.view(-1, self.num_perms, self.num_spks, targets.shape[-1]).mean(dim=-1).mean(2).min(1)
            self.model.encoder.last_loss_min_idx = loss_min_idx.detach()    # P3: the real CE-PIT assignment
            loss = loss.mean()
            # ---- P3: target-vs-interferer margin on the stream ----------------
            _p3_beta = float(__import__('os').environ.get("SIDECAR_P3_BETA", "0") or 0)
            if _p3_beta > 0 and self.training:
                from p3_training import connected_p3_loss
                if clean_lengths is None or mix_length is None:
                    raise ValueError("P3 training requires explicit clean_lengths and mix_length")
                if not self.model.encoder.sidecar_residual:
                    raise ValueError("P3 requires post-residual streams")
                _p3_margin = self.validate_p3_center()
                _st = getattr(self.model.encoder, "last_full_streams", None)
                _cl = getattr(self.model.encoder, "last_clean_feats", None)
                assert _st is not None and _cl is not None, (
                    "P3: beta>0 but the stream/clean tensors are missing")
                _p3_loss, _audit = connected_p3_loss(
                    _st, _cl, clean_audios, clean_lengths, mix_length,
                    loss_min_idx, self.num_spks, self._p3_center, _p3_margin)
                self.model.encoder.last_p3_audit = _audit
                loss = loss + _p3_beta * _p3_loss
                if int(__import__('os').environ.get("SIDECAR_P3_VERBOSE", "0")):
                    print("P3 stats:", _audit['stats'])
            # ---- end P3 ---------------------------------------------------------
            # ---- LOSS_V3: separation losses -----------------------------------
            _sep_lam = float(__import__('os').environ.get("SIDECAR_SEP_LAM", "0") or 0)
            if _sep_lam > 0 and self.training:
                _sep_mode = __import__('os').environ.get("SIDECAR_SEP_MODE", "stream_cos")
                _src = (getattr(self.model.encoder, "last_sidecar_streams", None) if _sep_mode == "sep_cos"
                        else getattr(self.model.encoder, "last_full_streams", None))
                assert _src is not None, (
                    "LOSS_V3: SIDECAR_SEP_LAM>0 (%s) but the tensor is missing" % _sep_mode)
                assert _src.shape[0] % self.num_spks == 0, (
                    "LOSS_V3: streams %s not divisible by num_spks %d" % (tuple(_src.shape), self.num_spks))
                _bs = _src.shape[0] // self.num_spks
                _g = _src.float().mean(dim=2).view(_bs, self.num_spks, -1)     # (B, S, C)
                _g = _g - _g.mean(dim=(0, 1), keepdim=True)                    # drop shared direction
                _gn = torch.nn.functional.normalize(_g, dim=-1)
                _cos_sep = (_gn[:, 0] * _gn[:, 1]).sum(-1)
                loss = loss + _sep_lam * (1.0 + _cos_sep).mean()

            _rec_lam = float(__import__('os').environ.get("SIDECAR_REC_LAM", "0") or 0)
            if _rec_lam > 0 and self.training:
                _rec_mode = __import__('os').environ.get("SIDECAR_REC_MODE", "dist")
                _s = getattr(self.model.encoder, "last_streams_rec", None)
                _c = getattr(self.model.encoder, "last_clean_feats", None)
                _h = getattr(self.model.encoder, "last_mix_feats", None)
                _a = float(self.model.encoder.sidecar_alpha.detach().reshape(-1)[0])
                assert _s is not None and _c is not None and _h is not None, (
                    "LOSS_V3: SIDECAR_REC_LAM>0 but a tensor is missing "
                    "(streams=%s clean=%s mix=%s)" % (_s is not None, _c is not None, _h is not None))
                assert _s.shape[0] == _c.shape[0] == _h.shape[0], (
                    "LOSS_V3: shape mismatch streams=%s clean=%s mix=%s"
                    % (tuple(_s.shape), tuple(_c.shape), tuple(_h.shape)))
                _bs = _s.shape[0] // self.num_spks
                _sf = _s.float()                                               # (B*S, C, T) separator output
                _cf = _c.float().transpose(1, 2)                               # (B*S, C, T) clean
                _hf = _h.float().transpose(1, 2)                               # (B*S, C, T) mixture
                _v = _cf - _hf                                                 # required correction
                _w = _v.norm(dim=1)                                            # (B*S, T)
                if _rec_mode == "cos":
                    _hh = torch.nn.functional.normalize(_hf, dim=1)
                    _vp = _v - (_v * _hh).sum(1, keepdim=True) * _hh
                    _cs = (torch.nn.functional.normalize(_sf, dim=1)
                           * torch.nn.functional.normalize(_vp, dim=1)).sum(1)
                    _per = ((1.0 - _cs) * _w).sum(1) / (_w.sum(1) + 1e-6)
                else:
                    # distance between what the separator contributes (alpha*s) and
                    # what is needed (v).  alpha is part of the measured quantity.
                    _err = self.model.encoder.sidecar_alpha * _sf - _v         # (B*S, C, T)
                    _rel = (_err ** 2).sum(1) / ((_v ** 2).sum(1) + 1e-6)      # relative sq. error, O(1)
                    _per = (_rel * _w).sum(1) / (_w.sum(1) + 1e-6)
                _per = _per.view(_bs, self.num_spks)                           # (B, S)
                _perms = list(itertools.permutations(range(self.num_spks)))
                _pit = __import__('os').environ.get("SIDECAR_REC_PIT", "tied")
                if _pit == "independent":
                    _cand = torch.stack([_per[:, list(p)].mean(dim=1) for p in _perms], dim=1)
                    _rec = _cand.min(dim=1).values
                else:
                    _rec = None
                    for _pi, _p in enumerate(_perms):
                        _cand = _per[:, list(_p)].mean(dim=1)
                        _rec = _cand if _rec is None else torch.where(loss_min_idx == _pi, _cand, _rec)
                loss = loss + _rec_lam * _rec.mean()
            # ---- end LOSS_V3 ----------------------------------------------------
            if self.target_asr:
                if hasattr(outputs, "target_spk_logits"):
                    # get the best permutation for target spk according to loss_min_idx
                    best_perm = [list(list(itertools.permutations(range(self.num_spks)))[i]) for i in loss_min_idx]
                    target_spk_logits = outputs['target_spk_logits'][torch.arange(len(best_perm)).unsqueeze(1), best_perm]
                    target_spk_loss = loss_fct(target_spk_logits.to(torch.float32), target_speaker).mean() * 0.01
                    # acc = (torch.argmax(target_spk_logits, dim=1) == target_speaker).float().mean()
                    # print(acc, target_spk_loss * 100)
                else:
                    target_spk_loss = self.model.encoder.proj_target(torch.zeros(1, 1, outputs[0].shape[-1], device=outputs[0].device)).sum() * 0
                    target_spk_loss += self.model.encoder.proj_target_2(torch.zeros(1, 1, 150, device=outputs[0].device)).sum() * 0
                loss += target_spk_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def register_soft_prompt_hook(self):

        def decoder_soft_prompt_hook(module, input_ids, inputs_embeds):
            if inputs_embeds.shape[1] == 1:
                # generating
                input_ids = input_ids[0].reshape(-1)
                if (100 <= input_ids[0] < 100+self.soft_prompt_len) and torch.all(input_ids.eq(input_ids[0])):
                    # relace the fake token with the soft prompt
                    inputs_embeds[:] = self.soft_prompt_embeds((input_ids[0]-100).reshape(1)).squeeze(0)
            else:
                # training or teacher-forcing evaluation (and use_cache=False generation)
                # FIX(v2): the original code was
                #     inputs_embeds[:, 1:self.soft_prompt_len+1] = ...
                # which crashes whenever the sequence is shorter than
                # soft_prompt_len+1 (e.g. use_cache=False generation sends the
                # growing prefix, so the 2nd step has length 2).  We patch only
                # the positions that actually exist, and map fake prefix ids
                # (100..100+L-1) to the learned prompts when present.
                ids = input_ids[0] if isinstance(input_ids, (tuple, list)) else input_ids
                if ids is not None and ids.dim() == 2:
                    ids = ids[0]
                n = min(self.soft_prompt_len, max(inputs_embeds.shape[1] - 1, 0))
                if n > 0:
                    prompt_ids = torch.arange(n, device=inputs_embeds.device)
                    if ids is not None:
                        seg = ids.reshape(-1)[1:1 + n]
                        if seg.numel() == n:
                            fake = (seg >= 100) & (seg < 100 + self.soft_prompt_len)
                            prompt_ids = torch.where(fake, seg - 100, prompt_ids)
                    inputs_embeds[:, 1:1 + n] = self.soft_prompt_embeds(prompt_ids)
            return inputs_embeds

        self.model.decoder.embed_tokens.register_forward_hook(decoder_soft_prompt_hook)
