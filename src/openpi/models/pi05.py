"""
PI0.5 model with subtask generation capability.
Based on pi0.py but adds subtask generation loss and sample_low_level_task method.
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision."""
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # Due to vmap, this is operating in a single example (not batch level).
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi05(_model.BaseModel):
    """
    PI0.5 model with subtask generation.
    
    Key differences from Pi0:
    1. Adds subtask generation loss (Cross-Entropy) in addition to flow matching loss
    2. Language tokens use autoregressive attention (ar_mask=True) for subtask generation
    3. Supports sample_low_level_task() method to generate subtasks
    """
    
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config  # Save config for accessing n_obs_steps
        self.pi05 = config.pi05
        if not self.pi05:
            raise ValueError("Pi05 model requires pi05=True in config")
            
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        # Get fake observation for image encoder initialization
        fake_image = next(iter(config.fake_obs().images.values()))
        # If multi-frame mode (n_obs_steps > 1), take the first frame for initialization
        # Image encoder expects [batch, h, w, c], but fake_obs returns [batch, n_obs_steps, h, w, c] when n_obs_steps > 1
        if self.config.n_obs_steps > 1 and fake_image.ndim == 5:
            fake_image = fake_image[:, 0]  # Take first frame: [batch, n_obs_steps, h, w, c] -> [batch, h, w, c]
        img.lazy_init(fake_image, train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Progress estimation head: uses pooled PaliGemma prefix features.
        hidden_dim = paligemma_config.width
        self.progress_mlp_in = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.progress_mlp_out = nnx.Linear(hidden_dim, 1, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        
        # Check if multi-frame stacking is enabled (n_obs_steps > 1)
        n_obs_steps = self.config.n_obs_steps
        if n_obs_steps > 1:
            # Multi-frame mode: process each frame and concatenate tokens
            # embed images for each time step
            for name in obs.images:
                image = obs.images[name]  # Expected shape: [batch, n_obs_steps, h, w, c]
                image_mask = obs.image_masks[name]  # Expected shape: [batch, n_obs_steps]
                
                # Handle both 4D (single-frame) and 5D (multi-frame) inputs for robustness
                if image.ndim == 4:
                    # Single-frame input in multi-frame mode: expand to 5D
                    image = image[:, np.newaxis, ...]  # [batch, 1, h, w, c]
                    if image_mask.ndim == 1:
                        image_mask = image_mask[:, np.newaxis]  # [batch, 1]
                
                batch_size, num_frames = image.shape[:2]
                # Reshape to process all frames at once: [batch * n_obs_steps, h, w, c]
                image_flat = einops.rearrange(image, "b t h w c -> (b t) h w c")
                image_tokens_flat, _ = self.PaliGemma.img(image_flat, train=False)
                # Reshape back: [batch, n_obs_steps, num_tokens_per_image, emb_dim]
                num_tokens_per_image = image_tokens_flat.shape[1]
                emb_dim = image_tokens_flat.shape[2]
                image_tokens = einops.rearrange(
                    image_tokens_flat, 
                    "(b t) s d -> b (t s) d", 
                    b=batch_size, 
                    t=num_frames, 
                    s=num_tokens_per_image
                )
                
                tokens.append(image_tokens)
                # Expand mask for each token: [batch, n_obs_steps] -> [batch, n_obs_steps * num_tokens]
                expanded_mask = einops.repeat(
                    image_mask,
                    "b t -> b (t s)",
                    s=num_tokens_per_image,
                )
                input_mask.append(expanded_mask)
                # image tokens attend to each other (no autoregressive)
                ar_mask += [False] * (num_frames * num_tokens_per_image)
        else:
            # Single-frame mode: original behavior
            # embed images
            for name in obs.images:
                image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

                tokens.append(image_tokens)
                input_mask.append(
                    einops.repeat(
                        obs.image_masks[name],
                        "b -> b s",
                        s=image_tokens.shape[1],
                    )
                )
                # image tokens attend to each other
                ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            # Handle multi-frame mode: tokenized_prompt may have shape [batch, n_obs_steps, l] instead of [batch, l]
            # Compress it to [batch, l] for PaliGemma.llm (use first frame, all frames are identical)
            tokenized_prompt = obs.tokenized_prompt
            tokenized_prompt_mask = obs.tokenized_prompt_mask
            if tokenized_prompt.ndim == 3:
                # Compress: [batch, n_obs_steps, l] -> [batch, l] (use first frame)
                tokenized_prompt = tokenized_prompt[:, 0, :]
            if tokenized_prompt_mask is not None:
                # If mask is [batch, n_obs_steps, l], compress to [batch, l]
                if tokenized_prompt_mask.ndim == 3:
                    tokenized_prompt_mask = tokenized_prompt_mask[:, 0, :]
            
            tokenized_inputs = self.PaliGemma.llm(tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(tokenized_prompt_mask)
            # KEY CHANGE: Use autoregressive attention for language tokens (for subtask generation)
            ar_mask += [True] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        # time MLP (for adaRMS)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        ### 1. Subtask-Generation Loss (Cross-Entropy Loss)
        # Compute one-hot targets: we predict *next* token, so shift the input tokens by one.
        # Get vocab size - use PALIGEMMA_VOCAB_SIZE constant
        from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE
        vocab_size = PALIGEMMA_VOCAB_SIZE
        
        # Handle multi-frame mode: tokenized_prompt may have shape [batch, n_obs_steps, l] instead of [batch, l]
        tokenized_prompt = observation.tokenized_prompt
        if tokenized_prompt.ndim == 3:
            # Compress: [batch, n_obs_steps, l] -> [batch, l] (use first frame)
            tokenized_prompt = tokenized_prompt[:, 0, :]
        
        targets = jax.nn.one_hot(
            tokenized_prompt[:, 1:],
            vocab_size,
        )

        # Use prefix tokens to perform subtask generation
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None], 
            mask=prefix_attn_mask, 
            positions=prefix_positions, 
            adarms_cond=[None, None],
            kv_cache=None
        )
        prefix_out = prefix_out[:, :-1]

        # decode from embedding to logits
        logits = self.PaliGemma.llm(prefix_out[:, -targets.shape[1] :], method='deembed')
        logp = jax.nn.log_softmax(logits, axis=-1)

        # Compute CE loss on token targets
        assert observation.token_loss_mask is not None, "Token loss mask is required for subtask generation"
        token_loss_mask = observation.token_loss_mask
        if token_loss_mask.ndim == 3:
            # Compress: [batch, n_obs_steps, l] -> [batch, l] (use first frame)
            token_loss_mask = token_loss_mask[:, 0, :]
        loss_mask = token_loss_mask[:, 1:]
        token_pplx = jnp.sum(targets * logp, axis=-1)
        subtask_generation_loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

        ### 2. Flow Matching Loss (MSE Loss)
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        attn_mask = attn_mask[:, -suffix_tokens.shape[1]:, :]  # Q is [B, action_dim, ...], KV is full length
        positions = jnp.cumsum(input_mask, axis=1) - 1
        positions = positions[:, -suffix_tokens.shape[1]:]
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], kv_cache=kv_cache, mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # Calculate flow loss
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # Progress estimation loss (trained on clean multi-task data).
        # prefix_out: [batch, seq_len, hidden_dim]
        pooled_prefix = jnp.mean(prefix_out, axis=1)
        h = self.progress_mlp_in(pooled_prefix)
        h = nnx.swish(h)
        progress_logits = self.progress_mlp_out(h)[..., 0]
        pred_progress = jax.nn.sigmoid(progress_logits)

        progress_label = observation.progress_label
        if progress_label is not None:
            if isinstance(progress_label, np.ndarray):
                progress_label = jnp.asarray(progress_label)
            progress_label = progress_label.reshape(pred_progress.shape)
            progress_loss = jnp.mean(jnp.square(pred_progress - progress_label))
            # from jax import debug as jax_debug
            # jax_debug.print(
            #     "progress_step: loss={pl:.4f}, label_mean={lm:.3f}, pred_mean={pm:.3f}",
            #     pl=progress_loss,
            #     lm=jnp.mean(progress_label),
            #     pm=jnp.mean(pred_progress),
            # )
        else:
            progress_loss = 0.0
        
      

        return subtask_generation_loss + jnp.mean(flow_loss, axis=-1) +  progress_loss

    @override
    def sample_low_level_task(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        max_decoding_steps: int = 20,
        PALIGEMMA_EOS_TOKEN: int = 1,
        temperature: float = 0.0,
    ) -> tuple[at.Int[at.Array, "b max_steps"], tuple, at.Bool[at.Array, "b prefix_len+max_steps"], at.Bool[at.Array, "prefix_len+max_steps"]]:
        """
        Generate low-level subtask tokens autoregressively.
        
        Args:
            rng: Random key
            observation: Model observation containing images and prompt
            max_decoding_steps: Maximum number of tokens to decode
            PALIGEMMA_EOS_TOKEN: EOS token ID (usually 1)
            temperature: Sampling temperature (0.0 for greedy decoding)
            
        Returns:
            output_tokens: Generated subtask tokens [batch, max_decoding_steps]
            kv_cache: KV cache for subsequent action generation
            mask: Token mask [batch, prefix_len + max_decoding_steps]
            ar_mask: Autoregressive mask [prefix_len + max_decoding_steps]
        """
        batch_size = observation.tokenized_prompt.shape[0]
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # left to right align all input token sequences
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # first fill KV cache with a forward pass of the prefix
        # pad attention mask to set the size of the KV cache (prefill_size + max_decoding_steps)
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings, None], mask=prefix_attn_mask, positions=prefix_positions, adarms_cond=[None, None]
        )

        # prepare decoding -- final logit decodes the first token
        last_logit = prefix_out[:, -1:]
        last_logit = self.PaliGemma.llm(last_logit, method='deembed')
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=jnp.int32)

        def step(carry):
            rng, last_logit, output_tokens, cache, _, step = carry

            # Sample token from last logit
            # Split RNG for this step
            rng, rng_step = jax.random.split(rng)
            token = jax.lax.cond(
                temperature > 0.0,
                lambda _: jax.random.categorical(rng_step, last_logit / temperature, axis=-1),
                lambda _: jnp.argmax(last_logit, axis=-1),
                operand=None,
            )
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping --> stop if all batch elements have EOS token
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            # Decode one step
            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefill_len[:, None] + step
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )

            (prefix_out, _), kv_cache = self.PaliGemma.llm(
                [token_embedding, None], mask=mask, positions=positions, adarms_cond=[None, None], kv_cache=cache
            )
            logits = self.PaliGemma.llm(prefix_out[:, -1:], method='deembed')
            last_logit = logits

            return rng, last_logit, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # Use lax.while_loop so we can jit the full decoding loop.
        _, _, output_tokens, kv_cache, _, _ = jax.lax.while_loop(
            cond, step, (rng, last_logit, output_tokens, kv_cache, False, 0)
        )

        # Callback function to decode and print subtask text (works inside JIT)
        def print_subtask(tokens):
            import sentencepiece
            from openpi.shared import download
            # Initialize tokenizer inside callback
            path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
            with path.open("rb") as f:
                sp_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
            
            for i in range(tokens.shape[0]):
                # Convert JAX array to numpy and remove padding
                token_array = np.array(tokens[i], dtype=np.int32)
                print(f"[DEBUG] Token array: {token_array}")
                non_padding = token_array[token_array != 0]
                print(f"[DEBUG] Non-padding tokens: {non_padding}")
                # Decode to text
                try:
                    subtask_text = sp_tokenizer.decode(non_padding.tolist())
                    print(f"[Pi0.5 Subtask]: {subtask_text}")
                    print(f"[DEBUG] Type of subtask_text: {type(subtask_text)}, repr: {repr(subtask_text)}")
                except Exception as e:
                    print(f"[ERROR] Decode failed: {e}")
        
        # Use jax.debug.callback to call Python code from JIT
        jax.debug.callback(print_subtask, output_tokens)
        jax.debug.print("[Pi0.5 Subtask] Debug Tokens: {tokens}", tokens=output_tokens)

        mask = jnp.concatenate([prefix_mask, (output_tokens != 0).astype(jnp.bool_)], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, jnp.ones(max_decoding_steps, dtype=jnp.bool_)], axis=0)
        
        return output_tokens, kv_cache, mask, ar_mask

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        assert batch_size == 1, "Batch size must be 1 for sample_actions, subtask can be of different length"
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Generate low-level subtask and get KV cache with subtask tokens
        output_tokens, kv_cache, prefix_mask, prefix_ar_mask = self.sample_low_level_task(
            rng, observation, max_decoding_steps=20, PALIGEMMA_EOS_TOKEN=1, temperature=0.0
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            query_attn_mask = full_attn_mask[:, -suffix_tokens.shape[1]:, :]  # [B, suffix_len, prefix_len + suffix_len]
            
            assert query_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_mask.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=query_attn_mask,
                positions=positions,
                kv_cache=kv_cache,  # kv_cache is not updated during multiple denoising steps
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

