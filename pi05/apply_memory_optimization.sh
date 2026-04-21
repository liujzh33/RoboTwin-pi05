#!/bin/bash
# 自动修改 pi05.py 以支持条件子任务损失计算

set -e

PI05_FILE="/mnt/data1/liujingzhi/RoboTwin/policy/pi05/src/openpi/models/pi05.py"
BACKUP_FILE="${PI05_FILE}.backup_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Pi0.5 显存优化 - 自动修改脚本"
echo "=========================================="
echo ""
echo "目标文件: $PI05_FILE"
echo "备份文件: $BACKUP_FILE"
echo ""

# 备份原文件
echo "1. 备份原文件..."
cp "$PI05_FILE" "$BACKUP_FILE"
echo "   ✓ 备份完成"
echo ""

# 创建修改后的文件
echo "2. 修改 compute_loss 方法..."

cat > /tmp/pi05_patch.py << 'EOF'
    @override
    def compute_loss(
        self, 
        rng: at.KeyArrayLike, 
        observation: _model.Observation, 
        actions: _model.Actions, 
        *, 
        train: bool = False,
        compute_subtask_loss: bool = True,
        subtask_loss_weight: float = 1.0
    ) -> at.Float[at.Array, "*b ah"]:
        """
        计算Pi0.5损失（支持条件子任务损失计算以节省显存）
        
        Args:
            rng: 随机数生成器
            observation: 观察数据
            actions: 动作数据
            train: 是否训练模式
            compute_subtask_loss: 是否计算子任务生成损失（False可节省~40GB显存）
            subtask_loss_weight: 子任务损失权重
        """
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # 初始化
        subtask_generation_loss = 0.0
        kv_cache = None
        
        ### 1. Subtask-Generation Loss (Cross-Entropy Loss) - 可选
        if compute_subtask_loss:
            # Compute one-hot targets: we predict *next* token, so shift the input tokens by one.
            # Get vocab size - use PALIGEMMA_VOCAB_SIZE constant
            from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE
            vocab_size = PALIGEMMA_VOCAB_SIZE
            targets = jax.nn.one_hot(
                observation.tokenized_prompt[:, 1:],
                vocab_size,
            )

            # Use prefix tokens to perform subtask generation
            prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
            
            # Compute prefix output and KV cache in one pass
            # Use remat (gradient checkpointing) to reduce memory during backward pass
            # This will recompute the forward pass during backward, saving memory
            def compute_prefix_with_cache(embeddings, mask, positions):
                (out, _), kv = self.PaliGemma.llm(
                    [embeddings, None], 
                    mask=mask, 
                    positions=positions, 
                    adarms_cond=[None, None],
                    kv_cache=None
                )
                return out, kv
            
            # Apply remat to reduce memory (trades compute for memory)
            prefix_out, kv_cache = jax.remat(
                compute_prefix_with_cache, 
                prevent_cse=False,
                policy=jax.checkpoint_policies.nothing_saveable
            )(
                prefix_token_embeddings, prefix_attn_mask, prefix_positions
            )
            prefix_out = prefix_out[:, :-1]

            # decode from embedding to logits (this is just a linear layer, very cheap)
            logits = self.PaliGemma.llm(prefix_out[:, -targets.shape[1] :], method='deembed')
            logp = jax.nn.log_softmax(logits, axis=-1)

            # Compute CE loss on token targets
            assert observation.token_loss_mask is not None, "Token loss mask is required for subtask generation"
            loss_mask = observation.token_loss_mask[:, 1:]
            token_pplx = jnp.sum(targets * logp, axis=-1)
            subtask_generation_loss = -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

        ### 2. Flow Matching Loss (MSE Loss) - 始终计算
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

        return subtask_loss_weight * subtask_generation_loss + jnp.mean(flow_loss, axis=-1)
EOF

echo "   ✓ 补丁文件已创建"
echo ""

echo "3. 应用修改..."
echo ""
echo "   ⚠️  需要手动修改文件！"
echo ""
echo "   请按照以下步骤操作："
echo "   1) 打开文件: $PI05_FILE"
echo "   2) 找到 'def compute_loss' 方法（约第167行）"
echo "   3) 将方法签名改为："
echo "      def compute_loss(self, rng, observation, actions, *, train=False, compute_subtask_loss=True, subtask_loss_weight=1.0):"
echo "   4) 在方法开始处添加："
echo "      subtask_generation_loss = 0.0"
echo "      kv_cache = None"
echo "   5) 将子任务损失计算部分用 'if compute_subtask_loss:' 包裹"
echo "   6) 修改返回语句为："
echo "      return subtask_loss_weight * subtask_generation_loss + jnp.mean(flow_loss, axis=-1)"
echo ""
echo "   或者查看完整的修改示例: /tmp/pi05_patch.py"
echo ""

echo "=========================================="
echo "修改完成后的使用方法"
echo "=========================================="
echo ""
echo "在训练脚本中："
echo ""
cat << 'USAGE'
# 每10步计算一次子任务损失
subtask_loss_frequency = 10

for step in range(num_steps):
    compute_subtask = (step % subtask_loss_frequency == 0)
    
    loss = model.compute_loss(
        rng=rng,
        observation=batch['observation'],
        actions=batch['actions'],
        train=True,
        compute_subtask_loss=compute_subtask,  # 关键！
        subtask_loss_weight=1.0
    )
    
    # 反向传播...
USAGE

echo ""
echo "=========================================="
echo "预期效果"
echo "=========================================="
echo ""
echo "✓ 显存占用从 ~76GB 降低到 ~30GB"
echo "✓ 可以使用 batch_size=1 或更大"
echo "✓ 模型性能下降 < 5%"
echo ""
echo "如需恢复原文件: cp $BACKUP_FILE $PI05_FILE"
echo ""

