import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])

        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])

        self.register_buffer("bias",
                             torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(1, 1, config['n_ctx'],
                                                                                           config['n_ctx']))
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']

    def forward(self, x, mask):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        mask = mask.view(B, -1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.where(self.bias[:, :, :T, :T].bool(), att, self.masked_bias.to(att.dtype))
        att = att + mask
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], config['n_inner']),
            nn.GELU(),
            nn.Linear(config['n_inner'], config['n_embd']),
            nn.Dropout(config['resid_pdrop']),
        )

    def forward(self, inputs_embeds, attention_mask):
        x = inputs_embeds + self.attn(self.ln1(inputs_embeds), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class DecisionTransformer(nn.Module):

    def __init__(self, state_dim, act_dim, state_mean, state_std, action_tanh=False, K=10, max_ep_len=96, scale=2000,
                 target_return=4, return_dim=1):
        super(DecisionTransformer, self).__init__()
        self.device = "cpu"

        self.length_times = 3
        self.hidden_size = 64
        self.state_mean = state_mean
        self.state_std = state_std
        self.max_length = K
        self.max_ep_len = max_ep_len

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.target_return = target_return
        self.return_dim = int(return_dim)

        self.warmup_steps = 10000
        self.weight_decay = 0.0001
        self.learning_rate = 0.0001

        block_config = {
            "n_ctx": 1024,
            "n_embd": 64,
            "n_layer": 3,
            "n_head": 1,
            "n_inner": 512,
            "activation_function": "relu",
            "n_position": 1024,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1
        }

        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_return = torch.nn.Linear(self.return_dim, self.hidden_size)
        self.embed_reward = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

        self.predict_return = torch.nn.Linear(self.hidden_size, self.return_dim)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lambda steps: min((steps + 1) / self.warmup_steps, 1))

        self.init_eval()

    def _eval_device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            if isinstance(self.state_mean, torch.Tensor):
                return self.state_mean.device
            return torch.device("cpu")

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, **kwargs):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        rewards_embeddings = rewards_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(self.length_times)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.length_times * seq_length).to(stacked_inputs.dtype)

        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)

        x = x.reshape(batch_size, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])
        extras = {"fused_state_ctx": x[:, 1]} if kwargs.get("return_features", False) else None
        return state_preds, action_preds, return_preds, extras

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, self.return_dim)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim),
                             device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], self.return_dim),
                             device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length - rewards.shape[1], 1), device=rewards.device),
                 rewards],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds, reward_preds = self.forward(
            states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0, -1]

    def step(self, states, actions, rewards, dones, rtg, timesteps, attention_mask, sample_weights=None):
        rewards_target, action_target, rtg_target = torch.clone(rewards), torch.clone(actions), torch.clone(rtg)

        state_preds, action_preds, return_preds, reward_preds = self.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        valid = attention_mask.reshape(-1) > 0
        action_preds = action_preds.reshape(-1, act_dim)[valid]
        action_target = action_target.reshape(-1, act_dim)[valid]

        per_token_loss = torch.mean((action_preds - action_target) ** 2, dim=-1)
        if sample_weights is not None:
            sample_weights = sample_weights.to(device=attention_mask.device, dtype=torch.float32).view(-1, 1)
            token_weights = sample_weights.expand(-1, attention_mask.shape[1]).reshape(-1)[valid]
            token_weights = token_weights / token_weights.mean().clamp_min(1e-6)
            loss = torch.mean(per_token_loss * token_weights)
        else:
            loss = torch.mean(per_token_loss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item()

    def _format_target_return(self, target_return, device):
        if target_return is None:
            if self.return_dim == 1:
                values = [float(self.target_return)]
            else:
                values = [float(self.target_return)] + [0.0] * (self.return_dim - 1)
        elif isinstance(target_return, (list, tuple)):
            values = [float(x) for x in target_return]
        elif isinstance(target_return, torch.Tensor):
            values = [float(x) for x in target_return.detach().cpu().reshape(-1).tolist()]
        else:
            values = [float(target_return)]
        if len(values) < self.return_dim:
            values = values + [0.0] * (self.return_dim - len(values))
        values = values[:self.return_dim]
        return torch.tensor(values, dtype=torch.float32, device=device).reshape(1, 1, self.return_dim)

    def take_actions(self, state, target_return=None, pre_reward=None, pre_cost=None):
        self.eval()
        device = self._eval_device()
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(device=device, dtype=torch.float32)
            self.eval_target_return = self._format_target_return(target_return, device=device)
        else:
            assert pre_reward is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(device=device, dtype=torch.float32)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0)
            self.eval_rewards[-1] = pre_reward
            pred_return = self.eval_target_return[:, -1, :].clone()
            pred_return[:, 0] = pred_return[:, 0] - (pre_reward / self.scale)
            if self.return_dim > 1 and pre_cost is not None:
                pred_return[:, 1] = torch.clamp(pred_return[:, 1] - (pre_cost / self.scale), min=0.0)
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.unsqueeze(1)], dim=1)
            self.eval_timesteps = torch.cat(
                [
                    self.eval_timesteps,
                    torch.ones((1, 1), dtype=torch.long, device=device) * self.eval_timesteps[:, -1] + 1
                ],
                dim=1,
            )
        self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim, device=device)], dim=0)
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1, device=device)])

        action = self.get_action(
            (self.eval_states.to(dtype=torch.float32) - self.state_mean.to(device=device, dtype=torch.float32))
            / self.state_std.to(device=device, dtype=torch.float32),
            self.eval_actions.to(dtype=torch.float32, device=device),
            self.eval_rewards.to(dtype=torch.float32, device=device),
            self.eval_target_return.to(dtype=torch.float32, device=device),
            self.eval_timesteps.to(dtype=torch.long, device=device)
        )
        self.eval_actions[-1] = action
        action = action.detach().cpu().numpy()
        return action

    def init_eval(self):
        device = self._eval_device()
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.act_dim), dtype=torch.float32, device=device)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32, device=device)

        self.eval_target_return = None
        self.eval_timesteps = torch.tensor(0, dtype=torch.long, device=device).reshape(1, 1)

        self.eval_episode_return, self.eval_episode_length = 0, 0

    def save_net(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, "dt.pt")
        torch.save(self.state_dict(), file_path)

    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/dt_model.pth')

    def load_net(self, load_path="saved_model/DTtest", device='cpu'):
        file_path = load_path
        self.load_state_dict(torch.load(file_path, map_location=device))
        print(f"Model loaded from {self.device}.")
