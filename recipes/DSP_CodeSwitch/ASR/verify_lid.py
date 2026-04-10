#!/usr/bin/env python3
"""Quick verification: does LID NLLLoss decrease with lr=0.0001 on a tiny GRU?"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Simulate CausalPromptGenerator
gru = nn.GRU(input_size=1024, hidden_size=256, batch_first=True)
lid_head = nn.Linear(256, 3)
nn.init.normal_(lid_head.weight, std=0.01)
nn.init.zeros_(lid_head.bias)

optimizer = torch.optim.Adam(list(gru.parameters()) + list(lid_head.parameters()), lr=0.0001)

# Simulate frozen backbone features (detached)
fake_features = torch.randn(1, 100, 1024).detach()  # 100 frames
# Simulate word-level LID targets: 10 words
fake_lid_words = torch.tensor([[1, 1, 2, 2, 1, 0, 1, 2, 1, 1]])  # VI VI EN EN VI SIL VI EN VI VI

# Interpolate to 100 frames
lid_float = fake_lid_words.float().unsqueeze(1)  # [1, 1, 10]
lid_interp = F.interpolate(lid_float, size=100, mode="nearest").squeeze(1).long()  # [1, 100]

print("Step | Loss")
print("-----|--------")
for step in range(50):
    optimizer.zero_grad()
    gru_out, _ = gru(fake_features)
    gru_out = gru_out.clamp(-5, 5)
    logits = lid_head(gru_out)
    log_probs = F.log_softmax(logits.clamp(-10, 10), dim=-1)
    
    # NLL loss (same as sb.nnet.losses.nll_loss without length masking)
    loss = F.nll_loss(log_probs.view(-1, 3), lid_interp.view(-1))
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(gru.parameters()) + list(lid_head.parameters()), 1.0)
    optimizer.step()
    
    if step % 5 == 0:
        print(f"  {step:3d} | {loss.item():.4f}")

print(f"\nFinal loss: {loss.item():.4f}")
print(f"Expected random: {torch.log(torch.tensor(3.0)).item():.4f} (ln(3))")
print(f"Converging: {'YES ✓' if loss.item() < 0.5 else 'NO ✗'}")
