"""EMR-WGAN, the generator the tutorial demonstrates (its Figure 1).

The architecture is the one drawn in the paper: a generator of fully connected
layers with **batch normalization**, a discriminator of fully connected layers
with **layer normalization**, and a Wasserstein divergence between them. The
normalization asymmetry is not incidental and is the detail most often lost when
this model is reimplemented -- batch norm in the critic would let it compare
samples *within* a batch, which leaks batch composition into a score that is
supposed to be per-sample, and it is incompatible with the gradient penalty for
the same reason. Layer norm is the standard substitute (Gulrajani et al., the
paper's reference 28).

Two constraints from the tutorial's Model Training section are enforced here:

- **One-hot preservation.** "A SoftMax layer should be attached to the output of
  the generator to preserve the one-hot constraint." Each categorical block gets
  its own SoftMax; everything else gets a sigmoid, which is what puts the whole
  output in [0,1] to match the preprocessed matrix.
- **No record-level clinical constraints.** The tutorial deliberately declines to
  add the penalty terms of its reference 36 -- "we ... refrain from imposing
  record-level constraints during model training to showcase the phenomenon of
  clinical knowledge violation in results". This implementation follows that,
  and `evaluate.py` measures the violations that result.

The tutorial is equally clear about what training does *not* give you: "the model
checkpoint that corresponds to the highest quality of the synthetic data is not
necessarily the one with the lowest training loss", and "there is no monotonic
relationship between training loss and the quality of synthetic data". So
`train` keeps checkpoints along the trajectory rather than the final weights, and
`run_tutorial.py` scores each one instead of trusting the loss curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from ml.synthetic.preprocess import MatrixSpec

# Adam settings from the WGAN-GP paper the tutorial cites; beta1=0.5 is the
# usual GAN choice and beta1=0.9 destabilises the critic here.
ADAM_BETAS = (0.5, 0.9)
GRADIENT_PENALTY_WEIGHT = 10.0
# EMR-WGAN updates the critic twice per generator step rather than the WGAN-GP
# default of 5: the paper reports the lighter ratio converges faster on EHR
# matrices without loss of quality.
CRITIC_STEPS = 2


@dataclass
class TrainConfig:
    noise_dim: int = 128
    generator_hidden: tuple[int, ...] = (256, 256)
    critic_hidden: tuple[int, ...] = (256, 256)
    batch_size: int = 256
    epochs: int = 300
    learning_rate: float = 2e-4
    checkpoint_every: int = 50
    seed: int = 0


@dataclass
class TrainResult:
    generator: Generator
    checkpoints: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)


class Generator(nn.Module):
    """Noise (+ optional label) -> a row of the preprocessed EHR matrix, in [0,1].

    `block_slices` are the one-hot column spans; each is SoftMax'd separately so
    exactly one level of each categorical variable is active, as the tutorial
    requires. Columns outside every block are sigmoid'd.

    `condition_dim > 0` selects the tutorial's **conditional** training paradigm,
    which "uses the label variables to guide model training, as well as the
    generation of the synthetic EHR data, which enables the control over the
    categories of the generated data in terms of the label variables". The label
    is concatenated to the noise vector, which is the mechanism the paper
    describes: "incorporating the label variables as extra input of the neural
    networks of the generator and the discriminator".
    """

    def __init__(
        self,
        noise_dim: int,
        n_columns: int,
        hidden: tuple[int, ...],
        block_slices: list[tuple[int, int]],
        condition_dim: int = 0,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.block_slices = block_slices

        layers: list[nn.Module] = []
        width = noise_dim + condition_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU(inplace=True)]
            width = size
        layers.append(nn.Linear(width, n_columns))
        self.net = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        inputs = noise if condition is None else torch.cat([noise, condition], dim=1)
        logits = self.net(inputs)
        out = torch.sigmoid(logits)
        for start, stop in self.block_slices:
            # Written into a clone rather than in place: an in-place write onto a
            # tensor that autograd needs for the sigmoid backward pass raises a
            # version-counter error, and cloning is cheaper than restructuring
            # the output into per-block concatenation.
            out = out.clone()
            out[:, start:stop] = torch.softmax(logits[:, start:stop], dim=1)
        return out

    @torch.no_grad()
    def sample(
        self,
        n: int,
        device: torch.device | None = None,
        condition: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw `n` synthetic rows, optionally at a caller-chosen label composition.

        Passing `condition` is how the tutorial's "determine composition" step
        (its Figure 3) is actually exercised: the caller decides how many
        positives the synthetic cohort contains rather than accepting whatever
        the generator's marginal happens to be. That matters here because the
        label is a 4%-prevalence concept and an unconditional generator drops it.
        """
        device = device or next(self.parameters()).device
        was_training = self.training
        # BatchNorm in eval mode uses running statistics, so a sample of 1 works
        # and -- more importantly -- generation stops depending on how many rows
        # happen to be requested at once.
        self.eval()
        noise = torch.randn(n, self.noise_dim, device=device)
        cond = None
        if self.condition_dim:
            if condition is None:
                raise ValueError("conditional generator requires a `condition` array")
            cond = torch.tensor(condition, dtype=torch.float32, device=device)
        out = self(noise, cond).cpu().numpy()
        if was_training:
            self.train()
        return out


class Critic(nn.Module):
    """Wasserstein critic: a real-valued score, no sigmoid, layer-normalized."""

    def __init__(self, n_columns: int, hidden: tuple[int, ...], condition_dim: int = 0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_columns + condition_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.LayerNorm(size), nn.LeakyReLU(0.2, inplace=True)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        inputs = x if condition is None else torch.cat([x, condition], dim=1)
        return self.net(inputs)


def _gradient_penalty(
    critic: Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    condition: torch.Tensor | None = None,
) -> torch.Tensor:
    """The WGAN-GP penalty: gradient norm pulled to 1 on real-fake interpolates.

    Only the data half is interpolated. The condition is the same on both sides
    of a given pair by construction, so interpolating it would be a no-op, and
    taking the gradient with respect to it would penalise the critic for its
    sensitivity to the label -- exactly the sensitivity conditional training is
    trying to build.
    """
    alpha = torch.rand(real.size(0), 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolated, condition)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    norm = gradients.norm(2, dim=1)
    return ((norm - 1.0) ** 2).mean()


def train(
    values: np.ndarray,
    spec: MatrixSpec,
    config: TrainConfig | None = None,
    conditions: np.ndarray | None = None,
) -> TrainResult:
    """Train EMR-WGAN on a preprocessed matrix, keeping checkpoints as it goes.

    Passing `conditions` -- an (n_rows, k) array of label indicators aligned to
    `values` -- selects the tutorial's conditional paradigm. Leave it None for
    the nonconditional one, which is what the tutorial's own demonstration uses.

    Returns the final generator plus a `checkpoints` dict of epoch -> state dict.
    The caller is expected to *score* those checkpoints rather than assume the
    last one is best, for the reason the tutorial gives: overtraining a GAN can
    degrade synthetic data quality, and the loss curve does not reveal it.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = torch.device("cpu")

    data = torch.tensor(values, dtype=torch.float32, device=device)
    n_rows, n_columns = data.shape
    condition_dim = 0
    cond_all: torch.Tensor | None = None
    if conditions is not None:
        if len(conditions) != n_rows:
            raise ValueError(f"conditions has {len(conditions)} rows against {n_rows} in values")
        cond_all = torch.tensor(np.asarray(conditions, dtype=np.float32), device=device)
        condition_dim = cond_all.shape[1]

    # BatchNorm needs at least 2 rows per batch to compute a variance, and a
    # trailing batch of 1 would crash training several epochs in. Capping the
    # batch at the dataset size keeps small cohorts (the stay-level matrix is
    # 140 rows) working without a special case.
    batch_size = max(2, min(config.batch_size, n_rows))

    generator = Generator(
        config.noise_dim, n_columns, config.generator_hidden, spec.block_slices(), condition_dim
    ).to(device)
    critic = Critic(n_columns, config.critic_hidden, condition_dim).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, betas=ADAM_BETAS)
    opt_c = torch.optim.Adam(critic.parameters(), lr=config.learning_rate, betas=ADAM_BETAS)

    result = TrainResult(generator=generator)
    generator.train()

    for epoch in range(1, config.epochs + 1):
        order = torch.randperm(n_rows, device=device)
        epoch_c, epoch_g, n_batches = 0.0, 0.0, 0

        for start in range(0, n_rows - 1, batch_size):
            index = order[start : start + batch_size]
            real = data[index]
            if real.size(0) < 2:
                continue
            cond = None if cond_all is None else cond_all[index]

            for _ in range(CRITIC_STEPS):
                noise = torch.randn(real.size(0), config.noise_dim, device=device)
                fake = generator(noise, cond).detach()
                penalty = _gradient_penalty(critic, real, fake, device, cond)
                loss_c = critic(fake, cond).mean() - critic(real, cond).mean()
                loss_c = loss_c + GRADIENT_PENALTY_WEIGHT * penalty
                opt_c.zero_grad(set_to_none=True)
                loss_c.backward()
                opt_c.step()

            noise = torch.randn(real.size(0), config.noise_dim, device=device)
            loss_g = -critic(generator(noise, cond), cond).mean()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            epoch_c += float(loss_c.item())
            epoch_g += float(loss_g.item())
            n_batches += 1

        if n_batches:
            result.history.append(
                {
                    "epoch": float(epoch),
                    "critic_loss": epoch_c / n_batches,
                    "generator_loss": epoch_g / n_batches,
                }
            )

        if epoch % config.checkpoint_every == 0 or epoch == config.epochs:
            result.checkpoints[epoch] = {
                k: v.detach().clone() for k, v in generator.state_dict().items()
            }

    return result


def generator_from_checkpoint(
    state: dict[str, torch.Tensor],
    spec: MatrixSpec,
    config: TrainConfig,
    condition_dim: int = 0,
) -> Generator:
    """Rebuild a generator at a saved checkpoint, ready to sample."""
    generator = Generator(
        config.noise_dim,
        spec.n_columns,
        config.generator_hidden,
        spec.block_slices(),
        condition_dim,
    )
    generator.load_state_dict(state)
    generator.eval()
    return generator
