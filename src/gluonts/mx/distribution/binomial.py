from typing import Dict, List, Optional, Tuple

import mxnet as mx
import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .deterministic import DeterministicOutput
from .distribution import (
    _sample_multiple,
    Distribution,
    getF,
    sigmoid,
    softplus,
)
from .distribution_output import DistributionOutput
from .mixture import MixtureDistributionOutput


class Binomial(Distribution):
    """
    Binomial distribution, i.e. the distribution of the number of successes in
    a fixed sequence of independent Bernoulli trials
    """

    is_reparameterizable = False

    @validated()
    def __init__(self, mu: Tensor, p: Tensor) -> None:
        self.mu = mu
        self.p = p
        self.n = self.mu / self.p

    @property
    def F(self):
        return getF(self.mu)

    @property
    def batch_shape(self) -> Tuple:
        return self.mu.shape

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def event_dim(self) -> int:
        return 0

    def log_prob(self, x: Tensor) -> Tensor:
        return x * self.F.log(self.p) + (self.mu / self.p - x) * self.F.log(
            1 - self.p
        )

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def stddev(self) -> Tensor:
        return self.n * self.p * (1 - self.p)

    def pmf(self, x):
        F = self.F
        px = (
            F.gamma(self.n)
            / (F.gamma(x) * F.gamma(self.n - x))
            * (self.p ** x)
            * (1 - self.p) ** (self.n - x)
        )
        # For non-integer n, n < k < n + 1 can output a negative probability
        px = F.where(px < 0, F.zeros_like(px), px)

        return px

    def sample(
        self, num_samples: Optional[int] = None, dtype=np.int32
    ) -> Tensor:
        def s(mu: Tensor, p: Tensor) -> Tensor:
            F = self.F
            n = mu / p
            max_n = int(F.ceil(mx.np.nanmax(n) + 1).asscalar())
            counts = mx.nd.array(range(max_n))
            pmf = self.pmf(counts)
            # For k > n + 1, probability is undefined. Practical to interpret as
            # 0 when building our cumulative table.
            pmf = F.where(mx.nd.contrib.isnan(pmf), F.zeros_like(pmf), pmf)

            # due to rounding, final sum could be greater than one
            cdf = F.clip(F.cumsum(pmf, 1), 0, 1)

            # due to rounding, final sum could be less than one, so we scale
            row_max = F.expand_dims(F.max(cdf, axis=1), 1)
            cdf = cdf / row_max

            # We get a sample from the uniform distribution and transform it to
            # a value from the Binomial CDF
            lessers = F.broadcast_lesser(
                F.sample_uniform(low=F.zeros_like(p), high=F.ones_like(p)),
                cdf,
            )

            # We create an array of potential counts of the same shape
            counts_arr = counts * F.ones_like(lessers)

            # We would like the minimum count satisfying the PMF, so we inflate
            # values above our random sample
            counts_arr = F.where(lessers, counts_arr, max_n * counts_arr + 1)

            return F.expand_dims(F.min(counts_arr, axis=1), 1)

        return _sample_multiple(
            s, mu=self.mu, p=self.p, num_samples=num_samples
        )

    @property
    def args(self) -> List:
        return [self.mu, self.p]


class BinomialOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"mu": 1, "p": 1}
    distr_cls: type = Binomial

    @classmethod
    def domain_map(cls, F, mu, p):
        epsilon = np.finfo(cls._dtype).eps  # machine epsilon

        mu = softplus(F, mu) + epsilon
        p = sigmoid(F, p)
        return mu.squeeze(axis=-1), p.squeeze(axis=-1)

    def distribution(
        self,
        distr_args,
        loc: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> Binomial:
        mu, p = distr_args
        if scale is None:
            return Binomial(mu, p)
        else:
            F = getF(mu)
            mu = F.broadcast_mul(mu, scale)
            return Binomial(mu, p, F)

    @property
    def event_shape(self) -> Tuple:
        return ()


class ZeroInflatedBinomialOutput(MixtureDistributionOutput):
    def __init__(self):
        super().__init__([BinomialOutput(), DeterministicOutput(0)])
