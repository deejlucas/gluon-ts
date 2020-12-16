from typing import Dict, List, Optional, Tuple

import mxnet as mx
import numpy as np

from gluonts.core.component import validated
from gluonts.mx import Tensor

from .deterministic import DeterministicOutput
from .distribution import _sample_multiple, Distribution, getF, sigmoid, softplus
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
        return x * self.F.log(self.p) + (self.mu / self.p - x) * self.F.log(1 - self.p)

    @property
    def mean(self) -> Tensor:
        return self.mu

    @property
    def stddev(self) -> Tensor:
        return self.n * self.p * self.q

    def sample(self, num_samples: Optional[int] = None, dtype=np.int32) -> Tensor:
        def s(n: Tensor, mu: Tensor) -> Tensor:
            F = self.F
            return F.broadcast_lesser(F.sample_uniform(low=F.zeros_like(self.p),
                                                       high=F.ones_like(self.p)),
                                      self.p)


        return _sample_multiple(s, mu=self.mu, n=self.p, num_samples=num_samples)

    @property
    def args(self) -> List:
        return [self.mu, self.n]


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
        mu, n = distr_args
        if scale is None:
            return Binomial(mu, n)
        else:
            F = getF(mu)
            mu = F.broadcast_mul(mu, scale)
            return Binomial(mu, n, F)

    @property
    def event_shape(self) -> Tuple:
        return ()


class ZeroInflatedBinomialOutput(MixtureDistributionOutput):
    def __init__(self):
        super().__init__([BinomialOutput(), DeterministicOutput(0)])
