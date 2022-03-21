"""
This is an implementation of the latin square sampler described in 
``
Dau, Hai-Dang, and Nicolas Chopin. "Waste-free Sequential Monte Carlo." arXiv preprint arXiv:2011.02328 (2020)
``

We first implement a generic Adaptive SMC Sampler Class.
This class samples from a posterior `p` of the form
`p(x) \propto m(x) exp(-V(x))`, where m is the prior and exp(-V) is the likelihood
"""

import numpy as np
import random as rand
from scipy.stats import multinomial
from scipy import stats
from scipy.optimize import root_scalar
from scipy.special import gammaln, softmax

class SymmetricMetropolis:
    def __init__(self, kernel):
        self.kernel = kernel

    def draw(self, x, pdf):
        """
        Draw a new sample from the kernel and accept/reject using
        Metropolis-Hastings.
        """
        new = self.kernel(x) # Draw a new sample from kernel
        accept_prob = min(1, pdf(new) / pdf(x))
        if rand.uniform(0, 1) < accept_prob:
            return new
        else:
            return x

    def kfold_steps(self, x, k, pdf):
        """
        Apply the metropolis kernel k times
        """
        out = x
        for _ in range(k):
            out = self.draw(out, pdf)
        return out


class AdaptiveSMC:
    def __init__(
        self,
        initial_distribution,
        V,
        kernel,
        kernel_steps,
        particle_number,
        lambda_max=1,
    ):
        self.metropolis = SymmetricMetropolis(kernel)
        self.kernel_steps = kernel_steps
        self.particle_number = particle_number
        self.initial_distribution = initial_distribution # This is the prior
        self.V = V
        # Initializing useful quantities for later
        self.iteration = 0  # Tracks the t variable
        self.particles = None
        self.unnormalized_logweights = None
        self.normalized_logweights = None
        self.lambdas = [0]
        self.lambda_max = lambda_max # Maximum lambda, for a standard sampler this is 1
        self.ess_min = particle_number / 2  # Standard choice for ESS_min
                                            # Omiros & Chopin states that the performance
                                            # of the algorithm is pretty robust to this choice.

    def initial_sample(self):
        """
        Sample from prior.
        """
        n = self.particle_number
        return self.initial_distribution.rvs(size=n)

    def multinomial_draw(self):
        """
        Returns an array of indices
        """
        # so we can just apply to softmax to calculate normalized weights
        assert np.isclose(sum(self.normalized_weights), 1) # Sanity Check
        return multinomial(self.particle_number, self.normalized_weights).rvs()[0]

    def resample(self):  # TODO
        """
        Choose indices to resample and apply k-fold Metropolis
        kernels.
        """
        k = self.kernel_steps
        resample_indices = self.multinomial_draw()
        # Apply the metropolis step k times to each resampled particles
        new_particles = [None for _ in range(self.particle_number)]
        j = 0
        for i, n in enumerate(resample_indices):
            if n == 0:
                continue
            new_particles[j : j + n] = [
                self.metropolis.kfold_steps(
                    self.particles[i], k,
                    lambda x: np.exp(self.logscore(x, self.lambdas[-1]))
                )
                for _ in range(n)
            ]
            j += n

        self.particles = new_particles  # Update particles
        return

    def get_lambda(self):
        """
        Implement numerical root finding of optimal lambda parameter.
        Pg. 336 of Omiros / Chopin.

        Basically get the next lambda such that the resulting ESS
        is equal to the minimum ESS threshold.
        """
        f = lambda delta: (
            sum(
                (
                    np.exp(-delta * self.V(p))
                    for p in self.particles
                )
            ) ** 2
                ) / (
            sum(
                (
                    np.exp(-2 * delta * self.V(p))
                    for p in self.particles
                )
            )
        ) - self.ess_min
        try:
            delta = root_scalar(
                f, 
                bracket=[0, self.lambda_max - self.lambdas[-1]],
                method="brentq"
            ).root  # Not sure about this bracket argument
        except ValueError: # <- If a solution is not found (see algorithm 17.3 in book)
            delta = 1 - self.lambdas[-1]
        assert delta > 0, f"delta: {delta}"

        # We deviate a little from the book here ;
        # the latin square sampler requires that lambda can go above 1.
        # So we replace 1 with self.lambda_max
        if delta < self.lambda_max - self.lambdas[-1]:
            self.lambdas.append(self.lambdas[-1] + delta)
        else:
            self.lambdas.append(self.lambda_max)
        return

    def logscore(self, x, l):
        """
        Calculate intermediate distribution without the normalising constant.
        """
        return -(l * self.V(x)) - self.initial_distribution.logpdf(x)

    def calc_weight(self, resample=False):
        if self.iteration == 0:
            self.unnormalized_logweights = [
                self.initial_distribution.logpdf(p)
                for p in self.particles
            ]
        else:
            lambda_t = self.lambdas[-1]
            lambda_t_minus_one = self.lambdas[-2]
            # If resampling happened, add nothing ; else we are just doing sequential
            # importance sampling, so multiply previous weights
            w_hat = self.unnormalized_logweights if resample else np.zeros(self.particle_number)
            logweights = [
                w_hat[i] + (self.logscore(p, lambda_t) - self.logscore(p, lambda_t_minus_one))
                for (i, p) in enumerate(self.particles)
            ]
            self.unnormalized_logweights = logweights
        self.normalized_weights = softmax(self.unnormalized_logweights)
        return

    def ess(self):
        """
        Calculate the effective sample size.
        """
        return 1 / sum((W**2 for W in self.normalized_weights))

    def run(self):
        while self.lambdas[-1] < self.lambda_max:
            if self.iteration == 0:
                self.particles = self.initial_sample() # Start with inital set of particles
            else:
                self.resample() # Do resampling and metropolis kernel steps
            self.get_lambda() # Calculate a new lambda by solving for lambda in ess - ess_min = 0
            self.calc_weight() # Recalculate weights
            print(f"Iteration {self.iteration} done!")
            print(f"λ : {self.lambdas[-1]}")

            self.iteration += 1


def sample(d):
    """
    Sample a permutation of 0, 1, ..., d-1
    """
    return np.random.choice(range(d), size=d, replace=False)


def sample_matrix(d):
    """
    Sample a d x d matrix where every row is a permutation of
    0, 1, ..., d-1
    """
    return np.matrix([sample(d) for _ in range(d)])


def latin_kernel(x):
    """
    Takes a d x d matrix and selects a row i
    and two columns j1 and j2 at random.
    then it swaps the values of x[i,j1] and x[i,j2]
    """
    d = x.shape[1]
    m = x.copy()
    i, j1, j2 = np.random.randint(low=0, high=d - 1, size=3)
    x1 = x[i, j1]
    x2 = x[i, j2]
    m[i, j1] = x2
    m[i, j2] = x1
    return m


def V_latin(x):
    """
    Calculate score of latin square.
    """
    d = x.shape[1]
    return (
        sum(
            (
                sum((x[i, j] == 1) for i in range(d)) ** 2
                for l in range(d)
                for j in range(d)
            )
        )
        - d ** 2
    )


class UniformPermutationMatrix(stats.rv_discrete):
    """
    Distribution of UniformPermutationMatrix
    """
    def __init__(self, d, seed=None):
        super().__init__(seed=seed)
        self.d = d

    def rvs(self, size=1):
        """
        Implement random sampling
        """
        return np.array([sample_matrix(self.d) for _ in range(size)])

    def logpdf(self, x):
        """
        Compute the log of (d!)**d
        """
        d = self.d
        return d * gammaln(d + 1)



class LatinSquareSMC(AdaptiveSMC):
    EPSILON = 1e-16

    def __init__(self, d, kernel_steps, particle_number):
        initial_distribution = UniformPermutationMatrix(d)
        V = V_latin
        kernel = latin_kernel
        super().__init__(
            initial_distribution,
            V,
            kernel,
            kernel_steps,
            particle_number,
            lambda_max = initial_distribution.logpdf(None) - np.log(self.EPSILON)
        )
            


## TESTING
smc = LatinSquareSMC(
    d=4,
    kernel_steps=50,
    particle_number=300
)
smc.run()
