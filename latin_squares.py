# TODO
# 1. Put a bunch of assert statements to rule out possible mistakes
# 2. Check that the UniformPermutationMatrix class makes sense with respect to what is 
#    in the waste-free smc paper. Not sure about the sampler

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

SEED = 42
np.random.seed(42)

class Metropolis:
    def __init__(self, kernel, kernel_steps):
        self.kernel = kernel
        self.kernel_steps = kernel_steps # Number of times MCMC
                                         # is applied to every particle

    def draw(self, x, pdf):
        """
        Draw a new sample from the kernel and accept/reject using
        Metropolis.
        """
        new = self.kernel(x) # Draw a new sample from kernel
        accept_prob = min(1, pdf(new) / pdf(x))
        if rand.uniform(0, 1) < accept_prob:
            return new
        else:
            return x

    def kfold_steps(self, x, pdf):
        """
        Apply the metropolis step `self.kernel_steps` times
        """
        out = x
        for _ in range(self.kernel_steps):
            out = self.draw(out, pdf)
        return out

class History:
    """
    Class that holds the values of weights at each iteration.
    Is also used to calculate the normalising constant
    """

class AdaptiveSMC:
    """
    Algorithm 17.3 of Papaspiliopoulos / Chopin Book
    """
    def __init__(
        self,
        initial_distribution,
        V,
        kernel,
        kernel_steps,
        particle_number,
        lambda_max=1,
        ess_min_ratio = 1/2
    ):
        self.metropolis = Metropolis(kernel, kernel_steps)
        self.particle_number = particle_number
        self.initial_distribution = initial_distribution # This is the distribution that you start with.
        self.V = V
        # Initializing useful quantities for later
        self.iteration = 0  # Tracks the t variable
        self.particles = None
        self.unnormalized_logweights = None
        self.normalized_logweights = None
        self.lambdas = [0]
        self.lambda_max = lambda_max # Maximum lambda, for a standard sampler this is 1
        self.ess_min = particle_number * ess_min_ratio  # Papaspiliopoulos & Chopin states that the performance
                                                        # of the algorithm is pretty robust to this choice.

        self.logLt = 0. # This will hold 

    def initial_sample(self):
        """
        Sample from prior.
        """
        n = self.particle_number
        return self.initial_distribution.rvs(size=n)

    def multinomial_draw(self):
        """
        Returns an array of indices.

        For example:
        if we have particles 5 particles,
        then we might draw
        [1,0,0,2,2]
        which means we will resample particle 1 once
        and particles 4 and 5 three times.
        """
        assert np.isclose(sum(self.normalized_weights), 1) # Sanity Check
        return multinomial(self.particle_number, self.normalized_weights).rvs()[0]

    def resample(self):  # TODO
        """
        Choose indices to resample and apply k-fold Metropolis
        kernels.
        """
        resample_indices = self.multinomial_draw()
        # Apply the metropolis step k times to each resampled particles
        new_particles = [None for _ in range(self.particle_number)] # Initialize vector of new particles
        print("Doing Metropolis Resampling...")
        j = 0
        # n = number of times the particle has been resampled
        for i, n in enumerate(resample_indices):
            if n == 0: # If the particle is not being resampled at all
                continue
            # Apply k metropolis steps to this particle n times
            new_particles[j : j + n] = [
                self.metropolis.kfold_steps(
                    self.particles[i],
                    lambda x: np.exp(self.logscore(x, self.lambdas[-1]))
                )
                for _ in range(n)
            ]
            j += n

        self.particles = new_particles  # Update particles
        print("Done!")
        return

    def get_lambda(self):
        """
        Implement numerical root finding of optimal lambda parameter.
        Pg. 336 of Papaspiliopoulos / Chopin.

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
            delta = self.lambda_max - self.lambdas[-1]
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

    def calc_weight(self):
        if self.iteration == 0:
            self.unnormalized_logweights = [
                self.initial_distribution.logpdf(p)
                for p in self.particles
            ]
        else:
            lambda_t = self.lambdas[-1]
            lambda_t_minus_one = self.lambdas[-2]
            logweights = [
                (self.logscore(p, lambda_t) - self.logscore(p, lambda_t_minus_one))
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

            self.calc_log_normalizing_constant_and_update()

            self.iteration += 1

    def calc_log_normalizing_constant_and_update(self):
        """
        See pg 305 of Papaspiliopoulos / Chopin.
        I cross referenced with the `particles` library by Chopin.

        We can caluculate logLt by 
        logLt = \sum_{s=0}^{t} log( \sum_{n=1}^{N} w_s^n )

        So for every iteration, we add calculate the log normalising constant
        and add it to `self.LogLt`.
        """
        self.logLt += np.log( np.mean(np.exp(self.unnormalized_logweights)) )



def sample(d):
    """
    Sample a permutation of 0, 1, ..., d-1
    """
    return np.random.choice(range(d), size=d, replace=False)


# TODO I'm not sure if this is actually correct
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
    d = x.shape[0]
    m = x.copy()
    i, j1, j2 = np.random.randint(low=0, high=d, size=3)
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
    Uniform Distribution over permutation matrices
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
    """
    The sampler that instantiates the latin square sampler.
    """
    EPSILON = 1e-16 # Precision of estimation of number latin squares

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
            lambda_max = initial_distribution.logpdf(None) - np.log(self.EPSILON) # Stop algorithm when lambda_t > log(p(d)/epsilon)
        )

    def run(self):
        super().run()
        return self.logLt
            


## TESTING
smc = LatinSquareSMC(
    d=5,
    kernel_steps=50,
    particle_number=200
)
logLt = smc.run()

## Test objects

latins = [
        np.matrix(
            [
                [0,1,2],
                [1,2,0],
                [2,0,1]
            ]
            )
        ]

# True number of latin squares
latin_sequence = [1, 2, 12, 576, 161280, 812851200, 61479419904000, 108776032459082956800, 5524751496156892842531225600, 9982437658213039871725064756920320000, 776966836171770144107444346734230682311065600000]
