"""
This is an implementation of the latin square sampler described in 
``
Dau, Hai-Dang, and Nicolas Chopin. "Waste-free Sequential Monte Carlo." arXiv preprint arXiv:2011.02328 (2020)
``

We first implement a generic Adaptive SMC Sampler Class.
This class samples from a posterior `p` of the form
`p(x) \propto m(x) exp(-V(x))`, where m is the prior and exp(-V) is the likelihood
"""
import math

import numpy as np
from scipy.stats import multinomial
from scipy.optimize import root_scalar
from scipy.special import softmax
from copy import deepcopy
from collections import Counter


SEED = 42
np.random.seed(SEED)


class Metropolis:  # DONE
    def __init__(self, kernel, kernel_steps):
        self.kernel = kernel
        self.kernel_steps = kernel_steps  # Number of times MCMC is applied to every particle

    def draw(self, x, pdf):
        """
        Draw a new sample from the kernel and accept/reject using
        Metropolis.
        """
        x_new = self.kernel.sample(x)  # Draw a new sample from kernel
        accept_prob = min(1, pdf(x_new) / pdf(x))  # ATTENTION: This is not a general form of Metropolis-Hastings
        # The above statement is correct since h(x_new | x) = h(x | x_new) for Latin Square problem.
        accept = np.random.binomial(1, accept_prob)
        return accept * x_new + (1 - accept) * x

    def kfold_steps(self, x, pdf):
        """
        Apply the metropolis step `self.kernel_steps` times
        """
        out = x
        for _ in range(self.kernel_steps):
            out = self.draw(out, pdf)
        return out


class AdaptiveSMC:
    """
    Algorithm 17.3 of Papaspiliopoulos / Chopin Book
    """
    def __init__(
        self,
        prior,
        V,
        kernel,
        kernel_steps,
        particle_number,
        lambda_max=1,
        ess_min_ratio=1/2
    ):
        self.metropolis = Metropolis(kernel, kernel_steps)
        self.particle_number = particle_number
        self.prior = prior  # This is the distribution that you start with.
        self.V = V
        # Initializing useful quantities for later
        self.iteration = -1  # Tracks the t variable
        self.particles = None
        self.w_log = None
        self.w_normalized = None
        self.lambd = 0
        self.delta = 0
        self.lambda_max = lambda_max  # Maximum lambda, for a standard sampler this is 1
        self.ess_min = particle_number * ess_min_ratio  # Papaspiliopoulos & Chopin states that the performance
                                                        # of the algorithm is pretty robust to this choice.
        self.logLt = 0.  # This will hold the cumulative value of the log normalising constant at time t.

    def initial_sample(self):  # DONE
        """
        Sample from the initial distribution.
        """
        n = self.particle_number
        return self.prior.rvs(size=n)

    def multinomial_draw(self):  # DONE
        """
        Returns an array of indices.

        For example:
        if we have 5 particles,
        then we might draw
        [1,0,0,2,2]
        which means we will resample particle 1 once
        and particles 4 and 5 three times.
        """
        assert self.w_normalized is None or np.isclose(sum(self.w_normalized), 1)  # Sanity Check
        return multinomial(n=self.particle_number, p=self.w_normalized).rvs()[0]

    def resample(self):  # DONE
        """
        Choose indices to resample and apply k-fold Metropolis
        kernels.
        """
        resample_indices = self.multinomial_draw()
        # Apply the metropolis step k times to each resampled particles
        new_particles = [None] * self.particle_number  # Initialize vector of new particles
        print("Doing Metropolis Resampling...")
        j = 0
        # n = number of times the particle has been resampled
        ## for i, n in enumerate(resample_indices):
        for particle_idx in (counter := Counter(resample_indices)):
            n = counter[particle_idx]
            if n == 0:  # If the particle is not being resampled at all
                continue
            # Apply k metropolis steps to this particle n times
            new_particles[j:(j + n)] = [
                self.metropolis.kfold_steps(
                    self.particles[particle_idx],
                    lambda x: np.exp(-self.lambd * self.V(x))  # here we don't use nu since it's const.: 1/(d!)^d
                )
                for _ in range(n)
            ]
            j += n

        self.particles = new_particles  # Update particles
        print("Done!")
        return

    def ess_form(self, delta):  # DONE
        V = np.array([self.V(p) for p in self.particles])
        w = np.exp(- delta * V)
        return np.sum(w)**2 / np.sum(w**2)

    def get_lambda(self):  # DONE
        """
        Implement numerical root finding of optimal lambda parameter.
        Pg. 336 of Papaspiliopoulos / Chopin.

        Basically get the next lambda such that the resulting ESS
        is equal to the minimum ESS threshold.
        """
        try:
            delta = root_scalar(lambda d: self.ess_form(d) - self.ess_min,
                                method='brentq',
                                bracket=[0, self.lambda_max - self.lambd]).root
        except ValueError:
            delta = self.lambda_max - self.lambd
        assert delta > 0, f"delta: {delta}"
        print(f"δ_{self.iteration}: {delta}")

        # We deviate a little from the book here ;
        # the latin square sampler requires that lambda can go above 1.
        # So we replace 1 with self.lambda_max
        self.delta = delta
        self.lambd = self.lambd + delta
        return


    def calc_weight(self) -> None:  # DONE
        self.w_log = np.array([- self.delta * self.V(p)
                               for p in self.particles])
        self.w_normalized = softmax(self.w_log)

    def ess(self):  # SKIPPED
        """
        Calculate the effective sample size.
        """
        return 1 / sum((W**2 for W in self.w_normalized))

    def run(self):  # DONE
        print(f"λmax = {self.lambda_max}")
        while self.lambd < self.lambda_max:
            self.iteration += 1
            if self.iteration == 0:
                self.particles = self.initial_sample()  # Start with inital set of particles
            else:
                self.resample()  # Do resampling and metropolis kernel steps
            self.get_lambda()  # Calculate a new lambda by solving for lambda in ess - ess_min = 0
            self.calc_weight()  # Recalculate weights
            print(f"Iteration {self.iteration} done!")
            print(f"λ_{self.iteration} : {self.lambd}")
            self.calc_log_normalizing_constant_and_update()
        print('SMC finished!')

    def calc_log_normalizing_constant_and_update(self):  # DONE
        """
        See pg 305 of Papaspiliopoulos / Chopin.
        I cross referenced with the `particles` library by Chopin.

        We can caluculate logLt by
        logLt = \sum_{s=0}^{t} log( \sum_{n=1}^{N} w_s^n )

        So for every iteration, we add calculate the log normalising constant
        and add it to `self.LogLt`.
        """
        self.logLt += np.log(np.mean(np.exp(self.w_log)))


def sample(d, seed=None):  # DONE
    """
    Sample a permutation of 0, 1, ..., d-1
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(d)


class LatinKernel:  # DONE
    def __init__(self):
        pass

    @staticmethod
    def sample(x_cur):
        """
        Takes a d x d matrix and selects a row i
        and two columns j1 and j2 at random.
        then it swaps the values of x[i,j1] and x[i,j2]
        """
        assert x_cur.shape[0] == x_cur.shape[1]
        d = x_cur.shape[0]
        x_new = deepcopy(x_cur)
        i = np.random.choice(d)
        j1, j2 = np.random.choice(d, size=2, replace=False)
        x_new[i, j1], x_new[i, j2] = x_new[i, j2], x_new[i, j1]
        return x_new


def V_latin(x):  # DONE
    """
    Calculate score of latin square.
    """
    d = x.shape[1]
    return sum(sum(sum(x[i, j] == l for i in range(d))**2 for l in range(d)) - d for j in range(d))


class UniformPermutationMatrix:  # DONE
    """
    Uniform Distribution over permutation matrices
    """
    def __init__(self, d, seed=None):
        self.d = d
        self.seed = seed

    def rvs(self, size=1):
        """
        Implement random sampling
        """
        return np.array([self.sample(self.d, self.seed) for _ in range(size)])

    @staticmethod
    def sample(d, seed=None):
        """
        Sample a d x d matrix where every row is a permutation of
        0, 1, ..., d-1
        """
        return np.matrix([sample(d, seed) for _ in range(d)])

    def logpdf(self, x=None):
        """
        Compute the log of 1 / (d!)**d
        """
        if x is not None:
            assert self.contains(x)
        return -self.d * sum(np.log(1 + np.arange(self.d)))

    def __contains__(self, item):
        template = np.arange(self.d)
        for row in item:
            if all(sorted(row) == template):
                continue
            else:
                return False


class LatinSquareSMC(AdaptiveSMC):
    """
    The sampler that instantiates the latin square sampler.
    """
    eps = 1e-16  # Precision of estimation of number latin squares

    def __init__(self, d, kernel_steps, particle_number):
        prior = UniformPermutationMatrix(d)
        V = V_latin
        kernel = LatinKernel()
        super().__init__(
            prior,
            V,
            kernel,
            kernel_steps,
            particle_number,
            lambda_max=prior.logpdf() - np.log(self.eps)  # Stop algorithm when lambda_t >= log(p(d)/epsilon)
        )
            

if __name__ == '__main__':
    d = 4
    kernel_steps = 500
    particle_number = int(2e5 / kernel_steps)
    smc = LatinSquareSMC(
        d=d,
        kernel_steps=kernel_steps,
        particle_number=particle_number
    )
    smc.run()
    p_d = UniformPermutationMatrix(d).logpdf()
    print(math.factorial(d)**d * np.exp(smc.logLt))
    # Test objects
    # latins = [
    #     np.matrix(
    #         [
    #             [0,1,2],
    #             [1,2,0],
    #             [2,0,1]
    #         ]
    #     )
    # ]
    
    # True number of latin squares
    # latin_sequence = [1, 2, 12, 576, 161280, 812851200, 61479419904000, 108776032459082956800,
    #                   5524751496156892842531225600, 9982437658213039871725064756920320000,
    #                   776966836171770144107444346734230682311065600000]
