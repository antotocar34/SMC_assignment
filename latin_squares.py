"""
Students: Alexandra M ́alaga, Antoine Carnec, Maxim Fedotov

This is an implementation of the latin square sampler described in
``
Dau, Hai-Dang, and Nicolas Chopin. "Waste-free Sequential Monte Carlo." arXiv preprint arXiv:2011.02328 (2020)
``

We first implement a generic Adaptive SMC Sampler Class.
This class samples from a posterior `p` of the form
`p(x) \propto m(x) exp(-V(x))`, where m is the prior and exp(-V) is the likelihood
"""
import math

import numpy
import numpy as np
from scipy.stats import multinomial
from scipy.optimize import root_scalar
from scipy.special import softmax
from copy import deepcopy
from collections import Counter


class Metropolis:  # DONE
    """
        Implements Metropolis-Hastings re-sampling given a particle.

        Attributes:
            kernel:       An object with method 'sample'; Markov kernel that draws a new    [callable]
                          sample given the current sample (particle).
            kernel_steps: Number of times the kernel is applied to a particle; defines a    [int]
                          'depth' of MCMC resampling.

        Methods:
            draw:         Draw a new sample from the kernel and accept/reject using Metropolis.
            kfold_steps:  Apply the Metropolis-Hastings step `self.kernel_steps` times.
    """
    def __init__(self, kernel: callable, kernel_steps: int):
        """
        Instantiates Metropolis-Hastings algorithm.

        Parameters:
            kernel:       Object with method 'sample'; Markov kernel that draws a new       [callable]
                          sample given the current sample (particle).
            kernel_steps: Number of times the kernel is applied to a particle; defines a    [int]
                          'depth' of MCMC resampling.
        """
        self.kernel = kernel  # Object which has method 'sample'.
        self.kernel_steps = kernel_steps  # Number of times MCMC is applied to every particle

    def draw(self, x: np.ndarray, pdf: callable) -> numpy.ndarray:
        """
        Draw a new sample from the kernel and accept/reject using Metropolis. Note that this function has a form that
        is applicable to the problem of Latin squares, but not in general.

        The probability of acceptance is defined in the way it is since:
        1. h(x_new | x) = h(x | x_new) for Latin Square problem, where h( | ) is a proposal distribution which
           corresponds to the Markov kernel.
        2. Prior probabilities of both particles x_new and x are equal since we set a uniform prior for the Latin
           squares enumeration problem.

        Parameters:
            x:   Current particle to which the specified Markov kernel is applied    [numpy.ndarray]
            pdf: Function that ascribes likelihood to the particles                  [callable]

        Returns:
            New particle                                                             [numpy.ndarray]
        """
        x_new = self.kernel.sample(x)  # Draw a new sample from kernel
        accept_prob = min(1, pdf(x_new) / pdf(x))  # ATTENTION: This is not a general form of Metropolis-Hastings
        accept = np.random.binomial(1, accept_prob)
        return accept * x_new + (1 - accept) * x

    def kfold_steps(self, x: np.ndarray, pdf: callable) -> np.ndarray:
        """
        Apply the Metropolis-Hastings step `self.kernel_steps` times.

        Parameters:
            x:   Current particle to which the specified Markov kernel is applied    [numpy.ndarray]
            pdf: Function that ascribes likelihood to the particles                  [callable]

        Returns:
            New particle                                                             [numpy.ndarray]
        """
        out = x
        for _ in range(self.kernel_steps):
            out = self.draw(out, pdf)
        return out


class AdaptiveSMC:
    """
        Implements Adaptive SMC algorithm (Algorithm 17.3) form the book:
        Papaspiliopoulos, Chopin (2020) An introduction to sequential Monte Carlo

        Attributes:
            prior:           A prior distribution according to which the initial sample is     [callable]
                             drawn. Corresponds to pi_0 distribution.
            V:               Function that computes a Log-Loss of a particle.                  [callable]
            kernel:          An object with method 'sample'; Markov kernel that draws a new    [callable]
                             sample given the current sample (particle).
            kernel_steps:    Number of times the kernel is applied to a particle; defines a     [int]
                             'depth' of MCMC resampling.
            particle_number: Size of the sample                                                 [int]
            lambda_max:      Lambda value after reaching which the algorithm stops.             [float]
                             We deviate a little from the book here; the latin square
                             sampler requires that lambda can go above 1. So we replace 1
                             with self.lambda_max.
            ess_min_ratio:   Ratio that defines the min Effective Sample Size that the          [float]
                             algorithm maintains at each step.
            verbose:         If True, the methods will print information about the process.     [bool]
            ess_min:         Minimal ESS defined by ess_min_ratio and particle_number.          [float]
            iteration:       Tracks number of iteration.                                        [int]
            w_log:           Unnormalized logarithmic weights. To calculate normalized ones.    [np.ndarray]
            w_normalized:    Normalized weights at time t. For t > 0 are used to sample         [np.ndarray]
                             ancestors at step t+1 from Multinomial(w_normalized).
            _lambda:         Tempering parameter 'lambda' at iteration t; it defines            [float]
                             the sequence of distributions.
            delta:           Defines lambda update, i.e. lambda_t - lambda_(t-1); is chosen     [float]
                             such that the updated sample maintains ESS_min.
            logLt:           Logarithm of the estimated normalized constant, basically:         [float]
                             \sum_{s=0}^{t} log( \sum_{n=1}^{N} w_s^n ).
    """
    def __init__(self, prior: callable, V: callable, kernel: callable, kernel_steps: int, particle_number: int,
                 lambda_max: float = 1., ess_min_ratio: float = 1/2, verbose: bool = False) -> None:
        """
        Instantiates Adaptive SMC sampler.

        Parameters:
            prior:           A prior distribution according to which the initial sample is      [callable]
                             drawn. Corresponds to pi_0 distribution.
            V:               Function that computes a Log-Loss of a particle.                   [callable]
            kernel:          An object with method 'sample'; Markov kernel that draws a new     [callable]
                             sample given the current sample (particle).
            kernel_steps:    Number of times the kernel is applied to a particle; defines a     [int]
                             'depth' of MCMC resampling.
            particle_number: Size of the sample                                                 [int]
            lambda_max:      Lambda value after reaching which the algorithm stops.             [float]
            ess_min_ratio:   Ratio that defines the min Effective Sample Size that the          [float]
                             algorithm maintains at each step.
            verbose:         If True, the methods will print information about the process.     [bool]
        """
        self.prior = prior  # This is the distribution that you start with.
        self.metropolis = Metropolis(kernel, kernel_steps)
        self.V = V
        self.particle_number = particle_number
        self.lambda_max = lambda_max  # Maximum lambda, for a standard sampler this is 1
        self.verbose = verbose
        self.ess_min = particle_number * ess_min_ratio  # Papaspiliopoulos & Chopin states that the performance
                                                        # of the algorithm is pretty robust to this choice.
        # Initializing useful quantities for later
        self.iteration = -1  # Tracks the t variable
        self.particles = None
        self.w_log = None  # unnormalized logweights
        self.w_normalized = None  # normalized weights
        self._lambda = 0.
        self.delta = 0.
        self.logLt = 0.  # This will hold the cumulative value of the log normalising constant at time t.

    def multinomial_draw(self):
        """
        Returns an array of indices.

        For example:
        If we have 5 particles, then we might draw [1,0,0,2,2], which means we will resample particle 1 once
        and particles 4 and 5 two times.

        Returns:
            Sample of size n from ( 0, 1, ..., len(w_normalized) ) with replacement according to    [numpy.ndarray]
            probabilities given by w_normalized.
        """
        assert self.w_normalized is None or np.isclose(sum(self.w_normalized), 1)  # Sanity Check
        return multinomial(n=self.particle_number, p=self.w_normalized).rvs()[0]

    def resample(self) -> None:
        """
        Choose indices to resample and apply k-fold Metropolis kernels.

        Returns:
            None

        Effects:
            Updates attribute 'particles'.
        """
        resample_indices = self.multinomial_draw()
        # Apply the metropolis step k times to each resampled particles
        new_particles = [None] * self.particle_number  # Initialize vector of new particles
        if self.verbose:
            print("Doing Metropolis Resampling...")
        j = 0
        # n = number of times the particle has been resampled
        for particle_idx in (counter := Counter(resample_indices)):
            n = counter[particle_idx]
            if n == 0:  # If the particle is not being resampled at all
                continue
            # Apply k metropolis steps to this particle n times
            new_particles[j:(j + n)] = [self.metropolis.kfold_steps(self.particles[particle_idx],
                                                                    lambda x: np.exp(-self._lambda * self.V(x))
                                                                    # here we don't use nu since it's const.: 1/(d!)^d
                                                                   )
                                        for _ in range(n)]
            j += n

        self.particles = new_particles  # Update particles
        if self.verbose:
            print("Resampling done!")

    def ess_form(self, delta: float) -> np.float64:
        """
        Function to compute ESS for the sample which corresponds to specific delta and V. We use it to
        input in the root-finding routine to calibrate delta to maintain ESS >= ESS_min.

        Parameters:
            self:  instance of Adaptive SMC class.    [object]
            delta: delta to calibrate.                [float]

        Returns:
            Effective Sample Size                     [numpy.float64]
        """
        V = np.array([self.V(p) for p in self.particles])
        return np.sum(np.exp(- delta * V))**2 / np.sum(np.exp(- 2 * delta * V))

    def update_lambda(self) -> None:
        """
        Implement numerical root finding of optimal lambda parameter.
        Pg. 336 of Papaspiliopoulos, Chopin (2020).

        Basically get the next lambda such that the resulting ESS
        is equal to the minimum ESS threshold.

        Parameters:
            self:  instance of Adaptive SMC class.    [AdaptiveSMC]

        Returns:
            None

        Effects:
            Updates attributes 'delta' and '_lambda'.
        """
        try:
            delta = root_scalar(lambda d: self.ess_form(d) - self.ess_min,
                                method='brentq',
                                bracket=[0, self.lambda_max - self._lambda]).root
        except ValueError:
            delta = self.lambda_max - self._lambda
        assert delta > 0, f"delta: {delta}"
        if self.verbose:
            print(f"δ_{self.iteration}: {delta}")
        self.delta = delta
        self._lambda = self._lambda + delta

    def update_weights(self) -> None:
        """
        Weight update according to a new delta found by 'update_lambda'.

        Parameters:
            self:  instance of Adaptive SMC class.    [AdaptiveSMC]

        Returns:
            None

        Effects:
            Updates attributes 'w_log' and 'w_normalized'.
        """
        self.w_log = np.array([- self.delta * self.V(p)
                               for p in self.particles])
        self.w_normalized = softmax(self.w_log)

    def update_logLt(self):
        """
        Updates the logarithm of the normalising constant by accumulating logarithm of mean of weights at each
        iteration. We do it this way since we are interested solely in the normalizing constant of the final
        distribution in the sequence.

        See pg 305 of Papaspiliopoulos / Chopin. I cross referenced with the `particles` library by Chopin.

        We can caluculate logLt by
        $$logLt = \sum_{s=0}^{t} log( \sum_{n=1}^{N} w_s^n )$$

        So for every iteration, we calculate the log normalising constant and add it to `self.LogLt`.

        Parameters:
            self:  instance of Adaptive SMC class.    [AdaptiveSMC]

        Returns:
            None

        Effects:
            Updates attribute 'logLt'.
        """
        self.logLt += np.log(np.mean(np.exp(self.w_log)))

    def ess(self):  # SKIPPED, see ess_form
        """
        Calculates the effective sample size.
        """
        return 1 / sum((W**2 for W in self.w_normalized))

    def run(self):
        """
        Runs the Adaptive SMC algorithm. See Algorithm 2 in the report.

        Parameters:
            self:  instance of Adaptive SMC class.    [AdaptiveSMC]

        Returns:
            None

        Effects:
            Updates all attributes. The logarithm of the estimate of the final normalising constant is kept in 'logLt'.
        """
        if self.verbose:
            print('---SMC started---')
            print(f"λ_max = {self.lambda_max}\n")
        while self._lambda < self.lambda_max:
            self.iteration += 1
            if self.iteration == 0:
                self.particles = self.prior.rvs(size=self.particle_number)  # Start with initial set of particles
            else:
                self.resample()  # Do resampling and metropolis kernel steps
            self.update_lambda()  # Calculate a new lambda by solving for lambda in ess - ess_min = 0
            self.update_weights()  # Recalculate weights
            self.update_logLt()  # Update the normalizing constant
            if self.verbose:
                print(f"Iteration {self.iteration} done!")
                print(f"λ_{self.iteration} : {self._lambda}")
        if self.verbose:
            print('---SMC finished---\n')


def sample(d, seed=None):
    """
    Sample a permutation of 0, 1, ..., d-1
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(d)


class LatinKernel:
    """
    Implements a Markov kernel to be used in SMC for the Latin square enumeration problem. See
    Dau, Hai-Dang, and Nicolas Chopin. "Waste-free Sequential Monte Carlo." (2020).

    Methods:
        sample: Samples new matrix given the current one by interchanging a pair of elements.    @static
    """
    def __init__(self):
        pass

    @staticmethod
    def sample(x_cur: numpy.ndarray) -> numpy.ndarray:
        """
        Takes a d x d matrix and selects a row i and two columns j1 and j2 at random.
        Then it swaps the values of x[i,j1] and x[i,j2].

        Parameters:
            x_cur:  Current particle to provide a new particle in re-sampling procedure.        [numpy.ndarray]

        Returns:
            New particle obtained by sampling form the Markov kernel given the current particle.    [numpy.ndarray]
        """
        assert x_cur.shape[0] == x_cur.shape[1]
        d = x_cur.shape[0]
        x_new = deepcopy(x_cur)
        i = np.random.choice(d)
        j1, j2 = np.random.choice(d, size=2, replace=False)
        x_new[i, j1], x_new[i, j2] = x_new[i, j2], x_new[i, j1]
        return x_new


def V_latin(x: numpy.ndarray) -> float:
    """
    Calculates score (log-loss) of a square matrix. V(x) = 0 if x is a Latin square and V(x) >= 1 if otherwise.

    Parameters:
        x:  Square matrix        [numpy.ndarray]

    Returns:
        New particle obtained by sampling form the Markov kernel given the current particle.    [numpy.ndarray]
    """
    d = x.shape[1]
    return sum(sum(sum(x[i, j] == l for i in range(d))**2 for l in range(d)) for j in range(d)) - d**2


class UniformPermutationMatrix:
    """
    Implements a random variable from Uniform Distribution over permutation matrices of size d x d. Can produce samples.

    Attributes:
        d:    defines size of the matrices, d x d.                               [int]
        seed: random seed                                                        [int]

    Methods:
        rvs:      Implements random sampling of the matrices.
        sample:   Samples a matrix from a uniform distribution over permutation matrices of size d x d.        @static
        logpdf:   Computes the log og probability mass function of a uniform distribution over
                  permutation matrices.
        contains: Checks whether an object 'item' belongs to the class of matrices which rows are permutations
                  of numbers from 0 to d - 1.
    """
    def __init__(self, d: int, seed=None) -> None:
        """
        Instantiates a random variable from Uniform Distribution over permutation matrices.

        Parameters:
            d:    defines size of the matrices, d x d.                               [int]
            seed: random seed                                                        [int]
        """
        self.d = d
        self.seed = seed

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Implements random sampling of the matrices.

        Parameters:
            size: Number of matrices to sample.    [int]

        Returns:
            Sample of 'size' matrices.             [np.ndarray]
        """
        return np.array([self.sample(self.d, self.seed) for _ in range(size)])

    @staticmethod
    def sample(d: int, seed=None) -> np.ndarray:
        """
        Sample a d x d matrix where every row is a permutation of 0, 1, ..., d-1.

        Parameters:
            d:    defines size of the matrices, d x d.                               [int]
            seed: random seed                                                        [int]

        Returns:
            Sampled matrix.                                                          [numpy.ndarray]
        """
        return np.matrix([sample(d, seed) for _ in range(d)])

    def logpdf(self, x=None) -> float:
        """
        Compute the log of 1 / (d!)**d

        Parameters:
            x:    Matrix of size d x d.    [numpy.ndarray]

        Returns:
            log of 1 / (d!)**d             [float]
        """
        if x is not None:
            assert self.__contains__(x)
        return -self.d * sum(np.log(1 + np.arange(self.d)))

    def __contains__(self, item: numpy.ndarray) -> bool:
        """
        Checks whether an object 'item' belongs to the class of matrices which rows are permutations of
        numbers from 0 to d - 1.

        Parameters:
            item: Matrix of size d x d.                              [numpy.ndarray]

        Returns:
            True if the item belongs to class, False otherwise.      [bool]
        """
        if item.shape != (self.d, self.d):
            return False
        template = np.arange(self.d)
        for row in item:
            if all(sorted(row) == template):
                continue
            else:
                return False
        return True


class LatinSquareSMC(AdaptiveSMC):
    """
    Instantiates an Adaptive SMC sampler for Latin Squares Enumeration problem.

    Attributes:
        Inherits attributes of Adaptive SMC +
        eps: Tolerance of estimation of the number of Latin squares.                            [float]
             Note that for basic Adaptive SMC that we use it is not very reliable.
             It works better for Waste-Free SMC samplers.
    """
    def __init__(self, d: int, kernel_steps: int, particle_number: int, verbose: bool = False, eps: float = 1e-16):
        """
        Instantiates Adaptive SMC sampler for Latin Squares Enumeration problem.

        Parameters:
            d:               defines size of the matrices, d x d.                               [int]
            kernel_steps:    Number of times the kernel is applied to a particle; defines a     [int]
                             'depth' of MCMC resampling.
            particle_number: Size of the sample                                                 [int]
            verbose:         If True, the methods will print information about the process.     [bool]
            eps:             Tolerance of estimation of the number of Latin squares.            [float]
                             Note that for basic Adaptive SMC that we use it is not very
                             reliable. It works better for Waste-Free SMC samplers.
        """
        self.eps = eps
        prior = UniformPermutationMatrix(d)
        V = V_latin
        kernel = LatinKernel()
        super().__init__(prior, V, kernel, kernel_steps, particle_number,
                         lambda_max=prior.logpdf() - np.log(self.eps),  # Stop algorithm when lambda_t >= log(p(d)/eps)
                         verbose=verbose)


if __name__ == '__main__':
    d = 4
    kernel_steps = 1000
    particle_number = int(2e5 / kernel_steps)
    smc = LatinSquareSMC(d=4,
                         kernel_steps=kernel_steps,
                         particle_number=particle_number,
                         verbose = True
                         )
    smc.run()
    logLt = smc.logLt
    num_latin = math.factorial(4)**4 * np.exp(smc.logLt)
    print(f'\nEstimated log of the normalizing constant: {logLt}\n')
    print(f"Estimated number of latin squares for d = {d}: {math.factorial(d)**d * np.exp(smc.logLt)}")
    # True number of latin squares
    latin_sequence = [1, 2, 12, 576, 161280, 812851200, 61479419904000, 108776032459082956800, 5524751496156892842531225600, 9982437658213039871725064756920320000, 776966836171770144107444346734230682311065600000]
