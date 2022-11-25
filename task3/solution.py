import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
import gpytorch
import torch
from torch.distributions import Normal
from scipy import interpolate


domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0

""" Solution """
# IDEA: Possible implementations -@hyeongkyunkim at 11/24/2022, 3:58:01 AM
# http://krasserm.github.io/2018/03/21/bayesian-optimization/
# https://machinelearningmastery.com/what-is-bayesian-optimization/
gpytorch.kernels.MaternKernel

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super().__init__(train_x, train_y, likelihood=likelihood) 
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5,lengthscale=0.5))

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    @property
    def output_scale(self):
        """Get output scale."""
        return self.covar_module.outputscale
    
    @property
    def length_scale(self):
        """Get length scale."""
        return self.covar_module.base_kernel.lengthscale
    
    @length_scale.setter
    def length_scale(self, value):
        self.covar_module.base_kernel.lengthscale = value 

class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        # IDEA: Possible implementations -@hyeongkyunkim at 11/23/2022, 2:34:30 AM
        # USE ExpectedImprovement with xi = 0.01 as an activation function

        # HACK: Temporary fix -@hyeongkyunkim at 11/23/2022, 3:59:42 AM
        x0 = torch.tensor([[0.0]])
        with torch.no_grad():
            y0 = torch.tensor([f(x0)]).type(torch.get_default_dtype())
        y0 = y0.type(torch.get_default_dtype())
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise_covar.noise = 0.01 ** 2
        gp = ExactGP(train_x = x0, train_y = y0, likelihood = likelihood)

        self.gp = gp
        self.gp.eval()
        self.gp.likelihood.eval()
        self.x = torch.linspace(domain[0][0], domain[0][1], 4000)

        self.xi = 0.0001 # xi = 0.01 -> 0.4848   0.001 -> 0.4674

        self.update_acquisition_function()

        self.ymax = 0

        pass

    def next_recommendation(self):
        """
        Recommend the next input to sample.
        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.

        #       2.2 Acquisition Functions for Bayesian Optimization
        #        ... determines what point in X should be evaluated next via a proxy optimization.

        return self.optimize_acquisition_function() 

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.
        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def update_acquisition_function(self):
        xmax = self.get_solution()
        ymax = self.ymax
        out = self.gp(self.x)
        dist = Normal(torch.tensor([0.]), torch.tensor([1.]))
        Z = (out.mean - ymax - self.xi) / out.stddev 
        idx = out.stddev == 0
        _acquisition_function = (out.mean - ymax - self.xi) * dist.cdf(Z) + out.stddev * torch.exp(dist.log_prob(Z))
        _acquisition_function[idx] = 0

        x = self.x.detach().numpy()
        y = _acquisition_function.detach().numpy()

        self._acquisition_function = interpolate.interp1d(x=x, y=y)

    def acquisition_function(self, x):
        """
        Compute the acquisition function.
        Parameters
        ----------
        x: np.ndarray
            x in domain of f
        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here

        # IDEA: Possible implementations -@hyeongkyunkim at 11/23/2022, 3:20:06 AM
        #         2.2 Acquisition Functions for Bayesian Optimization
        #           Expected Improvement |  Alternatively, one could choose to maximize the expected improvement
        #           (EI) over the current best. This also has closed form under the Gaussian process:
        #           In this work we will focus on the EI criterion, as it has been shown to be better-behaved than
        #           probability of improvement
        #           
        #           Expected Improvement Reference
        #           http://krasserm.github.io/2018/03/21/bayesian-optimization/
        #
        # This is the activation function following the method 'Expected Improvement'
        # xi will define the importance of exploration (higher xi, higher exploration)
        return self._acquisition_function(x) 

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.
        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here     
        # 2.2 Acquisition Functions for Bayesian Optimization
        
        # In general, these acquisition functions depend on the previous observations,
        # as well as the GP hyperparameters;
        # algorithm.update_gp(query_x, query_y)  # update GP model. 
        # x_valid = x_domain[v > SAFETY_THRESHOLD]
        

        inputs = self.gp.train_inputs[0]
        targets = self.gp.train_targets
        x = x.item()
        f = f.item()
        # if isinstance(x, np.ndarray):
        #     x = torch.Tensor(x)
        #     print('probe cat1')
        #     print(inputs.shape,x.shape)
        # else:
        #     x = torch.Tensor([x]).unsqueeze(-1)
        #     print('probe cat2')
        #     print(inputs.shape,x.shape)
        x = torch.Tensor([[x]])
        f = torch.Tensor([f])
        if v > SAFETY_THRESHOLD:
            new_inputs = torch.cat((inputs, x), dim=0)
            new_targets = torch.cat((targets, f), dim=-1)
            self.gp.set_train_data(new_inputs, new_targets, strict=False)
            self.update_acquisition_function()

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.
        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        # IDEA: Possible implementations -@hyeongkyunkim at 11/23/2022, 3:20:06 AM
        # Return x_opt that is believed to be the maximizer of the expected objective function
        # We can get the expected objective function by using the gaussian process.
        idx = self.gp.train_targets.argmax()
        if len(self.gp.train_targets) == 1:
            x_opt, self.ymax = self.gp.train_inputs[idx], self.gp.train_targets[idx]
        else:
            x_opt, self.ymax = self.gp.train_inputs[0][idx], self.gp.train_targets[idx]
        return x_opt


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])

def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # minus 2-norm e.g. -(x - 2.5)^2 

def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None] # column vector 4000x1,
    c_val = np.vectorize(v)(x_domain) # column vector 4000x1, every value is v(x)=2
    x_valid = x_domain[c_val > SAFETY_THRESHOLD] # column vector ?x1, keep only x_domain index where c_val > SAFETY_THRESHOLD
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0] # Choose one of x_valid randomly
    return x_init


def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point() # Initial Hyperparameters
    obj_val = f(x_init) # Initial Model accuracy
    cost_val = v(x_init) # Initial Model training speed
    agent.add_data_point(x_init, obj_val, cost_val) 

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)


    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()