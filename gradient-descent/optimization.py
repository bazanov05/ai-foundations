# Aliaksei Baranau

import numpy as np
from abc import ABC, abstractmethod
from typing import override, Callable
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os


def compute_gradient(
        function: Callable[[np.ndarray], float], 
        point: np.ndarray, 
        h: float = 1e-5
    ) -> np.ndarray:
    """Estimate the gradient of a scalar function with central differences.

    Each partial derivative is (f(x + h*e_i) - f(x - h*e_i)) / (2h), with
    truncation error O(h^2). The default h = 1e-5 balances this error against
    floating-point round-off.

    Args:
        function: Objective function; takes a 1D array and returns a float.
        point: 1D array, the point at which the gradient is evaluated.
        h: Finite-difference step.

    Returns:
        1D array with the same shape as `point`, the estimated gradient.
    """
    point = np.asarray(point, dtype=float)
    gradient = np.zeros_like(point, dtype=float)
    number_of_dims = point.shape[0]

    for i in range(number_of_dims):
        point_forward = point.copy()
        point_backward = point.copy()

        point_forward[i] += h
        point_backward[i] -= h

        # use Central Difference Scheme
        gradient[i] = (function(point_forward) - function(point_backward)) / (2 * h)

    return gradient


class Optimizer(ABC):
    """Base class for optimizers.

    Subclasses implement the update rule in `step`. Optimizers are stateful
    (velocity, accumulated gradients, step counter), so a new instance must
    be created for every run.

    Args:
        lr: Learning rate (step size).
    """
    def __init__(self, lr: float):
        self.lr = lr
    
    @abstractmethod
    def step(self, point: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        pass


class GD(Optimizer):
    """Plain gradient descent: x <- x - lr * grad.

    Args:
        lr: Learning rate.
    """
    def __init__(self, lr: float):
        super().__init__(lr)

    @override
    def step(self, point: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return point - self.lr * gradient


class GDWithMomentum(Optimizer):
    """Gradient descent with momentum.

    Update: v <- beta * v + grad;  x <- x - lr * v.

    Args:
        lr: Learning rate.
        beta: Momentum coefficient, in [0, 1).
    """
    def __init__(self, lr: float, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self.velocity = None

    @override
    def step(self, point: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.velocity is None:
            self.velocity = np.zeros_like(point, dtype=float)

        # v = beta * v + grad
        self.velocity = self.beta * self.velocity + gradient

        # x = x - lr * v
        return point - self.lr * self.velocity


class ADAGrad(Optimizer):
    """AdaGrad: per-coordinate step scaled by accumulated squared gradients.

    Update: G <- G + grad^2;  x <- x - lr * grad / (sqrt(G) + epsilon).
    G only grows, so the step shrinks over time.

    Args:
        lr: Base learning rate.
        epsilon: Small constant preventing division by zero.
    """
    def __init__(self, lr: float, epsilon: float = 1e-8):
        super().__init__(lr)
        self.squared_gradients = None
        self.epsilon = epsilon

    @override
    def step(self, point: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.squared_gradients is None:
            self.squared_gradients = np.zeros_like(point, dtype=float)

        self.squared_gradients += gradient ** 2
        adaptive_lr = self.lr / (np.sqrt(self.squared_gradients) + self.epsilon)

        return point - adaptive_lr * gradient


class Adam(Optimizer):
    """Adam: momentum combined with per-coordinate step normalization.

    Update (t = step number, starting at 1):
        m <- beta1 * m + (1 - beta1) * grad          (`momentum`)
        v <- beta2 * v + (1 - beta2) * grad^2        (`volatility`)
        m_hat = m / (1 - beta1^t),  v_hat = v / (1 - beta2^t)
        x <- x - lr * m_hat / (sqrt(v_hat) + epsilon)

    Args:
        lr: Learning rate.
        beta1: Decay rate of the first moment (running mean of gradients).
        beta2: Decay rate of the second moment (running mean of squared gradients).
        epsilon: Small constant preventing division by zero.
    """
    def __init__(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.time = 1
        self.momentum = None
        self.volatility = None

    @override
    def step(self, point: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.momentum is None:
            self.momentum = np.zeros_like(point, dtype=float)
            self.volatility = np.zeros_like(point, dtype=float)

        # 1. and 2. momentum
        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * gradient
        self.volatility = self.beta2 * self.volatility + (1 - self.beta2) * (gradient**2)

        # bias correction (unbiased estimators)
        m_hat = self.momentum / (1 - self.beta1**self.time)
        v_hat = self.volatility / (1 - self.beta2**self.time)

        self.time += 1

        return point - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


def optimize(
    obj_fun: Callable,
    start_point: np.ndarray,
    optimizer: Optimizer,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    h: float = 1e-5,
) -> tuple[np.ndarray, list[float]]:
    """Minimize an objective function with the given optimizer.

    The loop stops when:
    (1) the gradient norm is below `tolerance`, (2) `max_iterations` steps
    were made, or (3) the objective becomes non-finite or exceeds 1e15
    (treated as divergence). The point that caused divergence stays in the history.

    Args:
        obj_fun: Objective function; takes a 1D array and returns a float.
        start_point: 1D array, initial point.
        optimizer: Optimizer instance; stateful, so use a fresh one per call.
        max_iterations: Maximum number of optimizer steps.
        tolerance: Stop when the Euclidean norm of the gradient falls below it.
        h: Finite-difference step used for the numerical gradient.

    Returns:
        Tuple (trajectory, f_values). `trajectory` is an array of shape
        (n + 1, dim) with all visited points including the start;
        `f_values` is a list of n + 1 objective values at these points.
    """
    point = np.asarray(start_point, dtype=float)
    trajectory = [point.copy()]     # history of points
    f_values = [obj_fun(point)]     # history of function's values

    for _ in range(max_iterations):
        grad = compute_gradient(function=obj_fun, point=point, h=h)

        if np.linalg.norm(grad) < tolerance:
            break

        point = optimizer.step(point, grad)
        f_val = obj_fun(point)
            
        trajectory.append(point.copy())
        f_values.append(f_val)

        # stop if value explodes 
        if f_val > 1e15 or np.isnan(f_val) or np.isinf(f_val):
            break

    return np.asarray(trajectory), f_values


def sphere_function(point: np.ndarray) -> float:
    """f(x) = x1^2 + x2^2"""
    return float(point[0] ** 2 + point[1] ** 2)


def matyas_function(point: np.ndarray) -> float:
    """f(x) = 0.26*(x1^2 + x2^2) - 0.48*x1*x2"""
    return float(
    0.26 * (point[0] ** 2 + point[1] ** 2) - 0.48 * point[0] * point[1]
    )


@dataclass
class FunctionConfig:
    name: str
    func: Callable[[np.ndarray], float]


class OptimizationExperiment:
    """Runs optimization experiments for a single objective function.

    Args:
        config: Objective function to study.
    """
    def __init__(self, config: FunctionConfig):
        self.config = config

    def analyze_learning_rates(
        self, 
        optimizer_cls: type, 
        lrs: list[float], 
        start_point: np.ndarray,
        max_iterations: int = 100
    ) -> dict[float, list[float]]:
        """Run one optimizer for several learning rates from the same start point.

        Args:
            optimizer_cls: Optimizer class (not an instance); instantiated
                with each learning rate.
            lrs: Learning rates to test.
            start_point: Initial point shared by all runs.
            max_iterations: Iteration limit per run.

        Returns:
            Dict mapping each learning rate to the objective values along its
            run, including the start. Runs that stop early are shorter.
        """
        results = {}
        for lr in lrs:
            optimizer = optimizer_cls(lr=lr)

            # we do not need trajectory here - only f_values 
            _, f_vals = optimize(
                obj_fun=self.config.func,
                start_point=start_point,
                optimizer=optimizer,
                max_iterations=max_iterations
            )
            results[lr] = f_vals
        return results

    def analyze_trajectories(
        self, 
        optimizer_cls: type, 
        lr: float, 
        start_points: list[np.ndarray],
        max_iterations: int = 1000
    ) -> dict[tuple, np.ndarray]:
        """Run one optimizer with a fixed learning rate from several start points.

        Args:
            optimizer_cls: Optimizer class (not an instance).
            lr: Learning rate used in all runs.
            start_points: Initial points, one run per point.
            max_iterations: Iteration limit per run.

        Returns:
            Dict mapping each start point (as a tuple) to its trajectory,
            an array of shape (n + 1, dim).
        """
        results = {}
        for point in start_points:
            optimizer = optimizer_cls(lr=lr)
            trajectory, _ = optimize(
                obj_fun=self.config.func,
                start_point=point,
                optimizer=optimizer,
                max_iterations=max_iterations
            )
            results[tuple(point)] = trajectory
        return results


def plot_all_optimizers_grid(
    grid_results: dict[str, dict[float, list[float]]],
    title: str,
    save_path: str
):
    """Plot convergence curves for several optimizers in a 2x2 grid (log y axis).

    Args:
        grid_results: Maps optimizer name to a dict {learning rate: f values
            per iteration}. At most 4 optimizers (one per subplot).
        title: Figure title.
        save_path: Path of the output PNG file.

    Returns:
        None. The figure is saved to `save_path`.
    """
    _, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for idx, (opt_name, lr_data) in enumerate(grid_results.items()):
        ax = axes[idx]
        for lr, f_vals in lr_data.items():
            ax.plot(f_vals, label=f"lr = {lr}", linewidth=1.5)

        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-12)
        ax.set_title(opt_name)
        ax.set_xlabel("Iteration step")
        ax.set_ylabel("Objective function value f(x)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def generate_trajectory_3d_reports(
    obj_fun: Callable,
    fun_name: str,
    starting_points: list[np.ndarray],
    opt_trajectories: dict[str, dict[tuple, np.ndarray]],
    title: str,
    path: str
) -> None:
    """Plot optimizer trajectories on the 3D surface of the objective function.

    Creates one figure per start point: a 2x2 grid with one 3D subplot per
    optimizer, showing the path, the start (blue) and the final point (yellow).

    Args:
        obj_fun: Objective function; takes a 1D array of length 2.
        fun_name: Function name, used in output file names.
        starting_points: Start points; each must be a key (as a tuple) in the
            inner dicts of `opt_trajectories`.
        opt_trajectories: Maps optimizer name to {start point tuple: trajectory}.
            At most 4 optimizers.
        title: Figure title prefix.
        path: Output directory; files are named '<fun_name>_traj_point_<k>.png'.

    Returns:
        None. The figures are saved to `path`.
    """
    MIN_X, MAX_X = -10, 10
    PLOT_STEPS = 50

    # compute surface grid once for the entire objective function
    x = np.linspace(MIN_X, MAX_X, PLOT_STEPS)
    X1, X2 = np.meshgrid(x, x)
    Z = np.zeros_like(X1)
    for i in range(PLOT_STEPS):
        for j in range(PLOT_STEPS):
            Z[i, j] = obj_fun(np.array([X1[i, j], X2[i, j]]))

    optimizer_names = list(opt_trajectories.keys())

    # iterate through each starting point and generate a 2x2 grid
    for pt_idx, start_pt in enumerate(starting_points):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 11),
            subplot_kw={'projection': '3d'},
            constrained_layout=True,
        )
        axes = axes.flatten()

        fig.suptitle(
            f'{title} - Start: [{start_pt[0]}, {start_pt[1]}]',
            fontsize=16,
            fontweight='bold',
        )

        for opt_idx, opt_name in enumerate(optimizer_names):
            ax = axes[opt_idx]
            
            trajectory = opt_trajectories[opt_name][tuple(start_pt)]

            # calculate objective function value (Z) for every point in the trajectory
            traj_z = np.array([obj_fun(p) for p in trajectory])

            # plot the semi-transparent objective function surface
            ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.5, edgecolor='none')

            # plot the optimization trajectory path
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                traj_z,
                color='red',
                linewidth=2.0,
                marker='o',
                markersize=3,
                label='Optimization path',
                zorder=10,
            )

            # mark the starting and terminating points
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                traj_z[0],
                color='blue',
                s=50,
                label='Start',
                zorder=11,
            )
            ax.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                traj_z[-1],
                color='yellow',
                edgecolor='black',
                s=70,
                label='Final point',
                zorder=11,
            )

            ax.set_title(f'{opt_name}', fontsize=12)
            ax.set_xlabel('X1', fontsize=9)
            ax.set_ylabel('X2', fontsize=9)
            ax.set_zlabel('f(X)', fontsize=9)
            ax.tick_params(labelsize=8)
            ax.view_init(elev=35, azim=-120)

            if opt_idx == 0:
                ax.legend(loc='upper right', fontsize=8)

        filename = f'{fun_name.lower()}_traj_point_{pt_idx + 1}.png'
        full_path = os.path.join(path, filename)
        plt.savefig(full_path, dpi=200, bbox_inches='tight')
        plt.close(fig)


def main():
    """Run all experiments and save the plots to ./plots/.

    For the Sphere and Matyas functions: learning-rate comparison from
    [10, -10] for all optimizers, and trajectories (lr = 0.1) from three
    start points.
    """
    learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0, 1.1]
    optimizers = [GD, GDWithMomentum, ADAGrad, Adam]
    start_point = np.array([10.0, -10.0])
    start_points = [
        np.array([10.0, -10.0]),    # global maximum for range [-10, 10]
        np.array([8.0, 8.0]),       # same coord, flat surface
        np.array([-7.0, 2.0])       # diff coord
    ]

    base_path_convergence = "./plots/convergence/"
    base_path_trajectories = "./plots/trajectories/"
    os.makedirs(base_path_convergence, exist_ok=True)
    os.makedirs(base_path_trajectories, exist_ok=True)

    sphere_config = FunctionConfig(name="Sphere", func=sphere_function)
    matyas_config = FunctionConfig(name="Matyas", func=matyas_function)

    for config in [sphere_config, matyas_config]:
        experiment = OptimizationExperiment(config)
        func_name = config.name.lower()
        
        lr_data = {}
        traj_data = {}

        for optimizer in optimizers:
            lrs_to_f_values = experiment.analyze_learning_rates(
                optimizer_cls=optimizer, 
                lrs=learning_rates,
                start_point=start_point
            )
            trajectories = experiment.analyze_trajectories(
                optimizer_cls=optimizer,
                lr=0.1,
                start_points=start_points
            )
            opt_name = optimizer.__name__
            lr_data[opt_name] = lrs_to_f_values
            traj_data[opt_name] = trajectories

        title_conv = f"Learning Rate Comparison - {config.name}"
        path_conv = os.path.join(base_path_convergence, f"{func_name}_all_optimizers.png")
        
        plot_all_optimizers_grid(
            grid_results=lr_data,
            title=title_conv,
            save_path=path_conv
        )

        title_traj = f"Trajectories Comparison - {config.name}"
        generate_trajectory_3d_reports(
            obj_fun=config.func,
            fun_name=config.name,
            starting_points=start_points,
            opt_trajectories=traj_data,
            title=title_traj,
            path=base_path_trajectories
        )


if __name__ == "__main__":
    main()
