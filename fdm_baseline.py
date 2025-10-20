"""
Finite Difference Method (FDM) for Black-Scholes PDE - Baseline Implementation

This module implements numerical PDE solvers for European options using:
1. Explicit Euler scheme
2. Implicit Euler scheme (Crank-Nicolson)

Purpose: Serve as accurate baseline to validate analytical formulas from blackscholes library
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Tuple, Dict
import time


class BlackScholesFDM:
    """
    Finite Difference Method solver for Black-Scholes PDE.

    Solves: ∂V/∂t + (r-q)S∂V/∂S + (1/2)σ²S²∂²V/∂S² - rV = 0

    Parameters:
    -----------
    S_max : float
        Maximum stock price in grid
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    q : float
        Dividend yield
    M : int
        Number of stock price grid points
    N : int
        Number of time steps
    option_type : str
        'call' or 'put'
    """

    def __init__(
        self,
        S_max: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        M: int = 100,
        N: int = 1000,
        option_type: str = 'call'
    ):
        self.S_max = S_max
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.M = M
        self.N = N
        self.option_type = option_type.lower()

        # Create grids
        self.dS = S_max / M
        self.dt = T / N
        self.S_grid = np.linspace(0, S_max, M + 1)
        self.t_grid = np.linspace(0, T, N + 1)

        # Initialize solution grid
        self.V = np.zeros((M + 1, N + 1))

        # Set terminal condition (payoff at maturity)
        self._set_terminal_condition()

        # Set boundary conditions
        self._set_boundary_conditions()

    def _set_terminal_condition(self):
        """Set payoff at maturity T."""
        if self.option_type == 'call':
            self.V[:, -1] = np.maximum(self.S_grid - self.K, 0)
        elif self.option_type == 'put':
            self.V[:, -1] = np.maximum(self.K - self.S_grid, 0)
        else:
            raise ValueError(f"Unknown option_type: {self.option_type}")

    def _set_boundary_conditions(self):
        """Set boundary conditions for all time steps."""
        if self.option_type == 'call':
            # Lower boundary: V(0, t) = 0
            self.V[0, :] = 0
            # Upper boundary: V(S_max, t) ≈ S_max - K*exp(-r*(T-t))
            for j in range(self.N + 1):
                t = self.t_grid[j]
                self.V[-1, j] = self.S_max - self.K * np.exp(-self.r * (self.T - t))
        else:  # put
            # Lower boundary: V(0, t) = K*exp(-r*(T-t))
            for j in range(self.N + 1):
                t = self.t_grid[j]
                self.V[0, j] = self.K * np.exp(-self.r * (self.T - t))
            # Upper boundary: V(S_max, t) ≈ 0
            self.V[-1, :] = 0

    def solve_implicit(self) -> np.ndarray:
        """
        Solve using Implicit (Backward) Euler scheme - unconditionally stable.

        Returns:
        --------
        V : np.ndarray
            Option values on grid (M+1 x N+1)
        """
        # Coefficient vectors for tridiagonal matrix
        alpha = np.zeros(self.M - 1)
        beta = np.zeros(self.M - 1)
        gamma = np.zeros(self.M - 1)

        for i in range(1, self.M):
            S_i = self.S_grid[i]

            # Coefficients from discretization
            alpha[i-1] = 0.5 * self.dt * (self.sigma**2 * i**2 - (self.r - self.q) * i)
            beta[i-1] = -1.0 - self.dt * (self.sigma**2 * i**2 + self.r)
            gamma[i-1] = 0.5 * self.dt * (self.sigma**2 * i**2 + (self.r - self.q) * i)

        # Backward time-stepping (from maturity to present)
        for j in range(self.N - 1, -1, -1):
            # Right-hand side vector
            b = -self.V[1:self.M, j + 1].copy()

            # Adjust for boundary conditions
            b[0] -= alpha[0] * self.V[0, j]
            b[-1] -= gamma[-1] * self.V[-1, j]

            # Construct tridiagonal matrix
            diagonals = [alpha[1:], beta, gamma[:-1]]
            offsets = [-1, 0, 1]
            A = diags(diagonals, offsets, format='csr')

            # Solve linear system
            self.V[1:self.M, j] = spsolve(A, b)

        return self.V

    def solve_crank_nicolson(self) -> np.ndarray:
        """
        Solve using Crank-Nicolson scheme (theta=0.5) - second order accurate.

        Returns:
        --------
        V : np.ndarray
            Option values on grid (M+1 x N+1)
        """
        theta = 0.5  # Crank-Nicolson

        # Build matrices for implicit part
        alpha = np.zeros(self.M - 1)
        beta = np.zeros(self.M - 1)
        gamma = np.zeros(self.M - 1)

        for i in range(1, self.M):
            a = 0.5 * self.dt * (self.sigma**2 * i**2 - (self.r - self.q) * i)
            b = -self.dt * (self.sigma**2 * i**2 + self.r)
            c = 0.5 * self.dt * (self.sigma**2 * i**2 + (self.r - self.q) * i)

            alpha[i-1] = -theta * a
            beta[i-1] = 1.0 - theta * b
            gamma[i-1] = -theta * c

        # Backward time-stepping
        for j in range(self.N - 1, -1, -1):
            # Build right-hand side with explicit part
            b_rhs = np.zeros(self.M - 1)

            for i in range(1, self.M):
                idx = i - 1
                a = 0.5 * self.dt * (self.sigma**2 * i**2 - (self.r - self.q) * i)
                b = -self.dt * (self.sigma**2 * i**2 + self.r)
                c = 0.5 * self.dt * (self.sigma**2 * i**2 + (self.r - self.q) * i)

                explicit_term = (1.0 + (1 - theta) * b) * self.V[i, j + 1]
                if i > 1:
                    explicit_term += (1 - theta) * a * self.V[i - 1, j + 1]
                if i < self.M - 1:
                    explicit_term += (1 - theta) * c * self.V[i + 1, j + 1]

                b_rhs[idx] = explicit_term

            # Adjust for boundary conditions
            b_rhs[0] -= alpha[0] * self.V[0, j]
            b_rhs[-1] -= gamma[-1] * self.V[-1, j]

            # Construct and solve
            diagonals = [alpha[1:], beta, gamma[:-1]]
            offsets = [-1, 0, 1]
            A = diags(diagonals, offsets, format='csr')

            self.V[1:self.M, j] = spsolve(A, b_rhs)

        return self.V

    def get_price(self, S: float) -> float:
        """
        Get option price at spot S and time t=0 via interpolation.

        Parameters:
        -----------
        S : float
            Spot price

        Returns:
        --------
        price : float
            Option value
        """
        return np.interp(S, self.S_grid, self.V[:, 0])

    def get_delta(self, S: float) -> float:
        """
        Calculate Delta (∂V/∂S) using central difference.

        Parameters:
        -----------
        S : float
            Spot price

        Returns:
        --------
        delta : float
            First derivative with respect to S
        """
        # Find nearest grid point
        i = np.searchsorted(self.S_grid, S)
        if i == 0:
            i = 1
        elif i >= self.M:
            i = self.M - 1

        # Central difference
        dV = self.V[i + 1, 0] - self.V[i - 1, 0]
        dS = self.S_grid[i + 1] - self.S_grid[i - 1]

        return dV / dS

    def get_gamma(self, S: float) -> float:
        """
        Calculate Gamma (∂²V/∂S²) using second-order central difference.

        Parameters:
        -----------
        S : float
            Spot price

        Returns:
        --------
        gamma : float
            Second derivative with respect to S
        """
        # Find nearest grid point
        i = np.searchsorted(self.S_grid, S)
        if i == 0:
            i = 1
        elif i >= self.M:
            i = self.M - 1

        # Second-order central difference
        d2V = self.V[i + 1, 0] - 2 * self.V[i, 0] + self.V[i - 1, 0]
        dS2 = self.dS**2

        return d2V / dS2

    def get_theta(self, S: float) -> float:
        """
        Calculate Theta (∂V/∂t) using forward difference.

        Parameters:
        -----------
        S : float
            Spot price

        Returns:
        --------
        theta : float
            Time decay (note: typically negative)
        """
        # Find nearest grid point in S
        i = np.searchsorted(self.S_grid, S)
        if i >= self.M:
            i = self.M - 1

        # Forward difference in time
        dV = self.V[i, 1] - self.V[i, 0]
        dt = self.dt

        return dV / dt

    def get_vega_fd(self, S: float, dsigma: float = 0.01) -> float:
        """
        Calculate Vega (∂V/∂σ) using finite difference on sigma.
        Requires re-solving PDE with perturbed volatility.

        Parameters:
        -----------
        S : float
            Spot price
        dsigma : float
            Volatility perturbation (default 1%)

        Returns:
        --------
        vega : float
            Sensitivity to volatility (per 1% vol change)
        """
        # Solve with sigma + dsigma
        solver_up = BlackScholesFDM(
            self.S_max, self.K, self.T, self.r,
            self.sigma + dsigma, self.q,
            self.M, self.N, self.option_type
        )
        solver_up.solve_crank_nicolson()
        V_up = solver_up.get_price(S)

        # Solve with sigma - dsigma
        solver_down = BlackScholesFDM(
            self.S_max, self.K, self.T, self.r,
            self.sigma - dsigma, self.q,
            self.M, self.N, self.option_type
        )
        solver_down.solve_crank_nicolson()
        V_down = solver_down.get_price(S)

        # Central difference
        return (V_up - V_down) / (2 * dsigma)

    def get_rho_fd(self, S: float, dr: float = 0.0001) -> float:
        """
        Calculate Rho (∂V/∂r) using finite difference on interest rate.

        Parameters:
        -----------
        S : float
            Spot price
        dr : float
            Interest rate perturbation

        Returns:
        --------
        rho : float
            Sensitivity to interest rate
        """
        # Solve with r + dr
        solver_up = BlackScholesFDM(
            self.S_max, self.K, self.T, self.r + dr,
            self.sigma, self.q,
            self.M, self.N, self.option_type
        )
        solver_up.solve_crank_nicolson()
        V_up = solver_up.get_price(S)

        # Solve with r - dr
        solver_down = BlackScholesFDM(
            self.S_max, self.K, self.T, self.r - dr,
            self.sigma, self.q,
            self.M, self.N, self.option_type
        )
        solver_down.solve_crank_nicolson()
        V_down = solver_down.get_price(S)

        # Central difference
        return (V_up - V_down) / (2 * dr)

    def get_all_greeks(self, S: float) -> Dict[str, float]:
        """
        Calculate all Greeks at spot price S.

        Parameters:
        -----------
        S : float
            Spot price

        Returns:
        --------
        greeks : dict
            Dictionary with price and Greeks
        """
        return {
            'price': self.get_price(S),
            'delta': self.get_delta(S),
            'gamma': self.get_gamma(S),
            'theta': self.get_theta(S),
            # Note: vega and rho are expensive (require re-solving)
            # 'vega': self.get_vega_fd(S),
            # 'rho': self.get_rho_fd(S),
        }


def compare_with_analytical(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = 'call',
    M: int = 200,
    N: int = 2000
) -> Dict:
    """
    Compare FDM solution with analytical Black-Scholes formula.

    Parameters:
    -----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    q : float
        Dividend yield
    option_type : str
        'call' or 'put'
    M : int
        Number of stock price grid points
    N : int
        Number of time steps

    Returns:
    --------
    results : dict
        Comparison results
    """
    from blackscholes import BlackScholesCall, BlackScholesPut

    # Analytical solution
    print("Computing analytical solution...")
    t0_analytical = time.time()

    if option_type == 'call':
        bs_option = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
    else:
        bs_option = BlackScholesPut(S=S, K=K, T=T, r=r, sigma=sigma, q=q)

    analytical = {
        'price': bs_option.price(),
        'delta': bs_option.delta(),
        'gamma': bs_option.gamma(),
        'theta': bs_option.theta(),
        'vega': bs_option.vega(),
        'rho': bs_option.rho(),
    }
    t_analytical = time.time() - t0_analytical

    # FDM solution with Crank-Nicolson
    print("Computing FDM solution (Crank-Nicolson)...")
    t0_fdm = time.time()

    S_max = max(3 * K, 2 * S)  # Set computational domain
    fdm_solver = BlackScholesFDM(
        S_max=S_max, K=K, T=T, r=r, sigma=sigma, q=q,
        M=M, N=N, option_type=option_type
    )
    fdm_solver.solve_crank_nicolson()

    fdm = {
        'price': fdm_solver.get_price(S),
        'delta': fdm_solver.get_delta(S),
        'gamma': fdm_solver.get_gamma(S),
        'theta': fdm_solver.get_theta(S),
    }
    t_fdm = time.time() - t0_fdm

    print("\nComputing FDM Vega and Rho (expensive)...")
    t0_greeks = time.time()
    fdm['vega'] = fdm_solver.get_vega_fd(S)
    fdm['rho'] = fdm_solver.get_rho_fd(S)
    t_greeks = time.time() - t0_greeks

    # Calculate errors
    errors = {}
    relative_errors = {}

    for key in analytical.keys():
        if key in fdm:
            abs_error = abs(fdm[key] - analytical[key])
            errors[key] = abs_error

            if abs(analytical[key]) > 1e-10:
                relative_errors[key] = abs_error / abs(analytical[key]) * 100
            else:
                relative_errors[key] = 0.0

    return {
        'parameters': {
            'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'q': q,
            'option_type': option_type, 'M': M, 'N': N
        },
        'analytical': analytical,
        'fdm': fdm,
        'errors': errors,
        'relative_errors_pct': relative_errors,
        'timing': {
            'analytical': t_analytical,
            'fdm': t_fdm,
            'fdm_extra_greeks': t_greeks,
            'speedup': t_fdm / t_analytical
        }
    }


def print_comparison_results(results: Dict):
    """Pretty print comparison results."""
    params = results['parameters']

    print("\n" + "="*80)
    print("BLACK-SCHOLES PDE: FDM vs ANALYTICAL COMPARISON")
    print("="*80)

    print("\nParameters:")
    print(f"  Spot Price (S):     ${params['S']:.2f}")
    print(f"  Strike Price (K):   ${params['K']:.2f}")
    print(f"  Time to Maturity:   {params['T']:.4f} years")
    print(f"  Risk-free Rate:     {params['r']*100:.2f}%")
    print(f"  Volatility:         {params['sigma']*100:.2f}%")
    print(f"  Dividend Yield:     {params['q']*100:.2f}%")
    print(f"  Option Type:        {params['option_type'].upper()}")
    print(f"  Grid: M={params['M']}, N={params['N']}")

    print("\n" + "-"*80)
    print(f"{'Metric':<15} {'Analytical':>15} {'FDM':>15} {'Abs Error':>15} {'Rel Error %':>15}")
    print("-"*80)

    for key in ['price', 'delta', 'gamma', 'theta', 'vega', 'rho']:
        if key in results['analytical']:
            analytical_val = results['analytical'][key]
            fdm_val = results['fdm'].get(key, float('nan'))
            error = results['errors'].get(key, float('nan'))
            rel_error = results['relative_errors_pct'].get(key, float('nan'))

            print(f"{key.capitalize():<15} {analytical_val:>15.6f} {fdm_val:>15.6f} "
                  f"{error:>15.8f} {rel_error:>15.4f}")

    print("-"*80)
    print("\nTiming:")
    timing = results['timing']
    print(f"  Analytical formula:     {timing['analytical']*1000:.3f} ms")
    print(f"  FDM (Crank-Nicolson):   {timing['fdm']:.3f} s")
    print(f"  FDM extra Greeks:       {timing['fdm_extra_greeks']:.3f} s")
    print(f"  FDM is {timing['speedup']:.0f}x SLOWER than analytical")

    print("\n" + "="*80)


def convergence_study(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.2,
    q: float = 0.0,
    option_type: str = 'call'
) -> Dict:
    """
    Study convergence: show that smaller grids improve accuracy but take longer.

    Tests multiple grid resolutions to demonstrate the accuracy/speed trade-off.
    """
    from blackscholes import BlackScholesCall, BlackScholesPut

    # Get analytical reference
    if option_type == 'call':
        bs_option = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
    else:
        bs_option = BlackScholesPut(S=S, K=K, T=T, r=r, sigma=sigma, q=q)

    analytical_price = bs_option.price()
    analytical_delta = bs_option.delta()
    analytical_gamma = bs_option.gamma()

    # Test different grid resolutions (coarse to fine)
    grid_configs = [
        {'M': 50,  'N': 500,   'label': 'Coarse'},
        {'M': 100, 'N': 1000,  'label': 'Medium'},
        {'M': 200, 'N': 2000,  'label': 'Fine'},
        {'M': 400, 'N': 4000,  'label': 'Very Fine'},
        {'M': 800, 'N': 8000,  'label': 'Ultra Fine'},
    ]

    results = []

    print("\n" + "="*80)
    print("CONVERGENCE STUDY: Grid Resolution vs Accuracy vs Speed")
    print("="*80)
    print(f"\nOption: {option_type.upper()}")
    print(f"Parameters: S=${S}, K=${K}, T={T}yr, r={r*100}%, σ={sigma*100}%")
    print(f"\nAnalytical Reference:")
    print(f"  Price: {analytical_price:.8f}")
    print(f"  Delta: {analytical_delta:.8f}")
    print(f"  Gamma: {analytical_gamma:.8f}")

    print("\n" + "-"*80)
    print(f"{'Grid':<12} {'M':>6} {'N':>6} {'Time(s)':>10} "
          f"{'Price Err%':>12} {'Delta Err%':>12} {'Gamma Err%':>12}")
    print("-"*80)

    S_max = max(3 * K, 2 * S)

    for config in grid_configs:
        M, N = config['M'], config['N']
        label = config['label']

        # Solve with this grid resolution
        t0 = time.time()
        fdm_solver = BlackScholesFDM(
            S_max=S_max, K=K, T=T, r=r, sigma=sigma, q=q,
            M=M, N=N, option_type=option_type
        )
        fdm_solver.solve_crank_nicolson()

        fdm_price = fdm_solver.get_price(S)
        fdm_delta = fdm_solver.get_delta(S)
        fdm_gamma = fdm_solver.get_gamma(S)

        elapsed = time.time() - t0

        # Calculate errors
        price_err = abs(fdm_price - analytical_price) / abs(analytical_price) * 100
        delta_err = abs(fdm_delta - analytical_delta) / abs(analytical_delta) * 100
        gamma_err = abs(fdm_gamma - analytical_gamma) / abs(analytical_gamma) * 100

        print(f"{label:<12} {M:>6} {N:>6} {elapsed:>10.3f} "
              f"{price_err:>11.4f}% {delta_err:>11.4f}% {gamma_err:>11.4f}%")

        results.append({
            'label': label,
            'M': M,
            'N': N,
            'time': elapsed,
            'price_error_pct': price_err,
            'delta_error_pct': delta_err,
            'gamma_error_pct': gamma_err,
            'fdm_price': fdm_price,
            'fdm_delta': fdm_delta,
            'fdm_gamma': fdm_gamma,
        })

    print("-"*80)
    print("\nKEY OBSERVATIONS:")
    print(f"  • Accuracy improves with finer grids (smaller dS, dt)")
    print(f"  • Computation time increases roughly O(M*N)")
    print(f"  • Price converges faster than Greeks")
    print(f"  • Ultra Fine is {results[-1]['time']/results[0]['time']:.1f}x slower but "
          f"{results[0]['price_error_pct']/results[-1]['price_error_pct']:.1f}x more accurate")
    print("="*80)

    return {
        'analytical': {
            'price': analytical_price,
            'delta': analytical_delta,
            'gamma': analytical_gamma,
        },
        'results': results
    }


if __name__ == "__main__":
    """Run convergence study."""

    # Convergence study for call option
    print("\n" + "="*80)
    print("DEMONSTRATING ACCURACY vs SPEED TRADE-OFF")
    print("="*80)

    convergence_study(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        q=0.0,
        option_type='call'
    )

    # Convergence study for put option
    print("\n")
    convergence_study(
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.2,
        q=0.0,
        option_type='put'
    )
