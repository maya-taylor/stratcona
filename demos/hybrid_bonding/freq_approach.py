"""Frequentist confidence-interval demo for Black's equation.

This script uses the same EM MTTF observations as the Bayesian demo and
illustrates an important frequentist limitation: with data from only one test
condition (single temperature and single current density), the three Black's
equation parameters are not separately identifiable.
"""

import numpy as np
from matplotlib import pyplot as plt


BOLTZ_EV = 8.617e-5  # eV/K


def black_mttf(current_density, temp_k, ea, A, n):
    """Black's equation MTTF model.

    MTTF = A / J^n * exp(Ea / (kT))
    """
    return A / (current_density ** n) * np.exp(ea / (BOLTZ_EV * temp_k))


def run_frequentist_em_ci_demo():
    # Same rough points from electromigration.py
    mttf_data = np.array([211.0, 214.0, 215.0, 216.0, 218.0])

    # Same stress condition as electromigration.py
    temp_k = 350.0 + 273.15
    current_density = (20e-3 / 16) / (3.6**2 / 1e8)

    n_obs = mttf_data.size
    y_bar = float(np.mean(mttf_data))
    s = float(np.std(mttf_data, ddof=1))
    se_mean = s / np.sqrt(n_obs)

    # For n=5 (df=4), two-sided 95% t critical value.
    tcrit_95_df4 = 2.776445105

    print("=== Frequentist CI Demo for Black's Equation Parameters ===")
    print(f"Data: {mttf_data.tolist()}")
    print(f"Sample mean MTTF: {y_bar:.3f} h")
    print(f"Sample std MTTF:  {s:.3f} h")

    # -----------------------------------------------------------------
    # 1) Show why separate CIs for (Ea, n, A) are not estimable here.
    # -----------------------------------------------------------------
    # With one stress point, all observations share same (J, T), and
    # mu(Ea, n, A) depends only on this composite:
    #   c = ln(A) - n ln(J) + Ea / (kT)
    # so infinitely many (Ea, n, A) map to the same mean MTTF.
    lnJ = np.log(current_density)
    inv_kT = 1.0 / (BOLTZ_EV * temp_k)

    # Jacobian row for one observation wrt [Ea, n, A] at a reference point.
    ea_ref = 0.99
    n_ref = 1.36
    A_ref = 0.516
    mu_ref = black_mttf(current_density, temp_k, ea_ref, A_ref, n_ref)
    jac_row = np.array([
        mu_ref * inv_kT,
        -mu_ref * lnJ,
        mu_ref / A_ref,
    ])

    # Same jacobian row repeats for each observation -> rank 1 info matrix.
    J = np.tile(jac_row, (n_obs, 1))
    fisher_like = J.T @ J
    rank = int(np.linalg.matrix_rank(fisher_like))

    print("\nIdentifiability check (single stress condition):")
    print(f"Information matrix rank for [Ea, n, A]: {rank} (max 3)")
    print("Result: rank-deficient -> separate 95% CIs for Ea, n, and A are not")
    print("uniquely estimable from this dataset alone.")

    # -----------------------------------------------------------------
    # 2) Standard frequentist CI for identifiable quantity.
    # -----------------------------------------------------------------
    # Here, the condition-specific mean MTTF is identifiable.
    ci_mu_low = y_bar - tcrit_95_df4 * se_mean
    ci_mu_high = y_bar + tcrit_95_df4 * se_mean

    print("\n95% CI for identifiable mean MTTF at this (J, T):")
    print(f"mu in [{ci_mu_low:.3f}, {ci_mu_high:.3f}] hours")

    # Equivalent CI for composite parameter c = ln(mu) at this (J, T).
    c_hat = np.log(y_bar)
    c_ci = np.log([ci_mu_low, ci_mu_high])

    print("\nEquivalent CI for composite c = ln(A) - n ln(J) + Ea/(kT):")
    print(f"c_hat = {c_hat:.6f}")
    print(f"95% CI for c: [{c_ci[0]:.6f}, {c_ci[1]:.6f}]")

    # -----------------------------------------------------------------
    # 3) Plot only the standard confidence interval result.
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    x = np.arange(1, n_obs + 1)
    ax.scatter(x, mttf_data, color="black", s=80, label="Observed MTTF")
    ax.hlines(y=y_bar, xmin=0.7, xmax=n_obs + 0.3, color="#1f77b4", linewidth=3, label="Sample mean")
    ax.fill_between(
        [0.7, n_obs + 0.3],
        [ci_mu_low, ci_mu_low],
        [ci_mu_high, ci_mu_high],
        color="#1f77b4",
        alpha=0.2,
        label="95% confidence interval",
    )

    ax.set_xlim(0.5, n_obs + 0.5)
    ax.set_xlabel("Observation index", fontsize=13)
    ax.set_ylabel("MTTF (hours)", fontsize=13)
    ax.set_title("Standard Frequentist 95% Confidence Interval for Mean MTTF", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    filename = "electromigration_standard_confidence_interval.png"
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    print(f"\nStandard CI plot saved as '{filename}'")
    plt.show()


if __name__ == "__main__":
    run_frequentist_em_ci_demo()
    