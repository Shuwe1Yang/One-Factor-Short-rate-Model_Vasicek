import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.stats as st



class VasicekCZCB():
    def __init__(self):
        pass

    def vasicek_czcb_values(self, r0_, R_, ratio_, T_, sigma_, kappa_, theta_, M_, prob_=1e-6, max_iter_=10, max_grid_construct_=0.25, rs_=None):
        """
        Return list of short rates and list of option prices
        """
        r_min, dr, N, dt = self._vasicek_params(r0_, M_, sigma_, kappa_, theta_, T_, prob_, max_grid_construct_, rs_)
        r = np.r_[0: N] * dr + r_min
        v_m1 = np.ones(N)

        for i in range(1, M_+1):
            K = self._exercise_call_price(R_, ratio_, i*dt)
            eex = np.ones(N) * K
            sub_diag, diag, super_diag = self._vasicek_diagonals(sigma_, kappa_, theta_, r_min, dr, N, dt)
            v_m1, iter = self._iterate(sub_diag, diag, super_diag, v_m1, eex, max_iter_)

        return r, v_m1

    def _vasicek_params(self, r0_, M_, sigma_, kappa_, theta_, T_, prob_, max_grid_construct_, rs_=None):
        """ Return r_min, dr, N, dt
        """
        (rmin, rmax) = (rs_[0], rs_[-1]) if rs_ is not None else self.vasicek_limits(r0_, sigma_, kappa_, theta_, T_, prob_)

        dt = T_ / float(M_)
        N = self._calc_N(max_grid_construct_, dt, sigma_, rmax, rmin)
        dr = (rmax - rmin) / (N-1)

        return rmin, dr, N, dt

    def _calc_N(self, max_grid_construct_, dt_, sigma_, r_max_, r_min_):
        N = 0
        while True:
            N += 1
            grid_interval = dt_*(sigma_**2)/(((r_max_-r_min_)/float(N))**2)
            if grid_interval > max_grid_construct_:
                break
        return N

    def _vasicek_limits(self, r0_, sigma_, kappa_, theta_, T_, prob_=1e-6):
        """ Return Rmin and Rmax"""
        expectation = theta_ + (r0_ - theta_) * np.exp(-kappa_*T_)
        variance = (sigma_**2)*T_ if kappa_==0 else (sigma_**2)/(2*kappa_)*(1-np.exp(-2*kappa_*T_))
        std = np.sqrt(variance)
        rmin = st.norm.ppf(prob_, expectation, std)
        rmax = st.norm.ppf(1-prob_, expectation, std)
        return rmin, rmax

    def _vasicek_diagonals(self, sigma_, kappa_, theta_, r_min_, dr_, N_, dt_):
        """ Return the diagonals of implicit scheme"""
        rn = np.r_[0: N_] * dr_ + r_min_
        sub_diag = kappa_*(theta_-rn)*dt_/(2*dr_) - 0.5*(sigma_**2)*dt_/(dr_**2)
        diag = 1 + rn*dt_ + sigma_**2*dt_/(dr_**2)
        super_diag = -kappa_*(theta_-rn)*dt_/(2*dr_) - 0.5*(sigma_**2)*dt_/(dr_**2)

        # Implement BC
        if N_ > 0:
            v_subd0 = sub_diag[0]
            super_diag[0] = super_diag[0] - sub_diag[0]
            diag[0] += 2*v_subd0
            sub_diag[0] = 0

        if N_ > 1:
            v_superd_last = super_diag[-1]
            super_diag[-1] = super_diag[-1] - sub_diag[-1]
            diag[-1] += 2*v_superd_last
            super_diag[-1] = 0

        return sub_diag, diag, super_diag

    @staticmethod
    def _check_exercise(V_, eex_):
        """ Return list of Booleans"""
        return V_ > eex_

    @staticmethod
    def _exercise_call_price(R_, ratio_, tau_):
        """ Return the discounted value of strike price as a ratio
        """
        return ratio_*np.exp(-R_*tau_)

    def _vasicek_policy_diagonals(self, sub_diag_, diag_, super_diag_, v_old_, v_new_, eex_):
        """ Return new sub_diagonal, diagonal, super_diagonal"""
        has_early_exercise = self._check_exercise(v_new_, eex_)
        sub_diag_[has_early_exercise] = 0
        super_diag_[has_early_exercise] = 0
        policy = v_old_/eex_
        policy_values = policy[has_early_exercise]
        diag_[has_early_exercise] = policy_values
        return sub_diag_, diag_, super_diag_

    def _iterate(self, sub_diag_, diag_, super_diag_, v_old_, eex_, max_iter_=10):
        """ Return the number of iterations performed"""
        v_m1 = v_old_
        v_m = v_old_
        change = np.zeros(len(v_old_))
        pre_changes = np.zeros(len(v_old_))

        iter = 0
        while iter <= max_iter_:
            iter += 1
            v_m1 = self._tridiagonal_solve(sub_diag_, diag_, super_diag_, v_old_)
            sub_diag_, diag_, super_diag_ = self._vasicek_policy_diagonals(sub_diag_, diag_, super_diag_, v_old_, v_m1, eex_)

            is_eex = self._check_exercise(v_m1, eex_)
            change[is_eex] = 1

            if iter > 1:
                change[v_m1 != v_m] = 1

            is_no_more_eex = False if True in is_eex else True
            if is_no_more_eex:
                break

            v_m1[is_eex] = eex_[is_eex]
            changes = (change == pre_changes)

            is_no_further_changes = all((x == 1) for x in changes)
            if is_no_further_changes:
                break

            pre_changes = change
            v_m = v_m1

        return v_m1, (iter-1)

    @staticmethod
    def _tridiagonal_solve(a_, b_, c_, d_):
        """ Thomas algo for solving tridiagonal systems of equations"""
        nf = len(a_)  # Number of equation
        ac, bc, cc, dc = map(np.array, (a_, b_, c_, d_))

        for i in range(1, nf):
            w = ac[i] / bc[i-1]
            bc[i] = bc[i] - w * cc[i-1]
            dc[i] = dc[i] - w * dc[i-1]

        xc = ac
        xc[-1] = dc[-1] / bc[-1]

        for j in range(nf-2, -1, -1):
            xc[j] = (dc[j] - cc[j] * xc[j+1]) / bc[j]

        del bc, cc, dc
        return xc


def vasicek_model_test():
    r0 = 0.05
    R = 0.05
    ratio = 0.95
    sigma = 0.03
    kappa = 0.15
    theta = 0.05
    prob = 1e-6
    M = 250
    max_policy_iter = 10
    grid_struct_interval = 0.25
    rs = np.r_[0.0:2.0:0.1]

    Vasicek = VasicekCZCB()

    r, vals = Vasicek.vasicek_czcb_values(r0, R, ratio, 1., sigma, kappa, theta, M,
                                          prob, max_policy_iter, grid_struct_interval, rs)
    import matplotlib.pyplot as plt
    plt.title("Callable Zero Coupon Bond Values by r")
    plt.plot(r, vals, label='1 yr')

    for T in [5., 7., 10., 20.]:
        r, vals = Vasicek.vasicek_czcb_values(r0, R, ratio, T, sigma, kappa, theta, M,
                                              prob, max_policy_iter, grid_struct_interval, rs)
        plt.plot(r, vals, label=str(T) + ' yr', linestyle="--", marker=".")

    plt.ylabel("Value ($)")
    plt.xlabel("r")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    vasicek_model_test()


if __name__ == '__main__':
    main()
