import numpy as np
from scipy import constants as sp
from scipy.integrate import solve_ivp


class SimMagnetism:
    # This class holds all static methods for calculation of the magnetization dynamics.

    @staticmethod
    def equilibrate_mag(j_sam, spin_sam, arbsc_sam, s_up_eig_sq_sam, s_dn_eig_sq_sam, fs0, te0, tp0,
                        el_mag_mask, ms_sam, mag_num):
        # This method equilibrates the magnetization to its mean field equilibrium value at the initial temperature.

        # Input:
        # A bunch of parameters from the sample class, initialized in SimDynamics.get_t_m_maps()

        # Returns:
        # fss_eq_flat (numpy array). 1d-array of the occupation of the spin-z-components for the whole magnetic sample
        # in equilibrium with the initial temperature profile (for now only uniform) in a flattened format

        # increase the damping to speed up the equilibration process:
        arbsc_sam_eq = arbsc_sam*1e3

        # call solver to equilibrate magnetization:
        eq_sol = solve_ivp(lambda t, fs: SimMagnetism.get_m_eq_increments(fs, j_sam, spin_sam, arbsc_sam_eq,
                                                                          s_up_eig_sq_sam, s_dn_eig_sq_sam,
                                                                          te0, tp0, el_mag_mask, ms_sam, mag_num),
                           y0=fs0, t_span=(0, 10e-12), method='RK45')

        # flatten the output for further calculation:
        fs_eq_flat = eq_sol.y.T[-1]

        # the following only to inform the user about the state:
        fs_eq = np.reshape(fs_eq_flat, (mag_num, (int(2 * spin_sam[0] + 1))))
        mag_eq = SimMagnetism.get_mag(fs_eq, ms_sam, spin_sam)
        print('Equilibration phase done.')
        print('Equilibrium magnetization in magnetic layers: ' + str(mag_eq))
        print('at initial temperature: ' + str(tp0) + ' K')
        print()

        # return the equilibrium spin occupation:
        return fs_eq_flat

    @staticmethod
    def get_m_eq_increments(fss_flat, j_sam, spin_sam, arbsc_sam_eq, s_up_eig_sq_sam, s_dn_eig_sq_sam, te0, tp0,
                            el_mag_mask, ms_sam, mag_num):
        # This method handles the computation of increments of the spin occupations in the equilibration process.
        # It is the equivalent of get_t_m_increments in the equilibration phase.

        # Input:
        # A bunch of parameters from the sample and pulse objects. See documentation in the respective methods

        # Returns:
        # dfs_dt.flatten() (numpy array). 1d-array of the flattened equilibrium spin occupations.

        fss = np.reshape(fss_flat, (mag_num, (int(2 * spin_sam[0] + 1))))
        mag = SimMagnetism.get_mag(fss, ms_sam, spin_sam)
        dfs_dt = SimMagnetism.mag_occ_dyn(j_sam, spin_sam, arbsc_sam_eq, s_up_eig_sq_sam, s_dn_eig_sq_sam, mag, fss,
                                          te0, tp0, el_mag_mask)
        return dfs_dt.flatten()

    @staticmethod
    def mag_occ_dyn(j_sam, spin_sam, arbsc_sam, s_up_eig_sq_sam, s_dn_eig_sq_sam, mag, fs, te, tp, el_mag_mask):
        # This method computes the increments of spin-level occupation for the whole magnetic part of the sample.

        # Input:
        # j_sam (numpy array). 1d-array of exchange coupling J in all magnetic layers.
        # spin_sam (numpy array). 1d-array of the effective spin in all magnetic layers.
        # arbsc_sam (numpy array). 1d-array of pre-factors for the magnetization increments in all magnetic layers
        # s_up_eig_sq_sam (numpy array). 2d-array of the squared spin-up ladder operator in all magnetic layers
        # s_dn_eig_sq_sam (numpy array). 2d-array of the squared spin-down ladder operator in all magnetic layers
        # mag (numpy array). 1d-array of the magnetization in all magnetic layers.
        # fs (numpy array). 2d-array of the spin-level occupations (second dimension)
        # in all magnetic layers (first dimension).
        # te (numpy array). 1d-array of the current electron temperatures.
        # tp (numpy array). 1d-array of the current phonon temperatures.

        # Returns:
        # dfs_dt (numpy array). 2d-array of the increments in spin-level occupation in all magnetic layers

        h_mf = np.multiply(j_sam, mag)
        eta = np.divide(h_mf, np.multiply(2 * spin_sam * sp.k, te[el_mag_mask])).astype(float)
        incr_pref = arbsc_sam * tp * h_mf / 4 / spin_sam / np.sinh(eta)

        fs_up = np.multiply(s_up_eig_sq_sam, fs)
        fs_dn = np.multiply(s_dn_eig_sq_sam, fs)
        rate_up_fac = incr_pref * np.exp(-eta)
        rate_dn_fac = incr_pref * np.exp(eta)

        rate_up_loss = np.multiply(rate_up_fac[..., np.newaxis], fs_up)
        rate_dn_loss = np.multiply(rate_dn_fac[..., np.newaxis], fs_dn)

        rate_up_gain = np.roll(rate_up_loss, 1)
        rate_dn_gain = np.roll(rate_dn_loss, -1)

        dfs_dt = rate_up_gain + rate_dn_gain - rate_up_loss - rate_dn_loss
        return dfs_dt

    @staticmethod
    def get_mag(fs, ms_sam, spin_sam):
        # This method computes magnetization (increments) on the basis of the spin-level occupation (increments)
        # in all magnetic layers

        # Input:
        # fs (numpy array). 2d-array of the spin-level occupation (increments) (second dimension)
        # in all magnetic layers (first dimension)
        # ms_sam (numpy array). 2d-array of spin-levels (second dimension) of all magnetic layers (first dimension)
        # spin_sam (numpy array). 1d-array of effective spins of all magnetic layers

        # Returns:
        # mag (numpy array). 1d-array of the magnetization (increments) of all magnetic layers

        mag = -np.divide(np.sum(ms_sam * fs, axis=-1), spin_sam)

        return mag

    @staticmethod
    def get_mag_en_incr(mag, dm_dt, j_sam, vat_sam):
        # This method computes the increments of mean field magnetic energy density J*m*dm/dt/V_at

        # Input:
        # mag (numpy array). 1d-array of magnetization of all magnetic layers
        # dm_dt (numpy array). 1d-array of magnetization increments of all magnetic layers
        # j_sam (numpy array). 1d-array of mean field exchange constant of all magnetic layers
        # vat_sam (sumpy array). 1d-array of atomic volume of all magnetic layers

        # Returns
        # W_es (numpy array). 1d-array of the mean field magnetic energy increments of all magnetic layers

        return mag * dm_dt * j_sam / vat_sam

    @staticmethod
    def separate(fss_flat, mag_num, spin_sam):
        # This method stacks the occupations of spin levels in
        fss = np.reshape(fss_flat, (mag_num, (int(2 * spin_sam[0] + 1))))

        return fss


class SimNoMag(SimMagnetism):
    # This class defines the trivial magnetization methods in case the simulated sample has no magnetic subsystem

    @staticmethod
    def equilibrate_mag(j_sam, spin_sam, arbsc_sam, s_up_eig_sq_sam, s_dn_eig_sq_sam, fs0, te0, tp0,
                        el_mag_mask, ms_sam, mag_num):
        pass
    @staticmethod
    def get_m_eq_increments(fss_flat, j_sam, spin_sam, arbsc_sam_eq, s_up_eig_sq_sam, s_dn_eig_sq_sam, te0, tp0,
                            el_mag_mask, ms_sam, mag_num):
        pass

    @staticmethod
    def mag_occ_dyn(j_sam, spin_sam, arbsc_sam, s_up_eig_sq_sam, s_dn_eig_sq_sam, mag, fs, te, tp, el_mag_mask):
        return np.array([None])

    @staticmethod
    def get_mag(fs, ms_sam, spin_sam):
        pass

    @staticmethod
    def get_mag_en_incr(mag, dm_dt, j_sam, vat_sam):
        return 0

    @staticmethod
    def separate(fss_flat, mag_num, spin_sam):
        return np.array([None])
