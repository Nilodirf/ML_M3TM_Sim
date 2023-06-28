import numpy as np
from scipy import constants as sp
from finderb import finderb
from finderb import finderb_nest
from scipy.integrate import solve_ivp

class SimDynamics:
    def __init__(self, sample, pulse, end_time):
        # Input:
        # sample (object). The sample in use
        # pulse (object). The pulse excitation in use
        # end_time (float). Final time of simulation (including pulse delay) in s
        # Also returns:

        self.Sam = sample
        self.Pulse = pulse
        self.end_time = end_time

    def get_t_m_maps(self):
        # This method initiates all parameters needed for the dynamical simulation
        # and calls the solve_ivp function to run the sim.

        # Input:
        # self (object). The simulation object, taking sample and pulse as input

        # Returns:

        len_sam = self.Sam.len
        el_mask = self.Sam.el_mask
        mag_mask = self.Sam.magdyn_mask
        ce_gamma_sam = self.Sam.get_params('ce_gamma')[el_mask]
        mats, mat_ind = self.Sam.mats, self.Sam.matind
        cp_sam_grid, cp_sam = self.Sam.get_params('cp_T')
        gep_sam = self.Sam.get_params('gep')[el_mask]
        pulse_time_grid, pulse_map = self.Pulse.get_pulse_map()
        dz_sam = self.Sam.get_params('dz')
        kappa_e_sam = self.Sam.get_params('kappae')
        kappa_p_sam = self.Sam.get_params('kappap')
        kappa_e_dz_pref = np.divide(kappa_p_sam, np.power(dz_sam, 2))[el_mask]
        kappa_p_dz_pref = np.divide(kappa_e_sam, np.power(dz_sam, 2))
        j_sam = self.Sam.get_params('J')[mag_mask]
        spin_sam = self.Sam.get_params('spin')[mag_mask]
        arbsc_sam = self.Sam.get_params('arbsc')[mag_mask]
        s_up_eig_sq_sam = self.Sam.get_params('s_up_eig_squared')[mag_mask]
        s_dn_eig_sq_sam = self.Sam.get_params('s_dn_eig_squared')[mag_mask]
        ms_sam = self.Sam.get_params('ms')[mag_mask]
        mag_num = self.Sam.mag_num
        vat_sam = self.Sam.get_params('vat')[mag_mask]



    @staticmethod
    def get_t_m_increments(timestep, te_tp_fs_flat, len_sam, mats, mat_ind, mag_mask, el_mask, ce_gamma_sam,
                           cp_sam_grid, cp_sam, gep_sam, pulse_map, pulse_time_grid, kappa_e_dz_pref,
                           kappa_p_dz_pref, j_sam, spin_sam, arbsc_sam, s_up_eig_sq_sam, s_dn_eig_sq_sam,
                           ms_sam, mag_num, vat_sam):
        te = te_tp_fs_flat[:len_sam]
        tp = te_tp_fs_flat[len_sam:2*len_sam]
        fss_flat = te_tp_fs_flat[2*len_sam:]
        fss = fss_flat.reshape(mag_num, (2 * spin_sam[0] + 1))

        mag = SimDynamics.get_mag(fss, ms_sam, spin_sam)
        dfs_dt = SimDynamics.mag_occ_dyn(j_sam, spin_sam, arbsc_sam, s_up_eig_sq_sam, s_dn_eig_sq_sam,
                                         mag, fss, te[mag_mask], tp[mag_mask])
        dm_dt = SimDynamics.get_mag(dfs_dt, ms_sam, spin_sam)

        mag_en_t = SimDynamics.get_mag_en_incr(mag, dm_dt, j_sam, vat_sam)
        cp_sam_t = np.zeros_like(len_sam)
        for i, ind_list in enumerate(mat_ind):
            cp_sam_grid_t = finderb(te[ind_list], cp_sam_grid[i])
            cp_sam_t[ind_list] = cp_sam[cp_sam_grid_t]
        pulse_time = finderb(timestep, pulse_time_grid)
        pulse_t = pulse_map[pulse_time][el_mask]
        ce_sam_t = np.multiply(ce_gamma_sam[el_mask], te[el_mask])

        dte_dt, dtp_dt = SimDynamics.loc_temp_dyn(ce_sam_t, cp_sam_t, gep_sam, te[el_mask], tp[el_mask],
                                                          pulse_t, mag_en_t)
        dte_dt += SimDynamics.electron_diffusion(kappa_e_dz_pref, ce_sam_t, te[el_mask], tp[el_mask])
        dtp_dt += SimDynamics.phonon_diffusion(kappa_p_dz_pref, cp_sam_t, tp)

        dfs_dt_flat = dfs_dt.flatten()
        dTs = np.concatenate((dte_dt, dtp_dt))
        map_increments = np.concatenate((dTs, dfs_dt_flat))

        return map_increments

    @staticmethod
    def loc_temp_dyn(ce_sam_t, cp_sam_t, gep_sam, te, tp, pulse_t, mag_en_t):
        # This method computes the local temperature dynamics, namely electron-phonon-coupling, electron-spin-coupling
        # and the pulse excitation. The temperature dependence of all parameters shall be precomputed for the input.
        # dte_dt= 1/ce*(gep*(tp-te)+S(t)+ J*m/V_at*dm/dt)
        # dtp_dt= 1/cp*(gep*(te-tp))

        # Input:
        # ce_sam_t (numpy array). 1d-array holding the electronic heat capacities at the current temperature.
        # cp_sam_t (numpy array). 1d-array holding the phononic heat capacities at the current temperature.
        # gep_sam (numpy array). 1d-array holding the electron phonon coupling at the current temperature.
        # te (numpy array). 1d-array of the current electron temperatures.
        # tp (numpy array). 1d-array of the current phonon temperatures.
        # pulse_t (numpy array). 1d-array of the current pulse power.
        # mag_en_t (numpy array). 1d-array of magnetic energy losses/gains: J*m/V_at*dm/dt

        # Returns:
        # de_dt (numpy array). 1-d array of the electron temperature increments upon e-p-coupling and pulse excitation.
        # Returns 0 for layers where no electron dynamics happen.
        # dp_dt (numpy array). 1-d array of the phonon temperature increments upon e-p-coupling.
        # Returns 0 where there is no electron system to couple to.

        de_dt = np.zeros_like(te)
        dp_dt = np.zeros_like(tp)

        e_p_coupling = np.multiply(gep_sam, (tp- te))

        de_dt += e_p_coupling
        de_dt += pulse_t + mag_en_t
        de_dt = np.divide(de_dt, ce_sam_t)

        dp_dt -= np.divide(e_p_coupling, cp_sam_t)

        return de_dt, dp_dt

    @staticmethod
    def electron_diffusion(kappa_e_dz_pref, ce_sam_t, te, tp):
        # This method computes the electron heat diffusion in the layers where there is an electron bath.
        # dte_dt = kappa_e * te / tp * (laplace(te)) + kappa_e * grad(te / tp) * grad(te)

        # Input:
        # kappa_e_dz_pref (numpy array). 1d-array of kappa_e/dz**2 for the relevant layers
        # ce_sam_t (numpy array). 1d-array holding the electronic heat capacities at the current temperature.
        # te (numpy array). 1d-array of the current electron temperatures in the relevant layers.
        # tp (numpy array). 1d-array of the current phonon temperatures in the relevant layers.

        # Returns:
        # dte_dt_diff (numpy array). 1d-array of the electron temperature increments upon the effect of diffusion.

        te_diff_right = -np.concatenate((np.diff(te), np.zeros(1)))
        te_diff_left = -np.roll(te_diff_right, 1)
        te_double_diff = -np.concatenate((np.zeros(1), np.concatenate((np.diff(te,2), np.zeros(1)))))
        te_tp_double_diff = -np.concatenate((np.zeros(1), np.concatenate((np.diff(te/tp,2), np.zeros(1)))))

        term_1 = np.multiply(kappa_e_dz_pref * np.divide(te, tp), te_diff_right + te_diff_left)
        term_2 = 0.25 * kappa_e_dz_pref * te_tp_double_diff * te_double_diff

        dte_dt_diff = np.divide(term_1 + term_2, ce_sam_t)

        return dte_dt_diff

    @staticmethod
    def phonon_diffusion(kappa_p_dz_pref, cp_sam_t, tp):
        # This method computes the phononic heat diffusion in the whole sample.
        # dtp_dt = kappa_p * laplace(tp)

        # Input:
        # kappa_p_dz_pref (numpy array). 1d-array of kappa_p/dz**2 for the whole sample
        # cp_sam_t (numpy array). 1d-array holding the phononic heat capacities at the current temperature.
        # tp (numpy array). 1d-array of the current phonon temperatures.

        # Returns:
        # dtp_dt_diff (numpy array). 1d-array of the phonon temperature increments upon the effect of diffusion.

        tp_diff_right = -np.concatenate((np.diff(tp), np.zeros(1)))
        tp_diff_left = -np.roll(tp_diff_right, 1)

        dtp_dt_diff = np.divide(np.multiply(kappa_p_dz_pref, tp_diff_right+tp_diff_left), cp_sam_t)

        return dtp_dt_diff

    @staticmethod
    def mag_occ_dyn(j_sam, spin_sam, arbsc_sam, s_up_eig_sq_sam, s_dn_eig_sq_sam, mag, fs, te, tp):
        # This method computes the increments of spin-level occupation for the whole magnetic part of the sample.

        # Input:
        # j_sam (numpy array). 1d-array of exchange coupling J in all magnetic layers.
        # spin_sam (numpy array). 1d-array of the effective spin in all magnetic layers.
        # arbsc_sam (numpy array). 1d-array of pre-factors for the magnetization increments in all magnetic layers.
        # s_up_eig_sq_sam (numpy array). 2d-array of the squared spin-up ladder operator in all magnetic layers.
        # s_dn_eig_sq_sam (numpy array). 2d-array of the squared spin-down ladder operator in all magnetic layers.
        # mag (numpy array). 1d-array of the magnetization in all magnetic layers.
        # fs (numpy array). 2d-array of the spin-level occupations (second dimension)
        # in all magnetic layers (first dimension).
        # te (numpy array). 1d-array of the current electron temperatures.
        # tp (numpy array). 1d-array of the current phonon temperatures.

        # Returns:
        # dfs_dt (numpy array). 2d-array of the increments in spin-level occupation in all magnetic layers.

        h_mf = np.multiply(j_sam, mag)
        eta = np.divide(h_mf, np.multiply(2 * spin_sam * sp.k, te))
        incr_pref = np.divide(np.multiply(arbsc_sam, np.multiply(tp, h_mf)) / 4 / spin_sam, np.sinh(eta))

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

        return mag*dm_dt*j_sam/vat_sam
