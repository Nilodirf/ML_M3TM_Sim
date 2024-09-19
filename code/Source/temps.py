import numpy as np
from scipy import constants as sp

from .finderb import finderb


class SimTemperatures:
    # This class holds all static methods for calculation of the temperature dynamics.

    @staticmethod
    def loc_temp_dyn(ce_sam_t, cp_sam_t, gep_sam, te, tp, pulse_t, mag_en_t, el_mask,  el_mag_mask):
        # This method computes the local temperature dynamics, namely electron-phonon-coupling, electron-spin-coupling
        # and the pulse excitation. The temperature dependence of all parameters shall be precomputed for the input.
        # dte_dt= 1/ce*(gep*(tp-te)+S(t)+ J*m/V_at*dm/dt)
        # dtp_dt= 1/cp*(gep*(te-tp))

        # Input:
        # ce_sam_t (numpy array). 1d-array holding the electronic heat capacities at the current temperature
        # cp_sam_t (numpy array). 1d-array holding the phononic heat capacities at the current temperature
        # gep_sam (numpy array). 1d-array holding the electron phonon coupling constants
        # te (numpy array). 1d-array of the current electron temperatures
        # tp (numpy array). 1d-array of the current phonon temperatures
        # pulse_t (numpy array). 1d-array of the current pulse power
        # mag_en_t (numpy array). 1d-array of magnetic energy losses/gains: J*m/V_at*dm/dt
        # el_mag_mask (boolean array). 1d array of the length of numbers of layers that hold free electrons, True if
        # also magnetic, False if not

        # Returns:
        # de_dt (numpy array). 1-d array of the electron temperature increments upon e-p-coupling and pulse excitation
        # Returns 0 for layers where no electron dynamics happen
        # dp_dt (numpy array). 1-d array of the phonon temperature increments upon e-p-coupling
        # Returns 0 where there is no electron system to couple to

        e_p_coupling = np.multiply(gep_sam, (tp[el_mask] - te))

        de_dt = e_p_coupling + pulse_t
        de_dt[el_mag_mask] = mag_en_t
        de_dt = np.divide(de_dt, ce_sam_t)

        dp_dt = -np.divide(e_p_coupling, cp_sam_t[el_mask])

        return de_dt, dp_dt

    @staticmethod
    def electron_diffusion(kappa_e_dz_pref, ce_sam_t, te, tp):
        # This method computes the electron heat diffusion in the layers where there is an electron bath.
        # dte_dt = kappa_e * te / tp * (laplace(te)) + kappa_e * grad(te / tp) * grad(te)

        # Input:
        # kappa_e_dz_pref (numpy array). 1d-array of kappa_e/dz**2 for the relevant layers
        # ce_sam_t (numpy array). 1d-array holding the electronic heat capacities at the current temperature
        # te (numpy array). 1d-array of the current electron temperatures in the relevant layers
        # tp (numpy array). 1d-array of the current phonon temperatures in the relevant layers

        # Returns:
        # dte_dt_diff (numpy array). 1d-array of the electron temperature increments upon the effect of diffusion

        te_diff_right = np.concatenate((np.diff(te), np.zeros(1)))
        te_diff_left = -np.roll(te_diff_right, 1)
        te_double_diff = -np.concatenate((np.zeros(1), np.concatenate((np.diff(te, 2), np.zeros(1)))))
        te_tp_double_diff = -np.concatenate((np.zeros(1), np.concatenate((np.diff(te/tp, 2), np.zeros(1)))))

        term_1 = (np.multiply(kappa_e_dz_pref[:, 1], te_diff_right)
                  + np.multiply(kappa_e_dz_pref[:, 0], te_diff_left)) * np.divide(te, tp)
        term_2 = 0.25 * kappa_e_dz_pref[:, 0] * te_tp_double_diff * te_double_diff

        dte_dt_diff = np.divide(term_1 + term_2, ce_sam_t)

        return dte_dt_diff.astype(float)

    @staticmethod
    def phonon_diffusion(kappa_p_dz_pref, cp_sam_t, tp):
        # This method computes the phononic heat diffusion in the whole sample.
        # dtp_dt = kappa_p * laplace(tp)

        # Input:
        # kappa_p_dz_pref (numpy array). 1d-array of kappa_p/dz**2 for the whole sample
        # cp_sam_t (numpy array). 1d-array holding the phononic heat capacities at the current temperature
        # tp (numpy array). 1d-array of the current phonon temperatures

        # Returns:
        # dtp_dt_diff (numpy array). 1d-array of the phonon temperature increments upon the effect of diffusion

        tp_diff_right = np.concatenate((np.diff(tp), np.zeros(1)))
        tp_diff_left = -np.roll(tp_diff_right, 1)

        dtp_dt_diff = np.divide(np.multiply(kappa_p_dz_pref[:, 1], tp_diff_right) +
                                np.multiply(kappa_p_dz_pref[:, 0], tp_diff_left), cp_sam_t)

        return dtp_dt_diff.astype(float)

    @staticmethod
    def phonon_phonon_coupling(gpp_sam, cp_sam_t, cp2_sam_t, tp1, tp2):
        # This method computes phonon-phonon coupling for layers with two phononic subsystems.

        # Input:
        # gpp_sam (numpy array). 1d array with gpp_parameters for all necessary layers
        # cp_sam_t (numpy array). 1d-array holding the phononic heat capacities for ss1 at the current temperature
        # cp2_sam_t (numpy array). 1d-array holding the phononic heat capacities for ss2 at the current temperature
        # tp1 (numpy array). phonon temperature 1 of all layers with two phononic subsystems
        # tp2 (numpy array). phonon temperature 2 of all layers with two phononic subsystems

        # Returns:
        # dtp1_dt, dtp2_dt (numpy arrays). 1d arrays of the temperature rates due to phonon-phonon coupling

        p_p_coupling = gpp_sam*(tp2-tp1)

        dtp1_dt = np.divide(p_p_coupling, cp_sam_t)
        dtp2_dt = -np.divide(p_p_coupling, cp2_sam_t)

        return dtp1_dt, dtp2_dt

    @staticmethod
    def get_ce_t(te, ce_sam):
        # This method computes the electronic heat capacity of all layers of the sample. For now, only the linear
        # Sommerfeld approximation is implemented.

        # Input:
        # te (numpy array). 1d array of the electron temperatures of all relevant layers
        # ce_sam (numpy array). 1d array of the Sommerfeld coefficients of all relevant layers

        # Returns:
        # ce_sam*te (numpy array). 1d array of the Sommerfeld heat capacities of all relevant layers

        return ce_sam*te

    @staticmethod
    def get_cp_t(tp, cp_sam_grid, cp_sam, index_list):
        # This method looks up the closest precomputed values of phononic heat capacities available and returns the
        # phononic heat capacities for each requested layer in the sample

        # Input:
        # tp (numpy array). 1d array of the phononic temperatures in question
        # cp_sam_grid (numpy array). 2d array holding the temperatures for which the lattice heat capacity was
        # pre-computed (2nd dimension) for each material in the sample (1st dimension)
        # cp_sam (numpy array). 2d array with the corresponding values of heat capacity to cp_sam_grid
        # index_list (list). List of numpy arrays that holds the indices of layer positions (2nd dimension) of each
        # requested material in the sample (1st dimension)

        # Returns:
        # cp_sam_t (numpy array). 1d-array of the lattice heat capcities for each requested layer of the sample

        cp_sam_t = np.zeros_like(tp)
        for i, ind_list in enumerate(index_list):
            cp_sam_grid_t = finderb(tp[ind_list], cp_sam_grid[i])
            cp_sam_t[ind_list] = cp_sam[i][cp_sam_grid_t]

        return cp_sam_t
    @staticmethod
    def separate(te_tp_fs_flat, len_sam, len_sam_te, len_sam_tp2):
        # This method separates the electron and phonon temperatures from the 1 dimensional increments of all subsystems
        # computed in SimDynamics.get_t_m_increments()

        # Input:
        # te_p_fs_flat (numpy array). 1d array of the concatenated increments
        # len_sam, len_sam_te, len_sam_tp2 (ints). Number of layers in the sample, with electron temperature, with two
        # phonon systems

        # Returns:
        # te, tp (numpy arrays). 1d arrays of the electronic and phononic temperatures

        te = te_tp_fs_flat[:len_sam_te]
        tp = te_tp_fs_flat[len_sam_te:len_sam_te+len_sam+len_sam_tp2]

        return te, tp

    @staticmethod
    def temp_dyn(te, tp, ce_sam_t, cp_sam_t, gep_sam, pulse_t, mag_en_t, el_mask, el_mag_mask,
                 kappa_e_dz_pref, kappa_p_dz_pref, tp2_mask, gpp_sam, cp2_sam_t, len_sam_tp2, len_sam):
        # This method must be overwritten by selecting a proper temperature model in the main simulation function.
        # It is merely a placeholder.

        raise NotImplementedError("Subclasses must override the SimTemperatures.temp_dyn() method.")


class Sim11DD(SimTemperatures):
    @staticmethod
    def temp_dyn(te, tp, ce_sam_t, cp_sam_t, gep_sam, pulse_t, mag_en_t, el_mask, el_mag_mask,
                 kappa_e_dz_pref, kappa_p_dz_pref, tp2_mask, gpp_sam, cp2_sam_t, len_sam_tp2, len_sam):
        # This method combines the update functions defined in SimTemperatures for a setup of a 2TM with N>1 layers with
        # electronic subsystem, thus including diffusion calculation in e and p subsystems.

        dtp_dt = np.zeros(len_sam)

        dte_dt, dtp_dt[el_mask] = SimTemperatures.loc_temp_dyn(ce_sam_t, cp_sam_t, gep_sam, te, tp, pulse_t, mag_en_t,
                                                               el_mask,  el_mag_mask)

        dte_dt += SimTemperatures.electron_diffusion(kappa_e_dz_pref, ce_sam_t, te, tp[el_mask])

        dtp_dt += SimTemperatures.phonon_diffusion(kappa_p_dz_pref, cp_sam_t, tp)

        return dte_dt, dtp_dt


class Sim11LD(SimTemperatures):
    @staticmethod
    def temp_dyn(te, tp, ce_sam_t, cp_sam_t, gep_sam, pulse_t, mag_en_t, el_mask, el_mag_mask,
                 kappa_e_dz_pref, kappa_p_dz_pref, tp2_mask, gpp_sam, cp2_sam_t, len_sam_tp2, len_sam):
        # This method combines the update functions defined in SimTemperatures for a setup of a 2TM with N>1 layers with
        # only 1 layer with el. subsystem, thus including diffusion calculation in only p subsystem.

        dtp_dt = np.zeros(len_sam)

        dte_dt, dtp_dt[el_mask] = SimTemperatures.loc_temp_dyn(ce_sam_t, cp_sam_t, gep_sam, te, tp, pulse_t, mag_en_t,
                                                               el_mask,  el_mag_mask)

        dtp_dt += SimTemperatures.phonon_diffusion(kappa_p_dz_pref, cp_sam_t, tp)

        return dte_dt, dtp_dt


class Sim11LL(SimTemperatures):
    @staticmethod
    def temp_dyn(te, tp, ce_sam_t, cp_sam_t, gep_sam, pulse_t, mag_en_t, el_mask, el_mag_mask,
                 kappa_e_dz_pref, kappa_p_dz_pref, tp2_mask, gpp_sam, cp2_sam_t, len_sam_tp2, len_sam):
        # This method combines the update functions defined in SimTemperatures for a setup of a 2TM 1 layer,
        # thus no temperature diffusion is computed.

        dte_dt, dtp_dt = SimTemperatures.loc_temp_dyn(ce_sam_t, cp_sam_t, gep_sam, te, tp, pulse_t, mag_en_t,
                                                      el_mask,  el_mag_mask)

        return dte_dt, dtp_dt


class Sim12LL(SimTemperatures):
    @staticmethod
    def temp_dyn(te, tp, ce_sam_t, cp_sam_t, gep_sam, pulse_t, mag_en_t, el_mask, el_mag_mask,
                 kappa_e_dz_pref, kappa_p_dz_pref, tp2_mask, gpp_sam, cp2_sam_t, len_sam_tp2, len_sam):
        # This method combines the update functions defined in SimTemperatures for a setup of a 1+2TM 1 layer,
        # thus no temperature diffusion is computed, but phonon_phonon coupling is.

        tp1 = tp[:len_sam]
        tp2 = tp[len_sam:]

        dte_dt, dtp1_dt = SimTemperatures.loc_temp_dyn(ce_sam_t, cp_sam_t, gep_sam, te, tp1, pulse_t, mag_en_t,
                                                       el_mask,  el_mag_mask)
        dtp1_dt_pp, dtp2_dt = SimTemperatures.phonon_phonon_coupling(gpp_sam, cp_sam_t, cp2_sam_t, tp1[tp2_mask], tp2)
        dtp1_dt[tp2_mask] += dtp1_dt_pp
        dtp_dt = np.concatenate((dtp1_dt, dtp2_dt))

        return dte_dt, dtp_dt
