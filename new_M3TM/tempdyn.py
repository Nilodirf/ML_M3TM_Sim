import numpy as np


def e_p_coupling(te, tp, gep_sam, cp_sam, ce_sam):
    dcete_dt=gep_sam(tp-te)
    dcptp_dt=-dcete_dt

    dte_dt = 1/ce_sam*dcete_dt
    dtp_dt = 1/cp_sam*dcptp_dt

    return dte_dt, dtp_dt
