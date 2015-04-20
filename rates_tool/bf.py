import kde
import numpy as np

def fore_back_log_bayes_factor(log_Rfs, Rmax=100):
    """Returns the bayes factor for the foreground+background model
    versus the background-only model, assuming the given foreground
    count upper limit.

    """

    # Prior is flat in sqrt(R):
    sqrt_Rfs = np.exp(0.5*log_Rfs)
    fore_kde = kde.KDE(sqrt_Rfs)
    fore_kde2 = kde.KDE(-sqrt_Rfs)

    # Use image points to implement hard boundary at R = 0
    log_p0 = np.logaddexp(fore_kde(0), fore_kde2(0))

    # Savage-Dickie ratio: BF = prior(R = 0) / posterior(R = 0)
    return -np.log(np.sqrt(Rmax)) - log_p0
