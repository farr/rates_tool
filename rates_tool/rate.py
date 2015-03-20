import kde
import numpy as np

class RatePosterior(object):
    def __init__(self, bgs, fg, coinc):
        bg_kdes = [kde.KDE(bg) for bg in bgs]
        fg_kde = kde.KDE(fg)

        log_fg_ratios = []
        for coi in coinc:
            log_bg = np.sum([bkde(c) for bkde, c in zip(bg_kdes, coi)])
            log_fg = fg_kde(coi)

            log_fg_ratios.append(log_fg - log_bg)
        log_fg_ratios = np.array(log_fg_ratios)

        self._coinc = coinc
        self._log_fg = log_fg_ratios

    @property
    def coinc(self):
        return self._coinc
    @property
    def log_fg(self):
        return self._log_fg

    @property
    def dtype(self):
        return np.dtype([('log_Rf', np.float),
                         ('log_Rb', np.float)])
    def to_params(self, x):
        return np.atleast_1d(x).view(self.dtype).squeeze()

    def log_prior(self, p):
        p = self.to_params(p)

        log_Rf = p['log_Rf']
        log_Rb = p['log_Rb']

        # p(log(r)) d(log(r)) = p(log(r)) / r = p(r) dr
        # p(log(r)) = r p(r) = sqrt(r) = exp(0.5*log(r))
        #
        # log(p(log(r))) = 0.5*log(r)

        return 0.5*(log_Rf + log_Rb)

    def log_likelihood(self, p):
        p = self.to_params(p)

        log_Rf = p['log_Rf']
        log_Rb = p['log_Rb']
        
        return np.sum(np.logaddexp(log_Rf + self.log_fg, log_Rb)) - np.exp(log_Rf) - np.exp(log_Rb)

    def __call__(self, p):
        return self.log_prior(p) + self.log_likelihood(p)
