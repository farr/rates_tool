import kde
import numpy as np

class RatePosterior(object):
    """Posterior object for counting with confusion between foreground
    and background.
    """

    def __init__(self, bgs, fg, coinc):
        """Initialise the posterior.

        :param bgs: List of ``(N, Ndim)`` arrays, each containing
          background triggers in the ``Ndim``-dimensional classification
          space for a detector.

        :param fg: Array of shape ``(N, Ndet*Ndim)`` giving ``N``
          samples from the combined foreground classification space on
          ``Ndet`` detectors in ``Ndim`` dimensions.

        :param coinc: Array of shape ``(Ncoinc, Ndet*Ndim)`` giving the
          coincident events in all detectors.
        """

        bg_kdes = [kde.KDE(bg) for bg in bgs]
        fg_kde = kde.KDE(fg)

        log_fg_ratios = []
        for coi in coinc:
            log_bg = np.sum([bkde(c) for bkde, c in zip(bg_kdes, coi)])
            log_fg = fg_kde(coi)

            log_fg_ratios.append(log_fg - log_bg)
        log_fg_ratios = np.array(log_fg_ratios)

        self._coinc = coinc
        self._log_fg = log_fg_ratios.squeeze()

    @property
    def coinc(self):
        return self._coinc
    @property
    def log_fg(self):
        """The log of the foreground to background likelihood ratio.

        """
        return self._log_fg

    @property
    def dtype(self):
        """We use the log of the foreground and background rates, under
        the names ``log_Rf`` and ``log_Rb`` for parameters.

        """
        return np.dtype([('log_Rf', np.float),
                         ('log_Rb', np.float)])
    def to_params(self, x):
        return np.atleast_1d(x).view(self.dtype).squeeze()

    def log_prior(self, p):
        r"""The prior for each rate has a density in rate space of

        .. math::

          p(R) \propto \frac{1}{\sqrt{R}}

        """
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
        """Return the log-posterior for the confusion model.

        """
        
        return self.log_prior(p) + self.log_likelihood(p)

    def log_pbacks(self, p):
        """Returns the log of the probability that each coinc is a
        background event.
        """
        p = self.to_params(p)

        # pb = Rb*rhob/(Rf*rhof + Rb*rhob)
        # pb = Rb/(Rf*rhoratio + Rb)

        # log(pb) = log_Rb - log(Rf*rhoratio + Rb)

        log_Rf = p['log_Rf']
        log_Rb = p['log_Rb']

        return log_Rb - np.logaddexp(log_Rf + self.log_fg, log_Rb)

    def log_pfores(self, p):
        """Returns the log of the probability that each coinc is a
        foreground event.
        """
        p = self.to_params(p)

        log_Rf = p['log_Rf']
        log_Rb = p['log_Rb']

        return log_Rf + self.log_fg - np.logaddexp(log_Rf + self.log_fg, log_Rb)
        
