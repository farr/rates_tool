import numpy as np

class KDE(object):
    def __init__(self, pts):
        pts = np.atleast_1d(pts)

        if len(pts.shape) == 1:
            pts = pts.reshape((-1, 1))
        self._pts = np.random.permutation(pts)
        self._cov = np.cov(pts, rowvar=0) / self.N**(2/(self.ndim+4))

    @property
    def pts(self):
        return self._pts

    @property
    def N(self):
        return self.pts.shape[0]

    @property
    def ndim(self):
        return self.pts.shape[1]

    @property
    def cov(self):
        return self._cov

    def __call__(self, xs):
        xs = np.atleast_1d(xs)

        if self.ndim > 1:
            if len(xs.shape) == 1:
                xs = xs.reshape((1, -1))
                
            log_pdfs = []

            s, logdet = np.linalg.slogdet(self.cov)

            for x in xs:
                dx = self.pts - x

                chi2 = np.sum(dx*np.linalg.solve(self.cov, dx.T).T, axis=1)

                log_normals = -0.5*self.ndim*np.log(2*np.pi) - 0.5*logdet - 0.5*chi2

                log_mean = np.logaddexp.reduce(log_normals) - np.log(self.N)
                log_pdfs.append(log_mean)

            return np.array(log_pdfs)
        else:
            xs = xs.reshape((-1, 1))
            log_pdfs = []
            for x in xs:
                dx = self.pts - x

                chi2 = np.sum(dx*dx/self.cov, axis=1)

                log_normals = -0.5*np.log(2*np.pi) - 0.5*np.log(self.cov) - 0.5*chi2

                log_mean = np.logaddexp.reduce(log_normals) - np.log(self.N)
                log_pdfs.append(log_mean)
            return np.array(log_pdfs)
