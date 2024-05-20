import numpy as np
from scipy import stats


class KSTest:

    def __init__(self, l_hist_size, l_new_size, l_hist_min_size, alpha=0.05, bonferroni_correction=True):
        """
        bonferroni_correction: if True, then reject null hypothesis if the minimal p_value of all dimensions is smaller
        than alpha/num_dim. Otherwise reject null hypothesis if at least one dimension has a p value smaller than alpha.
        """
        self.l_new_size = l_new_size
        self.l_hist_size = l_hist_size
        self.l_hist_min_size = l_hist_min_size
        self.alpha = alpha  # significance level
        self.bonferroni_correction = bonferroni_correction
        self.l_hist = []
        self.l_new = []

        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

    def _one_d_test(self, dim_id):
        """
        st: the ks distance
        p_value: if p_value <= alpha, drift is detected
        """

        st, p_value = stats.ks_2samp(np.array(self.l_hist)[:, dim_id],
                                     np.array(self.l_new)[:, dim_id])
        return st, p_value

    def detect_drift(self, instances):
        """
        Argument
        instances: array in the shape: <N, n_dim>

        Function: detect drift by comparing the data instances in lists l_hist and l_new. At the beginning, instances
        are accumulated in l_new until reach the maximum size l_new_size. Then for every incoming new instance into
        l_new, the oldest instance in l_new will be moved to l_hist. A detection will be triggered only if l_hist
        contains at least l_hist_min_size instances.  l_new and l_hist are cleared when a drift is detected.
        """
        self.l_new += instances.tolist()
        result = False
        p_value_buf = []

        if len(self.l_new) > self.l_new_size:
            self.l_hist += self.l_new[:len(self.l_new) - self.l_new_size]
            self.l_new = self.l_new[len(self.l_new) - self.l_new_size:]
        else:  # no sufficient data to do the ks-test
            return result, p_value_buf

        if len(self.l_hist) > self.l_hist_size:
            self.l_hist = self.l_hist[len(self.l_hist) - self.l_hist_size:]

        if len(self.l_hist) < self.l_hist_min_size:
            return result, p_value_buf
        else:
            for dim_id in range(len(self.l_new[0])):
                st, p_value = self._one_d_test(dim_id)
                p_value_buf.append(round(p_value, 3))
                if p_value < self.alpha and not self.bonferroni_correction:
                    result = True
            if self.bonferroni_correction and min(p_value_buf) < self.alpha / len(self.l_new[0]):
                result = True

            if result:
                self.l_hist = []
                self.l_new = []
        return result, p_value_buf
