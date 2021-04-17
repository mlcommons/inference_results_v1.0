class QPSBisect:
    '''A wrapper which will return a cross term for binary search on QPS.

    For implementation purposes, note how this is both
    deterministic and stateless, which is why we reconstruct
    `plausible` each time and re-check through all `past_runs` each
    time

    TODO: generalize to check an arbitrary parameter and evaluate
    an arbitrary comparator/callback to determine next search step'''
    def __init__(self, lb, ub, ss, past_runs):
        self.lower_bound = lb
        self.upper_bound = ub
        self.step_size = ss
        self.plausible = range(lb, ub, ss)
        assert(len(self.plausible) > 2)
        self.past_runs = past_runs
    def _get_new_idx_bounds(self):
        lo = None
        hi = None
        for term, res in self.past_runs:
            val = term['server_target_qps']
            succ = res['result_validity'] == 'VALID'
            if succ:
                lo = val
            else:
                hi = val
        return (self.plausible.index(lo) if lo else 0 ,
                self.plausible.index(hi) if hi else len(self.plausible))
    def _bounds_idx_to_cross(self, lo_idx, hi_idx):
        mid_idx = (hi_idx + lo_idx) // 2
        return {'server_target_qps': self.plausible[mid_idx]}
    def get_next_cross(self):
        if self.past_runs:
            lo_idx, hi_idx = self._get_new_idx_bounds()
            to_ret = self._bounds_idx_to_cross(lo_idx, hi_idx)

            past_crosses = ({'server_target_qps': x[0]['server_target_qps']}
                             for x in self.past_runs)
            if to_ret in past_crosses:
                return None
            else:
                return to_ret
        else:
            return self._bounds_idx_to_cross(0, len(self.plausible))
