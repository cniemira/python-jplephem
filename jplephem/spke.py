"""Evaluate SPK files.

"""
from numpy import array, empty, empty_like, rollaxis

T0 = 2451545.0
S_PER_DAY = 86400.0
def jd(seconds):
    return T0 + seconds / S_PER_DAY

class ChebyshevBaseEvaluator():
    def __init__(self, seg, t1, t2):
        self.seg = seg
        self.component_count = self.Meta.component_count

        self.t1 = t1
        self.t2 = t2

        init, intlen, rsize, n = seg.daf.map_array(seg.end_i - 3, seg.end_i)
        initial_epoch = jd(init)
        self.interval_length = intlen / S_PER_DAY

        index, self.offset = divmod((t1 - initial_epoch) + t2,
                self.interval_length)
        self.index = index.astype(int)

        if (self.index < 0).any() or (self.index > n).any():
            final_epoch = initial_epoch + self.interval_length * n
            raise ValueError('segment only covers dates %.1f through %.1f'
                    % (initial_epoch, final_epoch))

        coefficient_count = int(rsize - 2) // self.component_count

        coefficients = seg.daf.map_array(seg.start_i, seg.end_i - 4)
        coefficients.shape = (int(n), rsize)

        coefficients = coefficients[:,2:]
        coefficients.shape = (int(n), self.component_count, coefficient_count)

        coefficients = rollaxis(coefficients, 1)
        (self.component_count, self.n,
                self.coefficient_count) = coefficients.shape

        omegas = (self.index == n)
        self.index[omegas] -= 1
        self.offset[omegas] += self.interval_length

        self.coefficients = coefficients[:,self.index]

    def generate(self):
        scalar = not getattr(self.t1, 'shape', 0) and not getattr(self.t2, 'shape', 0)

        # Chebyshev polynomial.
        T = empty((self.coefficient_count, len(self.index)))
        T[0] = 1.0
        T[1] = t1 = 2.0 * self.offset / self.interval_length - 1.0
        twot1 = t1 + t1
        for i in range(2, self.coefficient_count):
            T[i] = twot1 * T[i-1] - T[i-2]

        components = (T.T * self.coefficients).sum(axis=2)
        if scalar:
            components = components[:,0]

        yield components

        # Chebyshev differentiation.
        dT = empty_like(T)
        dT[0] = 0.0
        dT[1] = 1.0
        if self.coefficient_count > 2:
            dT[2] = twot1 + twot1
            for i in range(3, self.coefficient_count):
                dT[i] = twot1 * dT[i-1] - dT[i-2] + T[i-1] + T[i-1]
        dT *= 2.0
        dT /= self.interval_length

        rates = (dT.T * self.coefficients).sum(axis=2)
        if scalar:
            rates = rates[:,0]

        yield rates


#@handles(data_type=2)
class ChebyshevPositionOnlyEvaluator(ChebyshevBaseEvaluator):
    class Meta:
        component_count = 3

#@handles(data_type=3)
class ChebyshevPositionAndVeolocityEvaluator(ChebyshevBaseEvaluator):
    class Meta:
        component_count = 6


type_evaluators = {
#       1: ModifiedDifferenceArrayEvaluator,
        2: ChebyshevPositionOnlyEvaluator,
        3: ChebyshevPositionAndVeolocityEvaluator,
#       13: HermiteInterpolationWithUnequalTimeStepsEvaluator,
        }

def spke(segment, tdb, tdb2):
    if segment.data_type not in type_evaluators:
        raise ValueError('Unsupported data type: %d' % (segment.data_type,))
    return type_evaluators[segment.data_type](segment, tdb, tdb2)

