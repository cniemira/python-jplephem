"""Compute positions from a NASA SPICE SPK ephemeris kernel file.

http://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html

"""
from numpy import array, empty, empty_like, rollaxis
from .daf import DAF
from .names import target_names
from .spke import spke

T0 = 2451545.0
S_PER_DAY = 86400.0


def jd(seconds):
    """Convert a number of seconds since J2000 to a Julian Date."""
    return T0 + seconds / S_PER_DAY


class SPK(object):
    """A JPL SPK ephemeris kernel for computing positions and velocities.

    You can load an SPK using either a filename or an already opened
    file object::

        kernel = SPK(file_object)
        kernel = SPK.open('de431.bsp')

    Simply ``print(kernel)`` see which segments are inside.  You can
    loop across all of the segments in the list ``kernel.segments`` or,
    as a convenience, you can select a particular segment by providing a
    center and target integer in square brackets.  So ``kernel[3,399]``
    to select the segment that computes the distance between the
    Earth-Moon barycenter (3) and the Earth itself (399).

    To extract the text comments from the SPK use ``kernel.comments()``.

    """
    def __init__(self, file_object):
        self.daf = DAF(file_object)
        self.segments = [Segment(self.daf, *t) for t in self.daf.summaries()]
        self.pairs = dict(((s.center, s.target), s) for s in self.segments)

    @classmethod
    def open(cls, path):
        """Open the file at `path` and return an SPK instance."""
        return cls(open(path, 'rb'))

    def __str__(self):
        daf = self.daf
        d = lambda b: b.decode('latin-1')
        lines = (str(segment) for segment in self.segments)
        return 'File type {0} and format {1} with {2} segments:\n{3}'.format(
            d(daf.locidw), d(daf.locfmt), len(self.segments), '\n'.join(lines))

    def __getitem__(self, key):
        """Given (center, target) integers, return the last matching segment."""
        return self.pairs[key]

    def comments(self):
        """Return the file comments, as a string."""
        return self.daf.comments()


class Segment(object):
    """A single segment of an SPK file.

    There are several items of information about each segment that are
    loaded from the underlying SPK file, and made available as object
    attributes:

    segment.source - official ephemeris name, like 'DE-0430LE-0430'
    segment.start_second - initial epoch, as seconds from J2000
    segment.end_second - final epoch, as seconds from J2000
    segment.start_jd - start_second, converted to a Julian Date
    segment.end_jd - end_second, converted to a Julian Date
    segment.center - integer center identifier
    segment.target - integer target identifier
    segment.frame - integer frame identifier
    segment.data_type - integer data type identifier
    segment.start_i - index where segment starts
    segment.end_i - index where segment ends

    """
    def __init__(self, daf, source, descriptor):
        self.daf = daf
        self.source = source
        (self.start_second, self.end_second, self.target, self.center,
         self.frame, self.data_type, self.start_i, self.end_i) = descriptor
        self.start_jd = jd(self.start_second)
        self.end_jd = jd(self.end_second)

    def __str__(self):
        return self.describe(verbose=False)

    def describe(self, verbose=True):
        """Return a textual description of the segment."""
        center = titlecase(target_names.get(self.center, 'Unknown center'))
        target = titlecase(target_names.get(self.target, 'Unknown target'))
        text = ('{0.start_jd:.2f}..{0.end_jd:.2f}  {1} ({0.center})'
                ' -> {2} ({0.target})'.format(self, center, target))
        if verbose:
            text += ('\n  frame={0.frame} data_type={0.data_type} source={1}'
                     .format(self, self.source.decode('ascii')))
        return text

    def compute(self, tdb, tdb2=0.0):
        """Compute the component values for the time `tdb` plus `tdb2`."""
        for position in self.generate(tdb, tdb2):
            return position

    def compute_and_differentiate(self, tdb, tdb2=0.0):
        """Compute components and differentials for time `tdb` plus `tdb2`."""
        return tuple(self.generate(tdb, tdb2))

    def generate(self, tdb, tdb2):
        """Generate components and differentials for time `tdb` plus `tdb2`.

        Most uses will simply want to call the `compute()` method or the
        `compute_differentials()` method, for convenience.  But in those
        cases (see Skyfield) where you want to compute a position and
        examine it before deciding whether to proceed with the velocity,
        but without losing all of the work that it took to get to that
        point, this generator lets you get them as two separate steps.

        """
        scalar = not getattr(tdb, 'shape', 0) and not getattr(tdb2, 'shape', 0)
        if scalar:
            tdb = array((tdb,))

        if not hasattr(self, '_data'):
            self._data = spke(self, tdb, tdb2)

        x = self._data.generate()
        return x


def titlecase(name):
    """Title-case target `name` if it looks safe to do so."""
    return name if name.startswith(('1', 'C/', 'DSS-')) else name.title()
