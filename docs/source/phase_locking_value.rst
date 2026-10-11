Phase locking value: definition and mathematical checks
=======================================================

Definition and input contract
-----------------------------

:class:`torcheeg.transforms.PhaseLockingValue` measures consistency of the
instantaneous phase difference between two real signals. For a time window of
:math:`N` samples, let the analytic signals returned by
``scipy.signal.hilbert`` be

.. math::

   z_x[n] = A_x[n]e^{i\phi_x[n]}, \qquad
   z_y[n] = A_y[n]e^{i\phi_y[n]}.

The transform computes the standard, phase-only statistic

.. math::

   \operatorname{PLV}(x,y)
   = \left|\frac{1}{N}\sum_{n=0}^{N-1}
     e^{i(\phi_x[n]-\phi_y[n])}\right|.

Every time sample has equal weight; instantaneous amplitude does not weight
the phase differences. The existing correlation-transform convention applies:
input ``(channels, time)`` produces ``(1, channels, channels)``, and input
``(bands, channels, time)`` produces ``(bands, channels, channels)``.
``apply_to_baseline=True`` applies the same calculation to the baseline.

For the standard definition, see Aydore, Pantazis and Leahy (2013),
`A note on the phase locking value and its properties
<https://doi.org/10.1016/j.neuroimage.2013.02.008>`_. This transform averages
over the time samples of one window, rather than over trials at a fixed time.

The input must be real and finite, with a nonempty time axis. Nonfinite input
or analytic-signal values raise ``ValueError``. A zero analytic amplitude at
any participating sample also raises ``ValueError``: its phase is undefined,
and ``angle(0) == 0`` must not be treated as an observed phase. No fixed
amplitude threshold or epsilon is introduced.

A zero *real* sample is different from zero *analytic* amplitude. For example,
``[1, 0, -1, 0]`` has analytic signal ``[1, 1j, -1, -1j]`` and unit amplitude
throughout, so its real zero crossings are valid.

Mathematical checks
-------------------

For valid analytic signals, the following properties hold, subject to
floating-point tolerance:

* **Range:** :math:`0 \leq \operatorname{PLV} \leq 1`, by the triangle
  inequality for the mean of unit phasors. Clipping to this interval only
  corrects rounding error.
* **Symmetry:** exchanging the signals conjugates the mean phasor, leaving
  its magnitude unchanged.
* **Diagonal:** self-comparison has zero phase difference and PLV equal to one.
* **Constant phase difference:** any fixed phase offset has PLV equal to one;
  PLV does not measure whether the offset itself is zero.
* **Constant scaling:** multiplying either real signal by any nonzero real
  constant leaves PLV unchanged. Positive scaling preserves phase; negative
  scaling adds a constant phase offset.

Why the legacy correlation magnitude is different
-------------------------------------------------

``PhaseLockingCorrelation`` retains its historical calculation: the angle of
the normalized analytic-signal inner product

.. math::

   C = \frac{\sum_n A_x[n]A_y[n]e^{i(\phi_x[n]-\phi_y[n])}}
   {\sqrt{\sum_n A_x[n]^2\sum_n A_y[n]^2}}.

Thus its result is :math:`\arg C`, an amplitude-weighted resultant direction,
not an arithmetic mean of phase differences. Replacing the angle with
:math:`|C|` would still not give standard PLV.

For a concrete counterexample, use :math:`N=128`,
:math:`\theta_n=2\pi n/N`, and

.. math::

   x[n] = (1+\tfrac{1}{2}\cos\theta_n)\cos(10\theta_n), \qquad
   y[n] = (1-\tfrac{1}{2}\cos\theta_n)\cos(10\theta_n).

Both envelopes are positive, and the carrier and modulation sidebands lie
strictly between DC and Nyquist. Their discrete analytic signals have the same
phase at every sample, so PLV is one. Nevertheless,

.. math::

   |C| = \frac{1-\tfrac{1}{8}}{1+\tfrac{1}{8}} = \frac{7}{9}.

The separate ``PhaseLockingValue`` class provides the phase-only statistic
without changing the legacy transform's numerical behavior.

Finite windows and preprocessing
--------------------------------

The FFT-based Hilbert transform operates on the supplied finite window.
Boundary discontinuities and window length can affect estimated phase. The
properties above describe the resulting analytic signals; they do not guarantee
that arbitrary EEG windows provide reliable physiological phase estimates.

Bandpass filtering, selection of frequency bands, padding and edge trimming are
preprocessing decisions outside this transform. In particular, time-varying
amplitude modulation need not preserve analytic phase for arbitrary broadband
signals; the counterexample above deliberately separates carrier and modulation
frequencies. Low but nonzero analytic amplitudes remain valid mathematically,
although their phase can be sensitive to noise.
