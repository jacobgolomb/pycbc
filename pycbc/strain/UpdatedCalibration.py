#Authors Jacob Golomb, Richard Udall, Colm Talbot, Derek Davis

import pycbc
import numpy as np
import tables
from abc import (ABCMeta, abstractmethod)
from six import add_metaclass
from scipy.interpolate import UnivariateSpline, interp1d
from pycbc.filter import get_cutoff_indices
from pycbc.types import FrequencySeries

#Jacob says: Skip to CubicSpline

@add_metaclass(ABCMeta)
class Recalibrate(object):
    """ Base class for modifying calibration """
    name = None

    def __init__(self, ifo_name):
        self.ifo_name = ifo_name
        self.params = dict()
        
    @abstractmethod
    def apply_calibration(self, strain):
        """Apply calibration model

        This method should be overwritten by subclasses

        Parameters
        ----------
        strain : FrequencySeries
            The strain to be recalibrated.

        Return
        ------
        strain_adjusted : FrequencySeries
            The recalibrated strain.
        """
        return

    def map_to_adjust(self, strain, prefix='recalib_', **params):
        """Map an input dictionary of sampling parameters to the
        adjust_strain function by filtering the dictionary for the
        calibration parameters, then calling adjust_strain.

        Parameters
        ----------
        strain : FrequencySeries
            The strain to be recalibrated.
        prefix: str
            Prefix for calibration parameter names
        params : dict
            Dictionary of sampling parameters which includes
            calibration parameters.
        Return
        ------
        strain_adjusted : FrequencySeries
            The recalibrated strain.
        """

        self.params.update({
            key[len(prefix):]: params[key]
            for key in params if prefix in key and self.ifo_name in key})

        strain_adjusted = self.apply_calibration(strain)

        return strain_adjusted

    @classmethod
    def from_config(cls, cp, ifo, section):
        """Read a config file to get calibration options and transfer
        functions which will be used to intialize the model.

        Parameters
        ----------
        cp : WorkflowConfigParser
            An open config file.
        ifo : string
            The detector (H1, L1) for which the calibration model will
            be loaded.
        section : string
            The section name in the config file from which to retrieve
            the calibration options.
        Return
        ------
        instance
            An instance of the class.
        """
        all_params = dict(cp.items(section))
        params = {key[len(ifo)+1:]: all_params[key]
                  for key in all_params if ifo.lower() in key}
        params = {key: params[key] for key in params}
        params.pop('model')
        params['ifo_name'] = ifo.lower()

        return cls(**params)


class CubicSpline(Recalibrate):
    """Cubic spline recalibration

    see https://dcc.ligo.org/LIGO-T1400682/public

    Parameters
    ----------

    spline_points: array of frequencies where calibration curves are sampled (only used if supplying dict of parameters)
    params: dictionary of parameters 'amplitude_{ifo}_{#}' and 'phase_{ifo}_{#}, corresponding to spline points (only used if supplying dict of parameters)
        important note: this assumes that amplitude params are percent errors (i.e. 'amplitude_H1_0'=1.04 wold be a +4% miscalibration error
        in Hanford at the 0th index frequency spline point)
    calibration_file: HDF5 file from calibration group used to generate spline in amplitude and phase space (only used if generating from file)
    spline_index: index of the specific spline from the HDF5 file to use. Enumerated for reproducability (only used if generating spline from file)
    number of spline points
    """
    name = 'cubic_spline'

    def __init__(self, spline_points=None, params=None,
                 ifo_name = None, calibration_file = None, spline_index=0):
        Recalibrate.__init__(self, ifo_name=ifo_name)
        """Initializes the object
        Must supply either a calibration file or
        dictionary of parameters to initialize the splines.
        """
        
        self.spline_index = spline_index
        self.spline_points = spline_points
        self.calibration_file = calibration_file
        
        self.set_spline(params)
        
        
    def apply_calibration(self, strain):
        """Apply calibration model

        This applies cubic spline calibration to the strain.

        Parameters
        ----------
        strain : FrequencySeries
            The strain to be recalibrated.

        Return
        ------
        strain_adjusted : FrequencySeries
            The recalibrated strain.
        """
        amplitude_relative = self.amplitude_spline(strain.sample_frequencies.numpy())
        delta_phase = self.phase_spline(strain.sample_frequencies.numpy())

        strain_adjusted = strain * (amplitude_relative)\
            * (2.0 + 1j * delta_phase) / (2.0 - 1j * delta_phase)

        return strain_adjusted
    
    def set_spline(self, params=None):
        """Creates the cubic spline interpolations by either inputing a new
        dict of parameters, or using the calibration group HDF5 file.
        If this is called and the object does not have a calibration file or 
        parameter dictionary to use, it will throw an error.
        
        Parameters
        ----------

        params : dictionary, optional
            Parameters 'amplitude_{ifo}_{#}' and 'phase_{ifo}_{#}, corresponding to spline points (only used if supplying dict of parameters)
            important note: this assumes that amplitude params are percent errors (i.e. 'amplitude_H1_0'=1.04 wold be a +4% miscalibration error
            in Hanford at the 0th index frequency spline point)
        """
        
        if params:
            self.params.update(**params)
            self.calibration_amplitude = [self.params['amplitude_{}_{}'.format(self.ifo_name, ii)] for ii in range(self.n_points)]
            self.calibration_phase = [self.params['phase_{}_{}'.format(self.ifo_name, ii)] for ii in range(self.n_points)]
            self.calibration_frequencies = self.spline_points
        
        elif self.calibration_file:
            self.get_spline_params_from_file()
        
        self.amplitude_spline = interp1d(self.calibration_frequencies, self.calibration_amplitude, 'cubic', fill_value='extrapolate')
        self.phase_spline = interp1d(self.calibration_frequencies, self.calibration_phase, 'cubic', fill_value='extrapolate')

    def get_spline_params_from_file(self):        
        """Samples spline points (frequency, amplitude) and (frequency, phase) 
        from an HDF5 file with the format used by the calibration group.
        Stores these as arrays internally.
        """
        calibration_file = tables.open_file(self.calibration_file, 'r')
        calibration_amplitude = \
            calibration_file.root.deltaR.draws_amp_rel[self.spline_index:self.spline_index+1]
        calibration_phase = \
            calibration_file.root.deltaR.draws_phase[self.spline_index:self.spline_index+1]

        calibration_frequencies = calibration_file.root.deltaR.freq[:]

        calibration_file.close()

        if len(calibration_amplitude.dtype) != 0:  # handling if this is a calibration group hdf5 file
            calibration_amplitude = calibration_amplitude.view(np.float64).reshape(calibration_amplitude.shape + (-1,))
            calibration_phase = calibration_phase.view(np.float64).reshape(calibration_phase.shape + (-1,))
            calibration_frequencies = calibration_frequencies.view(np.float64)  
            
        self.calibration_amplitude = calibration_amplitude
        self.calibration_phase = calibration_phase
        self.calibration_frequencies = calibration_frequencies
        
    def calibration_optimal_snr(self, strain, psd = None, low_frequency_cutoff=None, high_frequency_cutoff=None):
        """Calculates the optimal SNR between an input waveform and
        a miscalibrated waveform. If no PSD is given, this is just
        proportional to a(non-weighted) inner product between 
        the waveform and miscalibrated waveform.
        
        Parameters
        ----------
        
        strain: pycbc FrequencySeries or TimeSeries object
            The input template waveform generated with pycbc.
        psd: FrequencySeries, optional
            The psd used to weight the accumulated power
        low_frequency_cutoff : {None, float}, optional
            The frequency to begin considering waveform power.
        high_frequency_cutoff : {None, float}, optional
            The frequency to stop considering waveform power.
            
        Return
        ------
        snr_cal_opt: float
        """
        if type(strain) == pycbc.types.TimeSeries:
            strain = strain.to_FrequencySeries()
            
        N = (len(strain)-1) * 2
        norm = 4.0 * strain.delta_f
        kmin, kmax = get_cutoff_indices(low_frequency_cutoff,
                                       high_frequency_cutoff, strain.delta_f, N)
        ht = strain[kmin:kmax]
        
        ht_calib = self.apply_calibration(ht)
        
        ht_calib = FrequencySeries(ht_calib.squeeze(), delta_f = ht.delta_f)

        if psd:
            try:
                np.testing.assert_almost_equal(ht.delta_f, psd.delta_f)
            except AssertionError:
                raise ValueError('Waveform does not have same delta_f as psd')

        if psd is None:
            sq = ht.inner(ht_calib)
        else:
            sq = ht.weighted_inner(ht_calib, weight = psd[kmin:kmax])

        return sq.real * norm