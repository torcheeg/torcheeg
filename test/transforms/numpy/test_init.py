import importlib
import unittest


class TestNumpyExports(unittest.TestCase):
    def test_public_exports_resolve_to_their_defining_modules(self):
        package = importlib.import_module('torcheeg.transforms.numpy')
        expected = {'ARRCoefficient': 'torcheeg.transforms.numpy.coefficient',
                    'BandApproximateEntropy': 'torcheeg.transforms.numpy.band_pyeeg',
                    'BandBinPower': 'torcheeg.transforms.numpy.band_pyeeg',
                    'BandDetrendedFluctuationAnalysis': 'torcheeg.transforms.numpy.band_pyeeg',
                    'BandDifferentialEntropy': 'torcheeg.transforms.numpy.band',
                    'BandDifferentialEntropyV1': 'torcheeg.transforms.numpy.band',
                    'BandHiguchiFractalDimension': 'torcheeg.transforms.numpy.band_pyeeg',
                    'BandHjorth': 'torcheeg.transforms.numpy.band_pyeeg',
                    'BandHurst': 'torcheeg.transforms.numpy.band_pyeeg',
                    'BandKurtosis': 'torcheeg.transforms.numpy.band',
                    'BandMeanAbsoluteDeviation': 'torcheeg.transforms.numpy.band',
                    'BandPetrosianFractalDimension': 'torcheeg.transforms.numpy.band_pyeeg',
                    'BandPowerSpectralDensity': 'torcheeg.transforms.numpy.band',
                    'BandSVDEntropy': 'torcheeg.transforms.numpy.band_pyeeg',
                    'BandSampleEntropy': 'torcheeg.transforms.numpy.band_pyeeg',
                    'BandSignal': 'torcheeg.transforms.numpy.band',
                    'BandSkewness': 'torcheeg.transforms.numpy.band',
                    'BandSpectralEntropy': 'torcheeg.transforms.numpy.band_pyeeg',
                    'BandTransform': 'torcheeg.transforms.numpy.band_pyeeg',
                    'CWTSpectrum': 'torcheeg.transforms.numpy.spectrum',
                    'Concatenate': 'torcheeg.transforms.numpy.concatenate',
                    'CorrelationTransform': 'torcheeg.transforms.numpy.correlation',
                    'DWTDecomposition': 'torcheeg.transforms.numpy.spectrum',
                    'Downsample': 'torcheeg.transforms.numpy.downsample',
                    'EEGTransform': 'torcheeg.transforms.base_transform',
                    'Flatten': 'torcheeg.transforms.numpy.flatten',
                    'HorizontalFlip': 'torcheeg.transforms.numpy.flip',
                    'LazyLoader': 'torcheeg.utils',
                    'MapChunk': 'torcheeg.transforms.numpy.concatenate',
                    'MeanStdNormalize': 'torcheeg.transforms.numpy.normalize',
                    'MinMaxNormalize': 'torcheeg.transforms.numpy.normalize',
                    'OrderElectrode': 'torcheeg.transforms.numpy.pick',
                    'PearsonCorrelation': 'torcheeg.transforms.numpy.correlation',
                    'PhaseLockingValue': 'torcheeg.transforms.numpy.correlation',
                    'PickElectrode': 'torcheeg.transforms.numpy.pick',
                    'RearrangeElectrode': 'torcheeg.transforms.numpy.rearrange',
                    'Reshape': 'torcheeg.transforms.numpy.reshape',
                    'SetSamplingRate': 'torcheeg.transforms.numpy.downsample',
                    'To2d': 'torcheeg.transforms.numpy.to',
                    'ToChunks': 'torcheeg.transforms.numpy.to',
                    'ToGrid': 'torcheeg.transforms.numpy.to',
                    'ToInterpolatedGrid': 'torcheeg.transforms.numpy.to',
                    'bin_power': 'torcheeg.transforms.numpy.band_pyeeg',
                    'butter_bandpass': 'torcheeg.transforms.numpy.band_pyeeg',
                    'embed_seq': 'torcheeg.transforms.numpy.band_pyeeg',
                    'get_swap_pair': 'torcheeg.transforms.numpy.flip',
                    'horizontal_flip': 'torcheeg.transforms.numpy.flip'}
        for name, module in expected.items():
            with self.subTest(name=name):
                direct = getattr(importlib.import_module(module), name)
                self.assertIs(getattr(package, name), direct)


if __name__ == "__main__":
    unittest.main()
