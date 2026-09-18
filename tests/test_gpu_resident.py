"""Numerical checks for the opt-in geometric, unmodulated GPU optical path."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


_MODEL = r'''
import contextlib
import io
import numpy as np
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Pyramid import Pyramid
from OOPAO.Detector import Detector
from OOPAO.tools.tools import get_array_module

with contextlib.redirect_stdout(io.StringIO()):
    src = Source('V', 8, display_properties=False)
    tel = Telescope(resolution=32, diameter=1, samplingTime=0.001)
    src * tel
    atm = Atmosphere(telescope=tel, r0=0.2, L0=25,
                     windSpeed=[5, 8], fractionalR0=[0.7, 0.3],
                     windDirection=[0, 45], altitude=[0, 5000])
    atm.initializeAtmosphere(tel)
    atm.generateNewPhaseScreen(seed=10)
    tel + atm
    modes = np.zeros((32 * 32, 2))
    modes[:, 0] = np.linspace(-1, 1, 32 * 32) * 1e-8
    modes[:, 1] = -modes[:, 0]
    dm = DeformableMirror(telescope=tel, nSubap=2,
                          coordinates=np.array([[0., 0.], [0.1, 0.1]]),
                          pitch=0.5, modes=modes, print_dm_properties=False)
    dm.coefs = np.array([0.2, -0.1])
    wfs = Pyramid(nSubap=8, telescope=tel, modulation=0, lightRatio=0,
                  postProcessing='fullFrame_sum_flux', n_pix_separation=4,
                  n_pix_edge=2)
    cam = Detector(nRes=64)
    frames, opds = [], []
    for _ in range(3):
        atm.update()
        src ** atm * tel * dm * wfs
        frames.append(wfs.cam.frame.copy())
        opd_backend = get_array_module(src.OPD)
        opds.append(opd_backend.asnumpy(src.OPD) if opd_backend is not np
                    else src.OPD.copy())
        src ** atm * tel * dm
        tel * cam

np.savez(sys.argv[1], frames=frames, opds=opds, science=cam.frame,
         source_backend=get_array_module(src.OPD).__name__,
         dm_backend=get_array_module(dm.OPD).__name__,
         pyramid_backend=get_array_module(wfs.raw_data).__name__,
         camera_backend=get_array_module(wfs.cam.frame).__name__)
'''


class GpuResidentTest(unittest.TestCase):
    def test_auto_falls_back_when_cupy_has_no_device(self):
        from OOPAO.runtime import array_backend

        class FakeCudaError(Exception):
            pass

        runtime = types.SimpleNamespace(CUDARuntimeError=FakeCudaError,
                                        getDeviceCount=lambda: 0)
        fake_cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(runtime=runtime))
        with mock.patch.dict(sys.modules, {'cupy': fake_cupy}):
            with mock.patch.dict(os.environ, {'OOPAO_BACKEND': 'auto'}):
                self.assertEqual(array_backend(), (np, False))
            with mock.patch.dict(os.environ, {'OOPAO_BACKEND': 'cuda'}):
                with self.assertRaisesRegex(RuntimeError, 'no device is visible'):
                    array_backend()

    def run_model(self, backend, precision, path, resident=False):
        env = os.environ.copy()
        env.update(OOPAO_BACKEND=backend, OOPAO_PRECISION=str(precision),
                   OOPAO_GPU_RESIDENT='1' if resident else '0')
        script = 'import sys\n' + _MODEL
        subprocess.run([sys.executable, '-c', script, str(path)], env=env,
                       cwd=Path(__file__).resolve().parents[1], check=True,
                       capture_output=True, text=True)
        return np.load(path)

    def test_cpu_backend_with_residency_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_model('cpu', 64, Path(directory) / 'cpu.npz',
                                    resident=True)
            self.assertEqual(result['source_backend'].item(), 'numpy')
            self.assertEqual(result['dm_backend'].item(), 'numpy')
            self.assertEqual(result['pyramid_backend'].item(), 'numpy')
            self.assertEqual(result['camera_backend'].item(), 'numpy')

    def test_cuda_matches_cpu(self):
        try:
            import cupy as cp
            usable = cp.cuda.runtime.getDeviceCount() > 0
        except (ImportError, RuntimeError):
            usable = False
        if not usable:
            self.skipTest('No usable CUDA device')
        with tempfile.TemporaryDirectory() as directory:
            for precision in (32, 64):
                cpu = self.run_model('cpu', precision,
                                     Path(directory) / f'cpu_{precision}.npz')
                gpu = self.run_model('cuda', precision,
                                     Path(directory) / f'gpu_{precision}.npz',
                                     resident=True)
                self.assertEqual(gpu['source_backend'].item(), 'cupy')
                self.assertEqual(gpu['dm_backend'].item(), 'cupy')
                self.assertEqual(gpu['pyramid_backend'].item(), 'cupy')
                self.assertEqual(gpu['camera_backend'].item(), 'numpy')
                for key in ('frames', 'opds', 'science'):
                    if precision == 64:
                        np.testing.assert_allclose(cpu[key], gpu[key],
                                                   rtol=1e-10, atol=1e-12)
                    else:
                        np.testing.assert_allclose(cpu[key], gpu[key],
                                                   rtol=2e-4, atol=2e-4)


if __name__ == '__main__':
    unittest.main()
