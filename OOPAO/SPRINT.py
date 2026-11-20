# -*- coding: utf-8 -*-
"""SPRINT: identification of deformable-mirror / WFS mis-registrations.

The numerical implementation is intentionally kept in this module:
interaction-matrix generation, sensitivity computation, persistence and
iterative estimation share the same model-generation path.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pfits

from .MisRegistration import MisRegistration
from .calibration.CalibrationVault import CalibrationVault
from .calibration.InteractionMatrix import InteractionMatrix
from .tools.tools import createFolder, warning


# Public parameter definitions. Indices are used by the sensitivity matrix.
MIS_REG_FIELDS = (
    "shiftX", "shiftY", "rotationAngle",
    "magnification", "radialScaling", "tangentialScaling",
)
MIS_REG_SHORT_NAMES = ("dX", "dY", "dRot", "dMagn", "dmX", "dmY")
MIS_REG_TITLES = (
    "Shift X", "Shift Y", "Rotation Angle",
    "Magnification", "radialScaling", "tangentialScaling",
)
MIS_REG_UNITS = ("[m]", "[m]", "[deg]", "[%]", "[%]", "[%]")
STROKE = 1e-9


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------

def select_mis_reg_indices(ind_mis_reg=None, n_mis_reg=None):
    """Return the selected indices in MIS_REG_FIELDS."""
    if ind_mis_reg is None:
        count = len(MIS_REG_FIELDS) if n_mis_reg is None else n_mis_reg
        indices = np.arange(count, dtype=int)
    else:
        indices = np.atleast_1d(np.asarray(ind_mis_reg, dtype=int))
        if n_mis_reg is not None and len(indices) > n_mis_reg:
            warning(
                "ind_mis_reg has {} entries but n_mis_reg = {}: only {} are identified."
                .format(len(indices), n_mis_reg,
                        [MIS_REG_FIELDS[i] for i in indices[:n_mis_reg]])
            )
            indices = indices[:n_mis_reg]

    if np.any(indices < 0) or np.any(indices >= len(MIS_REG_FIELDS)):
        raise ValueError("Mis-registration indices must be between 0 and 5.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Mis-registration indices must be unique.")
    return indices


def as_2d(matrix):
    """Return an interaction matrix with shape [n_signal, n_modes]."""
    matrix = np.asarray(matrix)
    return matrix.reshape(matrix.shape[0], -1)


def same_mis_registration(first, second):
    return np.allclose(first.as_array(), second.as_array())


def default_epsilon(pitch):
    """Default finite-difference perturbations."""
    epsilon = MisRegistration()
    epsilon.shiftX = np.round(pitch / 10, 4)
    epsilon.shiftY = np.round(pitch / 10, 4)
    epsilon.rotationAngle = 0.1
    # Magnification also affects the radial/tangential scaling convention.
    epsilon.magnification = 0.01
    epsilon.radialScaling = 0.01
    epsilon.tangentialScaling = 0.01
    return epsilon


def _warn_deprecated_arguments(function_name, kwargs):
    unused = [key for key, value in kwargs.items() if value is not None]
    if unused:
        warning(
            "{}: argument(s) {} are obsolete and ignored. Mis-registrations "
            "are applied with DeformableMirror.apply_mis_registration."
            .format(function_name, ", ".join(unused))
        )


# ---------------------------------------------------------------------------
# Shared interaction-matrix model
# ---------------------------------------------------------------------------

def interaction_matrix(obj, basis, mis_registration):
    """Generate an interaction matrix for one DM mis-registration."""
    dm = obj.dm.apply_mis_registration(mis_registration)
    return InteractionMatrix(
        obj.ngs, obj.tel, dm, obj.wfs, np.squeeze(basis.modes), STROKE, nMeasurements=1, invert=False, print_time=False,
    )


# ---------------------------------------------------------------------------
# Sensitivity matrix computation and optional FITS cache
# ---------------------------------------------------------------------------

def _sensitivity_cache_location(folder, system, zero_point, basis, dm):
    modes = np.squeeze(basis.modes)
    zonal = (
        modes.shape == (dm.nValidAct, dm.nValidAct)
        and np.array_equal(modes, np.eye(dm.nValidAct))
    )
    path = folder + system + zero_point.misRegName + "/" + ("zon/" if zonal else "mod/")
    suffix = "" if zonal else "_" + getattr(basis, "extra", "")
    return path, suffix


def _load_or_compute_calibration(folder, suffix, filename, title,
                                 compute, use_files, recompute, loaded_files):
    path = os.path.join(folder, "im_" + filename + suffix + ".fits") if use_files else None

    if path and not recompute and os.path.isfile(path):
        with pfits.open(path) as hdus:
            loaded_files.append(path)
            return CalibrationVault(hdus[1].data, invert=False)

    calibration = compute()
    if path:
        header = pfits.Header()
        header["TITLE"] = title
        pfits.HDUList([
            pfits.PrimaryHDU(header=header),
            pfits.ImageHDU(calibration.D),
        ]).writeto(path, overwrite=True)
    return calibration


def compute_sensitivity(
    obj,
    basis,
    zero_point,
    epsilon,
    indices=None,
    n_mis_reg=3,
    name_folder=None,
    name_system="",
    save=True,
    recompute=False,
):
    """Compute centered finite-difference sensitivity matrices.

    Returns (meta_sensitivity, calibration_at_zero_point).
    """
    selected = select_mis_reg_indices(indices, n_mis_reg)

    use_files = bool(save and name_folder is not None)
    folder, suffix = None, ""
    if use_files:
        folder, suffix = _sensitivity_cache_location(
            name_folder, name_system, zero_point, basis, obj.dm
        )
        createFolder(folder)

    loaded_files = []

    def cached(filename, title, compute):
        return _load_or_compute_calibration(
            folder, suffix, filename, title, compute,
            use_files, recompute, loaded_files,
        )

    def compute_zero():
        if same_mis_registration(obj.dm.misReg, zero_point):
            return InteractionMatrix(
                obj.ngs, obj.tel, obj.dm, obj.wfs, np.squeeze(basis.modes),
                STROKE, nMeasurements=1,
                invert=False, print_time=False,
            )
        return interaction_matrix(obj, basis, zero_point)

    calib_0 = cached("0", "INTERACTION MATRIX - INITIAL POINT", compute_zero)

    columns = []
    for index in selected:
        field = MIS_REG_FIELDS[index]
        delta = getattr(epsilon, field)
        if delta == 0:
            raise ValueError("Finite-difference epsilon for {} cannot be zero.".format(field))

        perturbed_calibrations = []
        for sign, suffix_sign, label in (
            (1, "p", "POSITIVE"), (-1, "m", "NEGATIVE")
        ):
            perturbed = MisRegistration(zero_point)
            setattr(perturbed, field, getattr(perturbed, field) + sign * delta)
            filename = "{}_{}_{}".format(
                MIS_REG_SHORT_NAMES[index], suffix_sign, np.abs(delta)
            )
            perturbed_calibrations.append(
                cached(
                    filename,
                    "INTERACTION MATRIX - {} MIS-REGISTRATION".format(label),
                    lambda value=perturbed: interaction_matrix(obj, basis, value),
                )
            )

        positive, negative = perturbed_calibrations
        derivative = (
            as_2d(positive.D) - as_2d(negative.D)
        ) / (2 * delta)
        columns.append(derivative.reshape(-1))

    if loaded_files:
        warning(
            "Existing sensitivity matrices loaded from\n{}\n"
            "Make sure that the loaded data correspond to your AO system!"
            .format(os.path.dirname(loaded_files[0]))
        )

    return CalibrationVault(np.stack(columns, axis=1)), calib_0


# Backward-compatible public function name and argument convention.
# Legacy arguments such as `fast` and `atm` are swallowed by
# **deprecated_kwargs and ignored with a warning.
def computeMetaSensitivityMatrix(
    tel, ngs, dm_0, wfs, basis, misRegistrationZeroPoint,
    epsilonMisRegistration, nameFolder=None, nameSystem="",
    save_sensitivity_matrices=True, recompute_sensitivity=False,
    n_mis_reg=3, ind_mis_reg=None,
    **deprecated_kwargs
):
    _warn_deprecated_arguments("computeMetaSensitivityMatrix", deprecated_kwargs)

    class _System:
        pass

    obj = _System()
    obj.tel, obj.ngs, obj.dm, obj.wfs = tel, ngs, dm_0, wfs
    return compute_sensitivity(
        obj, basis, misRegistrationZeroPoint, epsilonMisRegistration,
        indices=ind_mis_reg, n_mis_reg=n_mis_reg,
        name_folder=nameFolder, name_system=nameSystem,
        save=save_sensitivity_matrices,
        recompute=recompute_sensitivity,
    )


# ---------------------------------------------------------------------------
# Iterative estimation
# ---------------------------------------------------------------------------

def compute_scaling_factor(model_matrix, input_matrix, precision):
    """Estimate one least-squares optical gain per mode."""
    denominator = np.sum(model_matrix * model_matrix, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scaling = np.round(
            np.sum(model_matrix * input_matrix, axis=0) / denominator,
            precision,
        )
    invalid = ~np.isfinite(scaling) | (scaling == 0)
    if invalid.any():
        warning(
            "Invalid scaling factor for mode(s) {}; set to 1."
            .format(np.where(invalid)[0])
        )
        scaling[invalid] = 1
    return scaling


def _validity_tolerances(pitch, diameter, tolerance):
    shift = pitch * tolerance
    rotation = np.rad2deg(np.arctan(shift / (diameter / 2)))
    return np.array([shift, shift, rotation, 0.05, 0.05, 0.05])


class _ConvergencePlot:
    """Optional live plot of parameter values during estimation."""

    def __init__(self, indices, reference):
        self.reference = reference
        count = len(indices)
        from OOPAO.tools.displayTools import cl_plot

        self._cl_plot = cl_plot
        self.plot_obj = cl_plot(
            list_fig=[[[0, value], [0, value]] for value in reference],
            type_fig=["plot"] * count,
            list_title=[MIS_REG_TITLES[i] for i in indices],
            list_legend=[None] * count,
            list_label=[["Iteration Number", MIS_REG_UNITS[i]] for i in indices],
            list_lim=[None] * count,
            n_subplot=[count, 1],
            list_display_axis=[True] * count,
            list_ratio=[[0.95, 0.1], [1] * count],
            s=20,
        )

    def update(self, history):
        values = np.asarray(history)
        self.plot_obj.list_lim = [
            [-3 * value, 3 * value] for value in self.reference
        ]
        self._cl_plot(
            list_fig=[
                [np.arange(len(values)), values[:, index]]
                for index in range(values.shape[1])
            ],
            plt_obj=self.plot_obj,
        )
        plt.pause(0.01)
        return self.plot_obj.keep_going is not False


def estimate_mis_registration(
    obj,
    basis,
    calib_in,
    zero_point,
    epsilon,
    sensitivity=None,
    calib_0=None,
    n_iteration=3,
    precision=3,
    gain=1,
    tolerance=1 / 50,
    indices=None,
    previous_estimate=None,
    display=True,
    plot=True,
):
    """Run the iterative SPRINT estimate and return diagnostic results."""
    if sensitivity is None:
        sensitivity, calib_0 = compute_sensitivity(
            obj, basis, zero_point, epsilon,
            indices=indices,
            n_mis_reg=(len(indices) if indices is not None else 3),
        )

    count = sensitivity.M.shape[0]
    selected = select_mis_reg_indices(indices, count)
    fields = [MIS_REG_FIELDS[index] for index in selected]

    def values(registration):
        return np.array([getattr(registration, field) for field in fields], dtype=float)

    estimate = MisRegistration(zero_point)
    input_matrix = as_2d(calib_in.D)
    history = (
        [values(estimate)] if previous_estimate is None else list(previous_estimate)
    )
    scaling_history = [np.ones(input_matrix.shape[1])]
    live_plot = (
        _ConvergencePlot(selected, values(zero_point)) if plot else None
    )

    tel = obj.tel
    was_paired = tel.isPaired
    tel.isPaired = False
    last_calibration = calib_0

    try:
        for iteration in range(n_iteration):
            if iteration == 0 and calib_0 is not None:
                last_calibration = calib_0
            else:
                last_calibration = interaction_matrix(obj, basis, estimate)

            model_matrix = as_2d(last_calibration.D)
            scaling = compute_scaling_factor(model_matrix, input_matrix, precision)
            residual = (input_matrix / scaling - model_matrix).reshape(-1)
            update = np.round(
                gain * (sensitivity.M @ residual), precision
            )

            for field, delta in zip(fields, update):
                setattr(estimate, field, getattr(estimate, field) + delta)

            scaling_history.append(scaling)
            history.append(values(estimate))

            if display:
                estimate.print_()
            if live_plot is not None and not live_plot.update(history):
                break
    finally:
        tel.isPaired = was_paired

    for field in fields:
        setattr(estimate, field, np.round(getattr(estimate, field), precision))

    if n_iteration == 1 or len(history) < 2:
        valid = True
    else:
        difference = np.abs(history[-1] - history[-2])
        difference[np.isnan(difference)] = np.inf
        valid = bool(np.all(
            difference <= _validity_tolerances(
                obj.dm.pitch, tel.D, tolerance
            )[selected]
        ))

    if not valid:
        warning(
            "The mis-registration estimation did not converge: "
            "returning the zero point."
        )
        estimate = MisRegistration(zero_point)

    return {
        "mis_registration": estimate,
        "scaling_factors": scaling_history,
        "history": history,
        "validity": valid,
        "calibration": last_calibration,
    }


# Backward-compatible function name and legacy argument convention.
# Legacy arguments such as `fast`, `atm` and `wfs_mis_registrated` are
# swallowed by **deprecated_kwargs and ignored with a warning.
def estimateMisRegistration(
    tel, ngs, dm_0, wfs, basis, calib_in,
    misRegistrationZeroPoint, epsilonMisRegistration,
    calib_0=None, sensitivity_matrices=None,
    nameFolder=None, nameSystem="", nIteration=3,
    precision=3, gainEstimation=1, tolerance=1 / 50,
    ind_mis_reg=None, previous_estimate=None,
    return_all=False, display=True, plot=True,
    **deprecated_kwargs
):
    _warn_deprecated_arguments("estimateMisRegistration", deprecated_kwargs)

    class _System:
        pass

    obj = _System()
    obj.tel, obj.ngs, obj.dm, obj.wfs = tel, ngs, dm_0, wfs

    result = estimate_mis_registration(
        obj, basis, calib_in, misRegistrationZeroPoint,
        epsilonMisRegistration, sensitivity=sensitivity_matrices,
        calib_0=calib_0, n_iteration=nIteration, precision=precision,
        gain=gainEstimation, tolerance=tolerance,
        indices=ind_mis_reg, previous_estimate=previous_estimate,
        display=display, plot=plot,
    )

    if return_all:
        return (
            result["mis_registration"],
            result["scaling_factors"],
            result["history"],
            result["validity"],
            result["calibration"],
        )
    return result["mis_registration"]


# ---------------------------------------------------------------------------
# Public SPRINT interface
# ---------------------------------------------------------------------------

class SPRINT:
    """System Parameters Recurrent INvasive Tracking.

    The class orchestrates sensitivity calibration and iterative estimation.
    The numerical implementation lives in this module and uses one shared
    interaction-matrix model.
    """

    def __init__(
        self, obj, basis, nameFolder=None, nameSystem=None,
        mis_registration_zero_point=None, n_mis_reg=3,
        recompute_sensitivity=False, dm_input=None, ind_mis_reg=None,
        epsilon_mis_registration=None,
    ):
        print("Setting up SPRINT..")
        if dm_input is not None:
            warning(
                "SPRINT: dm_input is no longer needed and is ignored; "
                "mis-registrations are applied by obj.dm."
            )

        self.basis = basis
        self.basis.modes = np.squeeze(self.basis.modes)
        self.ind_mis_reg = select_mis_reg_indices(ind_mis_reg, n_mis_reg)
        self.n_mis_reg = len(self.ind_mis_reg)
        self.mis_reg_fields = [
            MIS_REG_FIELDS[index] for index in self.ind_mis_reg
        ]

        self.recompute_sensitivity = recompute_sensitivity
        self.name_system = "" if nameSystem is None else nameSystem
        self.nameFolder_sensitivity_matrice = (
            self._default_folder(obj) if nameFolder is None else nameFolder
        )

        if mis_registration_zero_point is None:
            warning(
                "No input mis_registration_zero_point. Using the DM current "
                "mis-registration as the zero-point:"
            )
            obj.dm.misReg.print_()
            mis_registration_zero_point = obj.dm.misReg

        self.mis_registration_zero_point = MisRegistration(
            mis_registration_zero_point
        )
        self.epsilonMisRegistration = (
            default_epsilon(obj.dm.pitch)
            if epsilon_mis_registration is None
            else epsilon_mis_registration
        )

        self.metaMatrix, self.calib_0 = self._compute_sensitivity(
            obj, self.mis_registration_zero_point, save=True
        )
        self._initial_state = (
            MisRegistration(self.mis_registration_zero_point),
            self.metaMatrix,
            self.calib_0,
        )
        print("Done!")

    @staticmethod
    def default_epsilon(pitch):
        return default_epsilon(pitch)

    @staticmethod
    def _default_folder(obj):
        parameters = getattr(obj, "param", None)
        try:
            return (
                parameters["pathInput"] + "/" +
                parameters["name"] + "/s_mat/"
            )
        except (TypeError, KeyError):
            warning(
                "No nameFolder provided (and no obj.param): "
                "the sensitivity matrices will not be saved."
            )
            return None

    def _compute_sensitivity(self, obj, zero_point, save):
        return compute_sensitivity(
            obj, self.basis, zero_point, self.epsilonMisRegistration,
            indices=self.ind_mis_reg, n_mis_reg=self.n_mis_reg,
            name_folder=self.nameFolder_sensitivity_matrice,
            name_system=self.name_system, save=save,
            recompute=self.recompute_sensitivity,
        )

    def estimate(
        self, obj, on_sky_slopes, n_iteration=3,
        n_update_zero_point=0, precision=3, gain_estimation=1,
        dm_input=None, tolerance=1 / 50, display=True, plot=True,
    ):
        """Estimate mis-registration parameters from WFS signals."""
        if dm_input is not None:
            warning("SPRINT.estimate: dm_input is obsolete and ignored.")

        measured = CalibrationVault(on_sky_slopes, invert=False)
        zero_point, self.metaMatrix, self.calib_0 = self._initial_state
        self.mis_registration_zero_point = MisRegistration(zero_point)
        print("SPRINT was setup around the following working point:")
        self.mis_registration_zero_point.print_()
        self.mis_registration_buffer = None

        for update_index in range(n_update_zero_point + 1):
            if update_index:
                print("----------------------------------")
                print("Mis-Registrations Intermediate Value:")
                self.mis_registration_out.print_()
                print("----------------------------------")
                print("Updating the set of sensitivity matrices...", end=" ")
                self.mis_registration_zero_point = MisRegistration(
                    self.mis_registration_out
                )
                self.metaMatrix, self.calib_0 = self._compute_sensitivity(
                    obj, self.mis_registration_zero_point, save=False
                )
                print("Done!")

            result = estimate_mis_registration(
                obj, self.basis, measured,
                self.mis_registration_zero_point,
                self.epsilonMisRegistration,
                sensitivity=self.metaMatrix,
                calib_0=self.calib_0,
                n_iteration=n_iteration,
                precision=precision,
                gain=gain_estimation,
                tolerance=tolerance,
                indices=self.ind_mis_reg,
                previous_estimate=self.mis_registration_buffer,
                display=display,
                plot=plot,
            )

            self.mis_registration_out = result["mis_registration"]
            self.scaling_factor = result["scaling_factors"]
            self.mis_registration_buffer = result["history"]
            self.validity_flag = result["validity"]
            self.calib_last = result["calibration"]

        print("----------------------------------")
        print("Final Mis-Registrations identified:")
        self.mis_registration_out.print_()
        print("Mis-registration Validity Flag: " + str(self.validity_flag))
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        return self.mis_registration_out