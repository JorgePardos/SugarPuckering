"""Smoke tests for the plotting layer, run headless on the Agg backend."""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.analysis import compute_puckering
from src.plotting import (
    plot_amplitude_histogram,
    plot_conformer_populations,
    plot_conformer_timeline,
    plot_fel_mercator,
    plot_furanose_time_series,
    plot_mercator,
    plot_pseudorotation_wheel,
    plot_stoddart,
    plot_time_series,
)

from conftest import make_ring
from test_furanose import make_furanose


@pytest.fixture
def trajectory_results():
    frames = [make_ring(0.57, theta, 270.0) for theta in np.linspace(5, 175, 25)]
    return compute_puckering(np.array(frames))


@pytest.fixture
def single_frame_results():
    return compute_puckering(np.array([make_ring(0.57, 90.0, 270.0)]))


@pytest.fixture
def fel_grid():
    """A two-basin surface on the full theta/phi domain."""
    theta, phi = np.meshgrid(np.linspace(0, 180, 40), np.linspace(0, 360, 60), indexing="ij")
    energy = (theta - 90.0) ** 2 / 400.0 + (phi - 180.0) ** 2 / 900.0
    return np.column_stack([theta.ravel(), phi.ravel(), energy.ravel()])


def _assert_png(path):
    with open(path, "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"


def test_mercator_writes_a_png(trajectory_results, tmp_path):
    path = plot_mercator(trajectory_results, str(tmp_path / "job"), title="job")
    assert path == str(tmp_path / "job") + "_mercator.png"
    _assert_png(path)


def test_mercator_handles_a_single_frame(single_frame_results, tmp_path):
    """One frame takes the star-marker branch instead of the colorbar branch."""
    _assert_png(plot_mercator(single_frame_results, str(tmp_path / "one"), title="one"))


def test_time_series_writes_a_png(trajectory_results, tmp_path):
    _assert_png(plot_time_series(trajectory_results, str(tmp_path / "job"), title="job"))


def test_fel_writes_a_png(fel_grid, tmp_path):
    _assert_png(plot_fel_mercator(fel_grid, str(tmp_path / "fel"), title="fel"))


def test_plots_do_not_leak_figures(trajectory_results, fel_grid, tmp_path):
    """Every plot must close its figure; the GUI can be run many times per session."""
    plt.close("all")
    before = len(plt.get_fignums())
    plot_mercator(trajectory_results, str(tmp_path / "a"))
    plot_time_series(trajectory_results, str(tmp_path / "a"))
    plot_fel_mercator(fel_grid, str(tmp_path / "a"))
    assert len(plt.get_fignums()) == before


def test_title_defaults_do_not_expose_the_path(trajectory_results, tmp_path):
    """Callers pass a path prefix; it must never end up rendered as the title."""
    from matplotlib import pyplot

    captured = {}
    original = pyplot.subplots

    def spy(*args, **kwargs):
        fig, ax = original(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    pyplot.subplots = spy
    try:
        plot_mercator(trajectory_results, str(tmp_path / "somejob"))
    finally:
        pyplot.subplots = original

    assert str(tmp_path) not in captured["ax"].get_title()


def test_fel_survives_unsampled_bins(fel_grid, tmp_path):
    """
    +inf marks bins the sampling never reached. They must be dropped, not clamped
    to the maximum: clamping turns each one into a spike among sampled neighbours
    and the interpolation renders visible speckle.
    """
    grid = fel_grid.copy()
    grid[::37, 2] = np.inf
    _assert_png(plot_fel_mercator(grid, str(tmp_path / "gaps"), title="gaps"))


def test_fel_survives_nan_bins(fel_grid, tmp_path):
    grid = fel_grid.copy()
    grid[::53, 2] = np.nan
    _assert_png(plot_fel_mercator(grid, str(tmp_path / "nans"), title="nans"))


def test_fel_rejects_a_fully_unusable_surface(fel_grid, tmp_path):
    grid = fel_grid.copy()
    grid[:, 2] = np.nan
    with pytest.raises(ValueError, match="no rows with finite"):
        plot_fel_mercator(grid, str(tmp_path / "empty"))


def test_fel_converts_radian_input(tmp_path):
    """A surface in radians must land on the same 0-360 / 0-180 degree axes."""
    theta, phi = np.meshgrid(np.linspace(0, np.pi, 30),
                             np.linspace(0, 2 * np.pi, 40), indexing="ij")
    energy = theta.ravel() * 2.0
    radians = np.column_stack([theta.ravel(), phi.ravel(), energy])
    _assert_png(plot_fel_mercator(radians, str(tmp_path / "rad"), angle_units="rad"))
    _assert_png(plot_fel_mercator(radians, str(tmp_path / "auto"), angle_units="auto"))


def test_fel_auto_units_do_not_misread_a_narrow_degree_surface(tmp_path):
    """
    The old heuristic looked at theta alone and converted anything under 4, so a
    degree-valued surface spanning a few degrees of theta got scaled by 57.
    Requiring *both* columns to fit inside 2*pi avoids that.
    """
    theta, phi = np.meshgrid(np.linspace(0, 3, 20), np.linspace(0, 360, 40), indexing="ij")
    data = np.column_stack([theta.ravel(), phi.ravel(), theta.ravel()])
    _assert_png(plot_fel_mercator(data, str(tmp_path / "narrow"), angle_units="auto"))


def test_fel_rejects_an_unknown_unit(fel_grid, tmp_path):
    with pytest.raises(ValueError, match="angle_units"):
        plot_fel_mercator(fel_grid, str(tmp_path / "bad"), angle_units="degrees")


@pytest.fixture
def furanose_results():
    return compute_puckering(np.array([make_furanose(0.4, p)
                                       for p in np.linspace(0, 350, 20)]))


def test_pseudorotation_wheel_writes_a_png(furanose_results, tmp_path):
    path = plot_pseudorotation_wheel(furanose_results, str(tmp_path / "fur"), title="fur")
    assert path == str(tmp_path / "fur") + "_wheel.png"
    _assert_png(path)


def test_pseudorotation_wheel_handles_a_single_frame(tmp_path):
    single = compute_puckering(np.array([make_furanose(0.4, 18.0)]))
    _assert_png(plot_pseudorotation_wheel(single, str(tmp_path / "one"), title="one"))


def test_furanose_time_series_writes_a_png(furanose_results, tmp_path):
    _assert_png(plot_furanose_time_series(furanose_results, str(tmp_path / "fur")))


def test_furanose_plots_do_not_leak_figures(furanose_results, tmp_path):
    plt.close("all")
    before = len(plt.get_fignums())
    plot_pseudorotation_wheel(furanose_results, str(tmp_path / "b"))
    plot_furanose_time_series(furanose_results, str(tmp_path / "b"))
    assert len(plt.get_fignums()) == before


def test_conformer_timeline_writes_a_png(trajectory_results, tmp_path):
    path = plot_conformer_timeline(trajectory_results, str(tmp_path / "job"), 6,
                                   title="job")
    assert path == str(tmp_path / "job") + "_timeline.png"
    _assert_png(path)


def test_conformer_timeline_works_for_furanoses(furanose_results, tmp_path):
    _assert_png(plot_conformer_timeline(furanose_results, str(tmp_path / "fur"), 5))


def test_conformer_timeline_orders_the_axis_by_latitude(tmp_path):
    """
    The vertical axis must follow the canonical conformer sequence, so an
    itinerary reads as a path rather than an arbitrary reshuffling.
    """
    from matplotlib import pyplot

    frames = [make_ring(0.57, theta, 90.0) for theta in (170.0, 5.0, 90.0)]
    results = compute_puckering(np.array(frames))

    captured = {}
    original = pyplot.subplots

    def spy(*args, **kwargs):
        fig, ax = original(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    pyplot.subplots = spy
    try:
        plot_conformer_timeline(results, str(tmp_path / "order"), 6)
    finally:
        pyplot.subplots = original

    from src.analysis import conformer_tex

    names = [t.get_text() for t in captured["ax"].get_yticklabels()]
    # Ticks are rendered as LaTeX, so compare against the TeX forms
    assert names == [conformer_tex(n) for n in ("4C1", "5S1", "1C4")]


def test_timeline_does_not_leak_figures(trajectory_results, tmp_path):
    plt.close("all")
    before = len(plt.get_fignums())
    plot_conformer_timeline(trajectory_results, str(tmp_path / "t"), 6)
    assert len(plt.get_fignums()) == before


def test_time_series_now_includes_the_amplitude(trajectory_results, tmp_path):
    """Q is the only parameter that shows a ring flattening; it needs a panel."""
    from matplotlib import pyplot

    captured = {}
    original = pyplot.subplots

    def spy(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured["axes"] = axes
        return fig, axes

    pyplot.subplots = spy
    try:
        plot_time_series(trajectory_results, str(tmp_path / "ts"))
    finally:
        pyplot.subplots = original

    assert len(captured["axes"]) == 3
    assert "Q" in captured["axes"][2].get_ylabel()


def test_path_is_broken_at_the_phi_seam():
    """
    A step from 355 to 5 degrees is a small move on a periodic axis but a line
    right across the map if drawn literally.
    """
    from src.plotting import _split_at_phi_wrap

    phi = np.array([340.0, 350.0, 355.0, 5.0, 15.0])
    theta = np.full(5, 90.0)
    segments = _split_at_phi_wrap(phi, theta)
    assert len(segments) == 2
    assert list(segments[0][0]) == [340.0, 350.0, 355.0]
    assert list(segments[1][0]) == [5.0, 15.0]


def test_path_stays_whole_when_it_never_wraps():
    from src.plotting import _split_at_phi_wrap

    phi = np.array([10.0, 40.0, 80.0, 120.0])
    segments = _split_at_phi_wrap(phi, np.full(4, 60.0))
    assert len(segments) == 1
    assert len(segments[0][0]) == 4


def test_connecting_the_path_can_be_turned_off(trajectory_results, tmp_path):
    _assert_png(plot_mercator(trajectory_results, str(tmp_path / "nc"), connect=False))
    _assert_png(plot_stoddart(trajectory_results, str(tmp_path / "ns"), connect=False))


def test_stoddart_writes_a_png(trajectory_results, tmp_path):
    path = plot_stoddart(trajectory_results, str(tmp_path / "job"), title="job")
    assert path == str(tmp_path / "job") + "_stoddart.png"
    _assert_png(path)


def test_stoddart_handles_a_single_frame(single_frame_results, tmp_path):
    _assert_png(plot_stoddart(single_frame_results, str(tmp_path / "one")))


@pytest.mark.parametrize("theta", [10.0, 170.0])
def test_stoddart_handles_an_empty_hemisphere(theta, tmp_path):
    """All frames in one hemisphere leaves the other dial with nothing to draw."""
    frames = [make_ring(0.57, theta, phi) for phi in (0.0, 120.0, 240.0)]
    results = compute_puckering(np.array(frames))
    _assert_png(plot_stoddart(results, str(tmp_path / f"hemi{theta:.0f}")))


def test_stoddart_does_not_leak_figures(trajectory_results, tmp_path):
    plt.close("all")
    before = len(plt.get_fignums())
    plot_stoddart(trajectory_results, str(tmp_path / "s"))
    assert len(plt.get_fignums()) == before


def test_fel_accepts_a_trajectory_overlay(fel_grid, trajectory_results, tmp_path):
    path = plot_fel_mercator(fel_grid, str(tmp_path / "ov"), overlay=trajectory_results,
                             suffix="_on_FEL")
    assert path.endswith("_on_FEL.png")
    _assert_png(path)


def test_fel_overlay_accepts_a_single_structure(fel_grid, single_frame_results, tmp_path):
    _assert_png(plot_fel_mercator(fel_grid, str(tmp_path / "one"),
                                  overlay=single_frame_results))


def test_fel_overlay_is_optional(fel_grid, tmp_path):
    _assert_png(plot_fel_mercator(fel_grid, str(tmp_path / "plain"), overlay=None))


def test_conformer_populations_writes_a_png(trajectory_results, tmp_path):
    path = plot_conformer_populations(trajectory_results, str(tmp_path / "job"), 6,
                                      title="job")
    assert path == str(tmp_path / "job") + "_populations.png"
    _assert_png(path)


def test_conformer_populations_works_for_furanoses(furanose_results, tmp_path):
    _assert_png(plot_conformer_populations(furanose_results, str(tmp_path / "fur"), 5))


def test_conformer_populations_folds_the_tail_into_other(tmp_path):
    """A trajectory touching more conformers than fit is summarised, not truncated."""
    frames = [make_ring(0.57, 90.0, phi) for phi in np.arange(0, 360, 5)]
    results = compute_puckering(np.array(frames))
    _assert_png(plot_conformer_populations(results, str(tmp_path / "many"), 6, max_bars=5))


def test_amplitude_histogram_writes_a_png(trajectory_results, tmp_path):
    path = plot_amplitude_histogram(trajectory_results, str(tmp_path / "job"), title="job")
    assert path == str(tmp_path / "job") + "_amplitude.png"
    _assert_png(path)


def test_amplitude_histogram_survives_a_constant_amplitude(tmp_path):
    """A rigid ring gives a zero-width range, which numpy refuses to bin."""
    frames = [make_ring(0.57, theta, 90.0) for theta in (30.0, 60.0, 120.0)]
    results = compute_puckering(np.array(frames))
    assert np.ptp(results[:, 1]) < 1e-12, "this fixture should have constant Q"
    _assert_png(plot_amplitude_histogram(results, str(tmp_path / "flat")))


def test_summary_plots_do_not_leak_figures(trajectory_results, tmp_path):
    plt.close("all")
    before = len(plt.get_fignums())
    plot_conformer_populations(trajectory_results, str(tmp_path / "c"), 6)
    plot_amplitude_histogram(trajectory_results, str(tmp_path / "c"))
    assert len(plt.get_fignums()) == before


def test_unsampled_bins_are_masked_by_default(fel_grid, tmp_path):
    """
    Unsampled bins must not be painted as if they had been explored. With most of
    a metadynamics surface unsampled, folding them into the ramp also costs the
    basins nearly all of the colour range.
    """
    grid = fel_grid.copy()
    grid[grid[:, 0] > 90.0, 2] = np.inf  # southern hemisphere never visited
    _assert_png(plot_fel_mercator(grid, str(tmp_path / "masked")))


def test_masking_leaves_the_scale_to_the_sampled_data(fel_grid, tmp_path):
    """The colourbar must span the measured energies, not the unsampled plateau."""
    from matplotlib import pyplot

    grid = fel_grid.copy()
    sampled_max = grid[grid[:, 0] <= 90.0, 2].max()
    grid[grid[:, 0] > 90.0, 2] = np.inf

    captured = []
    original = pyplot.subplots

    def spy(*args, **kwargs):
        fig, ax = original(*args, **kwargs)
        real = ax.contourf

        def record(*a, **kw):
            result = real(*a, **kw)
            captured.append(result.levels)
            return result

        ax.contourf = record
        return fig, ax

    pyplot.subplots = spy
    try:
        plot_fel_mercator(grid, str(tmp_path / "m"))
    finally:
        pyplot.subplots = original

    assert captured
    assert max(captured[0]) <= sampled_max + 1e-6


def test_unsampled_max_mode_still_available(fel_grid, tmp_path):
    grid = fel_grid.copy()
    grid[::11, 2] = np.inf
    _assert_png(plot_fel_mercator(grid, str(tmp_path / "clamped"), unsampled="max"))


def test_rejects_an_unknown_unsampled_mode(fel_grid, tmp_path):
    with pytest.raises(ValueError, match="unsampled must be"):
        plot_fel_mercator(fel_grid, str(tmp_path / "bad"), unsampled="blank")


def test_auto_energy_cap_uses_the_sampled_distribution(fel_grid, tmp_path):
    """'auto' must adapt to the data instead of assuming one energy scale."""
    _assert_png(plot_fel_mercator(fel_grid, str(tmp_path / "a"), energy_max="auto"))
    scaled = fel_grid.copy()
    scaled[:, 2] *= 100.0
    _assert_png(plot_fel_mercator(scaled, str(tmp_path / "b"), energy_max="auto"))


def test_rejects_an_unknown_energy_max_keyword(fel_grid, tmp_path):
    with pytest.raises(ValueError, match="energy_max must be"):
        plot_fel_mercator(fel_grid, str(tmp_path / "bad"), energy_max="high")


def test_fel_energy_cap_is_applied(fel_grid, tmp_path):
    _assert_png(plot_fel_mercator(fel_grid, str(tmp_path / "cap"), energy_max=5.0))


def test_capped_plateau_does_not_get_contoured(fel_grid, tmp_path):
    """
    Capping leaves a large exactly-constant plateau. A contour level landing on
    that value makes the algorithm chase floating-point noise and scribble
    thousands of tiny closed loops over it, so levels must stay strictly inside
    the range.
    """
    from matplotlib import pyplot

    captured = []
    original = pyplot.subplots

    def spy(*args, **kwargs):
        fig, ax = original(*args, **kwargs)
        real_contour = ax.contour

        def record(*a, **kw):
            captured.append(np.asarray(kw.get("levels", [])))
            return real_contour(*a, **kw)

        ax.contour = record
        return fig, ax

    pyplot.subplots = spy
    try:
        plot_fel_mercator(fel_grid, str(tmp_path / "cap"), energy_max=5.0, contour_step=1.0)
    finally:
        pyplot.subplots = original

    assert captured, "no contour levels were drawn"
    assert (captured[0] < 5.0).all(), f"a level sits on the plateau: {captured[0]}"


def test_a_completely_flat_surface_draws_no_contour_lines(tmp_path):
    theta, phi = np.meshgrid(np.linspace(0, 180, 20), np.linspace(0, 360, 20), indexing="ij")
    flat = np.column_stack([theta.ravel(), phi.ravel(), np.full(theta.size, 3.0)])
    _assert_png(plot_fel_mercator(flat, str(tmp_path / "flat")))


def test_fel_does_not_mutate_the_caller_array(fel_grid, tmp_path):
    original = fel_grid.copy()
    plot_fel_mercator(fel_grid, str(tmp_path / "pure"))
    assert np.array_equal(fel_grid, original)
