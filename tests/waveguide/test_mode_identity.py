"""Physical symmetry, residual-safe basis selection, and mesh-independent identity."""
import numpy as np
import pytest
from skfem import MeshTri

from fem import load
from fem.common import materials, Material, shapes
from fem.common.constants import C_0
from fem.common.errors import ConfigurationError
from fem.waveguide import ModeSolver2D
from fem.waveguide._mode_identity import detect_symmetry, _longitudinal_sectors, _sample_operators
from fem.waveguide.assembly import assemble_mode_system_2d, solve_qep_candidates
from fem.waveguide.meshing import FEMMesh2D, MeshInfo


def structured_solver(*, height=1., order=1, perturb=False):
    solver = ModeSolver2D(frequency=2*C_0, x_range=1., y_range=height)
    mesh = MeshTri.init_sqsymmetric().refined(2).scaled(np.array([1., height]))
    if perturb:
        points = mesh.p.copy()
        interior = np.setdiff1d(np.arange(mesh.nvertices), mesh.boundary_nodes())
        points[:, interior] += (np.random.default_rng(42).uniform(-.12, .12, (2, len(interior)))
                                * np.array([[1/8], [height/8]]))
        mesh = MeshTri(points, mesh.t.copy())
    solver._system = assemble_mode_system_2d(mesh, frequency=solver.frequency, k0=solver.k0,
        material_at=solver.geometry.material_at, element_order=order)
    solver._mesh_data = FEMMesh2D(mesh, np.ones(mesh.nelements, dtype=np.int32), {1: 'background'},
        {}, MeshInfo(mesh.nvertices, mesh.nelements, .01, .2, .2, order), solver.geometry.revision)
    solver._discretized_revision = solver.geometry.revision
    return solver


def solve(solver, **options):
    return solver.solve(num_modes=2, neff_guess=np.sqrt(15/16), backend='cpu',
        dense_linearization_limit=4, max_refinements=0, **options)


def test_continuous_symmetry_includes_tensors_boundaries_and_pml():
    square = ModeSolver2D(frequency=2*C_0, x_range=1., y_range=1.)
    group = detect_symmetry(square, 'auto')
    assert group.name == 'C4v'
    assert square.mesh_data is None
    assert detect_symmetry(square, 'auto') is group
    rectangle = ModeSolver2D(frequency=2*C_0, x_range=1., y_range=.7)
    assert detect_symmetry(rectangle, 'auto').name == 'C2v'
    anisotropic = ModeSolver2D(frequency=2*C_0, x_range=1., y_range=1.,
        background_material=Material(epsilon=(2., 3., 4.)))
    assert detect_symmetry(anisotropic, 'auto').name == 'C2v'
    with pytest.raises(ConfigurationError, match='do not support'):
        detect_symmetry(anisotropic, 'C4v')
    square.add_rectangle(x_range=(.11, .23), y_range=(.27, .38), material=Material(epsilon=2.))
    assert square._symmetry_cache is None
    assert detect_symmetry(square, 'auto').name == 'C1'
    conductor = ModeSolver2D(frequency=2*C_0, x_range=1., y_range=1.)
    conductor.add_rectangle(x_range=(.11, .23), y_range=(.27, .38), material=materials.PEC)
    assert detect_symmetry(conductor, 'auto').name == 'C1'
    pml = ModeSolver2D(frequency=2*C_0, x_range=1., y_range=1.)
    pml.add_pml(thickness=.15, direction='x-')
    assert detect_symmetry(pml, 'auto').name == 'C1v'
    with pytest.raises(ConfigurationError, match='do not support'):
        detect_symmetry(pml, 'C2v')


def test_c6v_is_detected_from_a_continuous_hexagonal_conductor_frame():
    solver = ModeSolver2D(frequency=2*C_0, x_range=1., y_range=1.)
    angles = np.arange(6)*np.pi/3
    vertices = tuple(zip(.5+.43*np.cos(angles), .5+.43*np.sin(angles)))
    frame = shapes.Difference(shape=shapes.Rectangle(bounds=((0., 1.), (0., 1.))),
                              tool=shapes.Polygon(points=vertices))
    solver.add_geometry(shape=frame, material=materials.PEC)
    assert detect_symmetry(solver, 'auto').name == 'C6v'
    assert detect_symmetry(solver, 'C6v').name == 'C6v'


def test_exact_doublet_has_one_irrep_two_partners_and_stable_phase(monkeypatch):
    solver = structured_solver()
    first = solve(solver, track_modes=True, track_options={'symmetry': 'C4v'})
    assert {m.metadata['symmetry_irrep'] for m in first} == {'C4v:E1'}
    assert [m.metadata['symmetry_partner'] for m in first] == [1, -1]
    assert all(m.metadata['symmetry_projection'] == 'applied' for m in first)
    assert abs(first[0].neff-first[1].neff) < 1e-12
    # Simulate another eigensolver returning a different complex basis of the
    # same exact eigenspace. Identity must not depend on that arbitrary basis.
    import fem.waveguide.solver_2d as module
    original = module.solve_qep_candidates
    def rotated(*args, **kwargs):
        values, vectors, method = original(*args, **kwargs)
        vectors[:, :2] = vectors[:, :2] @ np.array([[1, 1j], [1j, 1]])/np.sqrt(2)
        return values, vectors, method
    monkeypatch.setattr(module, 'solve_qep_candidates', rotated)
    second = solve(solver, track_modes=True, track_options={'symmetry': 'C4v'})
    for old, new in zip(first, second):
        assert old.metadata['mode_id'] == new.metadata['mode_id']
        assert new.metadata['tracking_overlap'] > 1-1e-10
        assert new.metadata['subspace_overlap'] > 1-1e-10
        assert new.residual < 1e-8 and new.divergence_residual < 1e-7
        overlap = np.vdot(old.metadata['coefficient_vector'], new.metadata['coefficient_vector'])
        assert overlap.real > 0 and abs(overlap.imag) < 1e-8*abs(overlap)


def test_tracking_without_symmetry_aligns_an_arbitrary_degenerate_basis(monkeypatch):
    solver = structured_solver()
    first = solve(solver, track_modes=True, track_options={'symmetry': None})
    import fem.waveguide.solver_2d as module
    original = module.solve_qep_candidates
    def rotated(*args, **kwargs):
        values, vectors, method = original(*args, **kwargs)
        vectors[:, :2] = vectors[:, :2] @ np.array([[.6, -.8], [.8, .6]])
        return values, vectors, method
    monkeypatch.setattr(module, 'solve_qep_candidates', rotated)
    second = solve(solver, track_modes=True, track_options={'symmetry': None})
    assert all(m.metadata['symmetry_irrep'] is None for m in second)
    assert all(m.metadata['subspace_overlap'] > 1-1e-10 for m in second)
    assert all(m.metadata['tracking_overlap'] > 1-1e-10 for m in second)
    assert [m.metadata['mode_id'] for m in second] == [m.metadata['mode_id'] for m in first]


def test_split_mesh_modes_are_not_silently_replaced_by_invalid_combinations():
    solver = structured_solver(perturb=True)
    raw = solve(solver)
    identified = solve(solver, track_modes=True, track_options={'symmetry': 'C4v', 'degeneracy_tolerance': 1e-4})
    np.testing.assert_allclose(np.sort(raw.neff.real), np.sort(identified.neff.real), atol=1e-12)
    assert all(m.metadata['symmetry_projection'] == 'residual_rejected' for m in identified)
    assert all(m.residual < 1e-8 for m in identified)
    assert [m.metadata['symmetry_partner'] for m in identified] == [1, -1]


def test_requested_single_mode_uses_complete_doublet_before_truncation():
    solver = structured_solver()
    result = solver.solve(num_modes=1, neff_guess=np.sqrt(15/16), track_modes=True, track_options={'symmetry': 'C4v'},
                          max_refinements=0, backend='cpu', dense_linearization_limit=4)
    assert len(result) == 1
    assert result.metadata['identity_candidates'] == 2
    assert result.metadata['truncated_degenerate_cluster']
    assert result[0].metadata['cluster_size'] == 2
    assert result[0].metadata['symmetry_partner'] == 1


def test_incomplete_eigensolver_multiplet_is_reported_unresolved(monkeypatch):
    solver = structured_solver()
    complete = solve(solver, track_modes=True, track_options={'symmetry': 'C4v'})
    value, vector = complete[0].neff, solver.eigenvectors[:, :1].copy()
    import fem.waveguide.solver_2d as module
    monkeypatch.setattr(module, 'solve_qep_candidates',
        lambda *args, **kwargs: (np.array([value]), vector, 'test-incomplete'))
    result = solver.solve(num_modes=1, neff_guess=np.sqrt(15/16), track_modes=True, track_options={'symmetry': 'C4v'},
                          max_refinements=0, backend='cpu', dense_linearization_limit=4)
    assert not result[0].metadata['symmetry_resolved']
    assert result[0].metadata['symmetry_irrep'] is None
    assert result[0].metadata['symmetry_subspace_residual'] > .9


def test_same_irrep_te_tm_remain_distinct():
    solver = structured_solver(height=.7, order=2)
    target = np.sqrt(1-(1+1/.7**2)/16)
    result = solver.solve(num_modes=2, neff_guess=target, track_modes=True,
        track_options={'symmetry': 'C2v', 'degeneracy_tolerance': .005}, max_refinements=0, backend='cpu', dense_linearization_limit=4)
    assert len({m.metadata['symmetry_irrep'] for m in result}) == 1
    assert {m.polarization for m in result} == {'TE', 'TM'}
    for mode in result:
        key = 'longitudinal_electric_fraction' if mode.polarization == 'TE' else 'longitudinal_magnetic_fraction'
        assert mode.metadata[key] < 1e-12


def test_longitudinal_kernels_recover_te_tm_from_hybrid_combinations():
    solver = structured_solver(height=.7, order=2)
    target = np.sqrt(1-(1+1/.7**2)/16)
    values, vectors, _ = solve_qep_candidates(solver.system, target=target, candidate_count=8,
        tolerance=1e-11, dense_linearization_limit=4, backend='cpu')
    mixed = vectors[:, :2] @ np.array([[1., 1j], [1j, 1.]])/np.sqrt(2)
    _, operators = _sample_operators(solver, solver.system, detect_symmetry(solver, 'C2v'))
    recovered, sectors = _longitudinal_sectors(
        solver, solver.system, mixed, target, operators['identity'])
    assert [len(sector) for sector in sectors] == [1, 1]
    classifications = []
    for vector in recovered.T:
        fields = solver._field_data(solver.system, solver.system.expand(vector), target)['fields']
        classifications.append(solver._polarization(fields))
    assert classifications == ['TE', 'TM']


@pytest.mark.gmsh
def test_identity_survives_remeshing_and_persistence(tmp_path):
    solver = ModeSolver2D(frequency=2*C_0, x_range=1., y_range=1.)
    solver.mesh(resolution=(6, 6), element_order=2)
    first = solve(solver, track_modes=True)
    cached = solver._symmetry_cache
    solver.refine(1.4)
    second = solve(solver, track_modes=True)
    assert solver._symmetry_cache is cached
    assert [m.metadata['mode_id'] for m in second] == [m.metadata['mode_id'] for m in first]
    assert [m.metadata['symmetry_partner'] for m in second] == [1, -1]
    assert all(m.metadata['tracking_overlap'] > .999 for m in second)
    assert all(m.residual < 1e-8 for m in second)
    second.save(tmp_path/'identified.h5')
    restored = load(tmp_path/'identified.h5')
    assert restored[0].metadata['symmetry_irrep'] == 'C4v:E1'
    assert restored[1].metadata['mode_id'] == second[1].metadata['mode_id']
    solver.set_boundary(material=materials.PMC)
    assert solver._mode_reference is None and solver._symmetry_cache is None


@pytest.mark.gmsh
def test_automatic_refinement_matches_subspaces_between_passes():
    solver = ModeSolver2D(frequency=2*C_0, x_range=1., y_range=1.)
    solver.mesh(resolution=(5, 5), element_order=2)
    result = solver.solve(num_modes=2, neff_guess=np.sqrt(15/16), backend='cpu',
        track_modes=True, max_refinements=1, adaptive_tolerance=1e-5,
        dense_linearization_limit=4)
    assert len(result.metadata['adaptive_history']) == 2
    assert [m.metadata['mode_id'] for m in result] == [0, 1]
    assert all(m.metadata['tracking_overlap'] > .99 for m in result)
    assert all(m.metadata['subspace_overlap'] > .99 for m in result)


@pytest.mark.parametrize('options', [
    {'track_options': {'symmetry': 'C9v'}}, {'track_options': {'symmetry': True}},
    {'track_options': {'degeneracy_tolerance': 0}},
    {'track_options': {'degeneracy_tolerance': float('nan')}}, {'track_modes': 1},
    {'track_options': []}, {'track_options': {'unknown': 1}},
    {'track_options': {'overlap_threshold': 1.1}},
    {'track_options': {'overlap_threshold': 0}},
    {'track_options': {'symmetry_tolerance': -1}},
])
def test_invalid_identity_controls(options):
    with pytest.raises(ConfigurationError):
        solve(structured_solver(), **options)


def test_partial_track_options_use_defaults_and_do_not_mutate_input():
    solver = structured_solver()
    options = {'overlap_threshold': .8}
    first = solve(solver, track_modes=True, track_options=options)
    assert options == {'overlap_threshold': .8}
    assert first.metadata['track_options'] == dict(symmetry='auto', degeneracy_tolerance=1e-5,
        overlap_threshold=.8, symmetry_tolerance=.25)
    second = solve(solver, track_modes=True, track_options={})
    assert [m.metadata['mode_id'] for m in second] == [0, 1]
    assert all(m.metadata['tracking_overlap'] > .99 for m in second)
    disabled = solve(solver, track_modes=False, track_options=options)
    assert 'mode_id' not in disabled[0].metadata
    assert solver._mode_reference is None
