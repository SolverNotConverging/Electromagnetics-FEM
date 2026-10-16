"""An opaque Boolean frame must retain and classify its inner aperture wall."""
import numpy as np

from fem import ModeSolver2D, materials, shapes
from fem.waveguide.meshing import _facet_matches_shape


def test_difference_inner_wall_is_classified_without_filling_the_hole():
    frame = shapes.Difference(shape=shapes.Rectangle(bounds=((-1., 2.), (-1., 2.))),
                              tool=shapes.Rectangle(bounds=((0., 1.), (0., 1.))))
    endpoints = np.array([[0., .2], [0., .8]])
    assert _facet_matches_shape(frame, endpoints, endpoints.mean(axis=0), 1e-9)
    inside = endpoints + np.array([.5, 0.])
    assert not _facet_matches_shape(frame, inside, inside.mean(axis=0), 1e-9)


def test_copper_difference_frame_keeps_only_aperture_and_impedance_walls():
    width, height, wall = .02286, .01016, .001
    bounds = ((-wall, width+wall), (-wall, height+wall))
    solver = ModeSolver2D(frequency=10e9, x_range=bounds[0], y_range=bounds[1])
    solver.add_geometry(name='copper_frame', material=materials.copper,
        shape=shapes.Difference(shape=shapes.Rectangle(bounds=bounds),
            tool=shapes.Rectangle(bounds=((0., width), (0., height)))))
    mesh = solver.mesh(resolution=(12, 6), element_order=2, material_aware=False,
                       interface_refinement=None, boundary_refinement=None)
    centers = mesh.nodes[mesh.elements].mean(axis=1)
    assert np.all((centers[:, 0] > 0) & (centers[:, 0] < width))
    assert np.all((centers[:, 1] > 0) & (centers[:, 1] < height))
    np.testing.assert_array_equal(np.sort(mesh.boundary_facets['copper_frame']),
                                  np.sort(mesh.mesh.boundary_facets()))
    assert mesh.boundary_facets['outer_pec'].size == 0
