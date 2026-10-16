fem.scattering user API
=======================

Version 2.0.1. This reference covers the deliberately supported user API.
Implementation helpers are documented in their source modules.

Workflow and units
------------------

All lengths are in metres, frequencies in hertz, and constitutive values are relative.
Construction and configuration use keyword arguments. Call ``mesh()`` to build
the initial mesh, ``solve()`` to obtain and store a typed result, and ``show()``
to inspect it interactively. ``mesh_data`` and ``result`` are initially None.
Geometry edits invalidate both; remeshing invalidates the result. Automatic
meshing reuses the last explicit settings. Calling ``show()`` without a result
raises ``NoResultError``.

Adaptive defaults are ``max_refinements=2`` and ``adaptive_tolerance=0.05``.
Zero refinements performs one solve. ``solve_info`` distinguishes algebraic
residuals from adaptive discretization residuals and records stopping reasons.
Python mode and case indices are zero-based.

``solve()`` and frequency sweeps neither save nor open windows. Call
``result.save(path)`` explicitly. ``load_result(path)`` returns inspection-ready
results without solving. Archives use ``fem-results`` schema ``1.0``;
old or convention-incompatible files raise ``PersistenceError``. Loaded
results can plot, show, and save; they do not restart a solver or restore callbacks.
``plot()`` returns a Matplotlib Figure without opening a window.
``show(block=True)`` waits for the viewer; ``block=False`` returns immediately.

The time convention is ``exp(+i*omega*t)`` with guided propagation
``exp(-i*beta*z)``. Passive materials have nonpositive imaginary constitutive
values; passive forward attenuation is ``-Im(beta)``.

Scattering uses a two-dimensional x/z mesh for **2.5D full-vector** fields,
with invariant factor ``exp(-i*ky*y)``. Scalar epsilon and mu are supported.
Integrated power accounting requires passive materials and a lossless uniform
lead. Port modes remain a separate implementation from standalone mode solvers.
PML supports x, z, or all; it applies at both ends of the selected axis.

Supported exports
-----------------

``ClosedContourFields``, ``Diagnostic``, ``DiagnosticReport``, ``FarFieldResult``, ``IncidentMode``, ``LayeredExterior``, ``MaterialError``, ``ModeProjectionError``, ``ModeSolverError``, ``PortMode``, ``PortModeSet``, ``ScatteringResult``, ``ScatteringSolver2D``, ``ScatteringSweepResult``, ``WaveguideScatteringError``.

Solver construction and operations
----------------------------------

``ScatteringSolver2D``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D(*, frequency: 'float', angle: 'float | None' = None, ky: 'float | None' = None, x_range: 'float | Sequence[float]', z_range: 'float | Sequence[float]', background_material: 'materials.Material' = Material(name='vacuum', epsilon=(1+0j), mu=(1+0j)), boundary: 'materials.IdealBoundary | None' = None) -> 'None'

Full-vector 2.5D FEM scattering simulation in SI units.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``frequency``
     - ``float``
     - Required
     - ``—``
     - Operating frequency in hertz; finite and positive.
   * - ``angle``
     - ``float | None``
     - Optional
     - ``None``
     - Physical incidence angle in degrees, strictly between -90 and 90; mutually exclusive with ky.
   * - ``ky``
     - ``float | None``
     - Optional
     - ``None``
     - Real invariant-direction wavenumber in radians per metre; mutually exclusive with angle.
   * - ``x_range``
     - ``float | Sequence[float]``
     - Required
     - ``—``
     - Physical x extent or increasing bounds, in metres.
   * - ``z_range``
     - ``float | Sequence[float]``
     - Required
     - ``—``
     - Physical z extent or increasing bounds, in metres.
   * - ``background_material``
     - ``materials.Material``
     - Optional
     - ``Material(name='vacuum', epsilon=(1+0j), mu=(1+0j))``
     - Predefined bulk Material assigned to unfilled space.
   * - ``boundary``
     - ``materials.IdealBoundary | None``
     - Optional
     - ``None``
     - Predefined PEC or PMC exterior-boundary material.

Returns: a configured ``ScatteringSolver2D``.

``ScatteringSolver2D.add_geometry``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_geometry(*, shape, material, name=None, clip=False, background=False)

Assign a material; background objects also belong to the straight lead.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``shape``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined fem.common.shapes object in metres.
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined bulk, ideal-boundary, or supported SIBC material.
   * - ``name``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.
   * - ``clip``
     - ``bool``
     - Optional
     - ``False``
     - Intersect the shape with the solver domain; otherwise out-of-bounds objects raise GeometryError.
   * - ``background``
     - ``bool``
     - Optional
     - ``False``
     - Include this region in both the unperturbed lead and actual device.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.add_rectangle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_rectangle(*, x_range, z_range, material, name=None, clip=False, background=False)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x_range``
     - ``float | tuple[float, float] / m``
     - Required
     - ``—``
     - Physical x extent or increasing bounds, in metres.
   * - ``z_range``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Physical z extent or increasing bounds, in metres.
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined bulk, ideal-boundary, or supported SIBC material.
   * - ``name``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.
   * - ``clip``
     - ``bool``
     - Optional
     - ``False``
     - Intersect the shape with the solver domain; otherwise out-of-bounds objects raise GeometryError.
   * - ``background``
     - ``bool``
     - Optional
     - ``False``
     - Include this region in both the unperturbed lead and actual device.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.add_circle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_circle(*, center, radius, material, name=None, clip=False)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``center``
     - ``tuple[float, ...] / m``
     - Required
     - ``—``
     - Physical centre coordinates in metres.
   * - ``radius``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Positive radius in metres.
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined bulk, ideal-boundary, or supported SIBC material.
   * - ``name``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.
   * - ``clip``
     - ``bool``
     - Optional
     - ``False``
     - Intersect the shape with the solver domain; otherwise out-of-bounds objects raise GeometryError.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.add_polygon``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_polygon(*, points, material, name=None, clip=False)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``points``
     - ``sequence[tuple[float, ...]] / m``
     - Required
     - ``—``
     - Ordered polygon vertex coordinates in metres.
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined bulk, ideal-boundary, or supported SIBC material.
   * - ``name``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.
   * - ``clip``
     - ``bool``
     - Optional
     - ``False``
     - Intersect the shape with the solver domain; otherwise out-of-bounds objects raise GeometryError.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.add_slot``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_slot(*, geometry, z_range, name=None)

Cut a finite opening in an existing background PEC sheet.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``geometry``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A handle returned by add_geometry or a geometry convenience method.
   * - ``z_range``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Physical z extent or increasing bounds, in metres.
   * - ``name``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Optional name used for later identification and diagnostics.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.set_material``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.set_material(*, geometry, material)

Reassign a predefined material and invalidate mesh/result.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``geometry``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A handle returned by add_geometry or a geometry convenience method.
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined bulk, ideal-boundary, or supported SIBC material.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.set_shape``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.set_shape(*, geometry, shape)

Replace a shape in metres and invalidate mesh/result.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``geometry``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A handle returned by add_geometry or a geometry convenience method.
   * - ``shape``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined fem.common.shapes object in metres.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.set_material_field``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.set_material_field(*, material, background_material)

Use named actual/background SpatialMaterial fields instead of objects.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined bulk, ideal-boundary, or supported SIBC material.
   * - ``background_material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - Predefined bulk Material assigned to unfilled space.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.set_boundary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.set_boundary(*, material)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``material``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A predefined bulk, ideal-boundary, or supported SIBC material.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.add_pml``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.add_pml(*, thickness: 'float', direction: 'str' = 'all', order: 'int' = 3, target_reflection: 'float' = 1e-08) -> 'None'

Add matched layers on both ends of the selected x/z axis.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``thickness``
     - ``float``
     - Required
     - ``—``
     - PML thickness in metres, at each selected exterior end.
   * - ``direction``
     - ``str``
     - Optional
     - ``'all'``
     - Propagation direction for solve; selected coordinate direction for PML.
   * - ``order``
     - ``int``
     - Optional
     - ``3``
     - Polynomial order of the PML profile.
   * - ``target_reflection``
     - ``float``
     - Optional
     - ``1e-08``
     - Desired PML amplitude reflection ratio in (0, 1).

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.set_monitors``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.set_monitors(*, left: 'float', right: 'float') -> 'None'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``left``
     - ``float``
     - Required
     - ``—``
     - Left monitor/reference-plane position in metres.
   * - ``right``
     - ``float``
     - Required
     - ``—``
     - Right monitor/reference-plane position in metres.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.set_matched_ports``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.set_matched_ports(enabled: 'bool' = True) -> 'None'

Use outgoing modal maps at both z ends instead of longitudinal PML.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``enabled``
     - ``bool``
     - Optional
     - ``True``
     - Enable or disable near-to-far-field recording.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.set_nf2ff_contour``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.set_nf2ff_contour(*, x_range=None, z_range=None, exterior=None) -> 'None'

Capture a closed rectangular scattered-field Huygens contour.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``x_range``
     - ``float | tuple[float, float] / m``
     - Optional
     - ``None``
     - Physical x extent or increasing bounds, in metres.
   * - ``z_range``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - Physical z extent or increasing bounds, in metres.
   * - ``exterior``
     - ``array-like or scalar``
     - Optional
     - ``None``
     - LayeredExterior description of the medium outside the near-to-far-field contour.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.remove``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.remove(*, geometry)

Remove an owned geometry object or the slot returned by add_slot().

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``geometry``
     - ``array-like or scalar``
     - Required
     - ``—``
     - A handle returned by add_geometry or a geometry convenience method.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.mesh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.mesh(*, max_element_size: 'float | None' = None, element_order: 'int' = 1, quadrature_order: 'int' = 4, wavelength_elements: 'int' = 4, refine_interfaces: 'bool' = True, dielectric_refinement_factor: 'float' = 0.5, pec_refinement_factor: 'float' = 0.5, pec_refinement_distance: 'float | None' = None) -> 'Mesh2D'

Generate the Gmsh mesh and reveal the selected maximum edge size.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``max_element_size``
     - ``float | None``
     - Optional
     - ``None``
     - Maximum initial element edge length in metres.
   * - ``element_order``
     - ``int``
     - Optional
     - ``1``
     - Finite-element polynomial order supported by this backend.
   * - ``quadrature_order``
     - ``int``
     - Optional
     - ``4``
     - Element integration order.
   * - ``wavelength_elements``
     - ``int``
     - Optional
     - ``4``
     - Minimum number of initial elements per local wavelength.
   * - ``refine_interfaces``
     - ``bool``
     - Optional
     - ``True``
     - Enable local refinement at dielectric interfaces.
   * - ``dielectric_refinement_factor``
     - ``float``
     - Optional
     - ``0.5``
     - Mesh-size multiplier near dielectric interfaces.
   * - ``pec_refinement_factor``
     - ``float``
     - Optional
     - ``0.5``
     - Mesh-size multiplier near PEC boundaries.
   * - ``pec_refinement_distance``
     - ``float | None``
     - Optional
     - ``None``
     - Distance in metres over which PEC mesh refinement applies.

Returns: the initial mesh stored in ``mesh_data``.

``ScatteringSolver2D.solve_modes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.solve_modes(*, backend: 'str' = 'auto', side: "Literal['left', 'right']" = 'left', num_modes: 'int' = 4, neff_guess: 'complex | None' = None, num_elements: 'int | None' = None, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05) -> 'PortModeSet'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``backend``
     - ``str``
     - Optional
     - ``'auto'``
     - Execution backend: auto (default) selects GPU when available, otherwise SciPy CPU; cpu and gpu force the selected backend.
   * - ``side``
     - ``Literal['left', 'right']``
     - Optional
     - ``'left'``
     - Port label, left or right.
   * - ``num_modes``
     - ``int``
     - Optional
     - ``4``
     - Number of modes requested; positive integer.
   * - ``neff_guess``
     - ``complex | None``
     - Optional
     - ``None``
     - Dimensionless complex effective-index search target.
   * - ``num_elements``
     - ``int | None``
     - Optional
     - ``None``
     - Number of elements in the lead-mode cross-section discretization.
   * - ``max_refinements``
     - ``int``
     - Optional
     - ``2``
     - Maximum mesh refinements after the initial solve; zero means one solve.
   * - ``adaptive_tolerance``
     - ``float``
     - Optional
     - ``0.05``
     - Relative discretization-residual stopping threshold.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.set_incident_mode``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.set_incident_mode(mode: 'int | PortMode', *, side: "Literal['left', 'right']" = 'left', reference_plane: 'float | None' = None, amplitude: 'complex' = 1.0) -> 'IncidentMode'

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``mode``
     - ``int | PortMode``
     - Required
     - ``—``
     - Zero-based mode index.
   * - ``side``
     - ``Literal['left', 'right']``
     - Optional
     - ``'left'``
     - Port label, left or right.
   * - ``reference_plane``
     - ``float | None``
     - Optional
     - ``None``
     - Incident phase reference position in metres.
   * - ``amplitude``
     - ``complex``
     - Optional
     - ``1.0``
     - Complex incident-mode amplitude.

Returns: the configured geometry/excitation handle, or None for in-place configuration.

``ScatteringSolver2D.solve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.solve(*, linear_solver: "Literal['direct']" = 'direct', linear_solver_tolerance: 'float' = 1e-10, projection_condition_limit: 'float' = 1000000000000.0, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05, backend: 'str' = 'auto') -> 'ScatteringResult'

Solve without saving or opening a viewer.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``linear_solver``
     - ``Literal['direct']``
     - Optional
     - ``'direct'``
     - Sparse linear-system algorithm; currently direct.
   * - ``linear_solver_tolerance``
     - ``float``
     - Optional
     - ``1e-10``
     - Algebraic linear-system residual tolerance.
   * - ``projection_condition_limit``
     - ``float``
     - Optional
     - ``1000000000000.0``
     - Maximum permitted condition number for port-mode projection.
   * - ``max_refinements``
     - ``int``
     - Optional
     - ``2``
     - Maximum mesh refinements after the initial solve; zero means one solve.
   * - ``adaptive_tolerance``
     - ``float``
     - Optional
     - ``0.05``
     - Relative discretization-residual stopping threshold.
   * - ``backend``
     - ``str``
     - Optional
     - ``'auto'``
     - Execution backend: auto (default) selects GPU when available, otherwise SciPy CPU; cpu and gpu force the selected backend.

Returns: the physics-specific result stored in ``result``.

``ScatteringSolver2D.sweep``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.sweep(frequencies: 'Sequence[float]', *, backend: 'str' = 'auto', linear_solver: "Literal['direct']" = 'direct', linear_solver_tolerance: 'float' = 1e-10, projection_condition_limit: 'float' = 1000000000000.0, max_refinements: 'int' = 2, adaptive_tolerance: 'float' = 0.05, mesh_options: 'Mapping[str, object] | None' = None, mode_options: 'Mapping[str, object] | None' = None, incident_mode: 'int' = 0, amplitude: 'complex' = 1.0, reference_plane: 'float | None' = None, mode_factory: 'Callable[[float], PortModeSet] | None' = None) -> 'ScatteringSweepResult'

Solve independent points on an increasing ordinary-frequency grid.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``frequencies``
     - ``Sequence[float]``
     - Required
     - ``—``
     - Strictly increasing positive frequencies in hertz.
   * - ``backend``
     - ``str``
     - Optional
     - ``'auto'``
     - Execution backend: auto (default) selects GPU when available, otherwise SciPy CPU; cpu and gpu force the selected backend.
   * - ``linear_solver``
     - ``Literal['direct']``
     - Optional
     - ``'direct'``
     - Sparse linear-system algorithm; currently direct.
   * - ``linear_solver_tolerance``
     - ``float``
     - Optional
     - ``1e-10``
     - Algebraic linear-system residual tolerance.
   * - ``projection_condition_limit``
     - ``float``
     - Optional
     - ``1000000000000.0``
     - Maximum permitted condition number for port-mode projection.
   * - ``max_refinements``
     - ``int``
     - Optional
     - ``2``
     - Maximum mesh refinements after the initial solve; zero means one solve.
   * - ``adaptive_tolerance``
     - ``float``
     - Optional
     - ``0.05``
     - Relative discretization-residual stopping threshold.
   * - ``mesh_options``
     - ``Mapping[str, object] | None``
     - Optional
     - ``None``
     - Keyword arguments forwarded to mesh() for each frequency case.
   * - ``mode_options``
     - ``Mapping[str, object] | None``
     - Optional
     - ``None``
     - Keyword arguments forwarded to solve_modes() for each frequency case.
   * - ``incident_mode``
     - ``int``
     - Optional
     - ``0``
     - Zero-based incident lead-mode index.
   * - ``amplitude``
     - ``complex``
     - Optional
     - ``1.0``
     - Complex incident-mode amplitude.
   * - ``reference_plane``
     - ``float | None``
     - Optional
     - ``None``
     - Incident phase reference position in metres.
   * - ``mode_factory``
     - ``Callable[[float], PortModeSet] | None``
     - Optional
     - ``None``
     - Callback constructing incident-mode data at each frequency; see the sweep guide.

Returns: a frequency sweep result.

``ScatteringSolver2D.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSolver2D.show(*, block: 'bool' = True)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``block``
     - ``bool``
     - Optional
     - ``True``
     - Wait for the interactive viewer to close when true.

Returns: the interactive viewer window.

Returned results
----------------

Result objects are returned by solving or loading. Their constructors are
implementation details. Inspect ``mesh_data``, ``metadata``, ``solve_info``, and
``frequency`` where applicable. Modal sets support iteration, indexing, ``neff``
and ``beta``; each mode provides sampled fields and residual diagnostics.

``ScatteringResult.plot``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.plot(*, component: 'str | None' = None, quantity: 'str' = 'real')

Return a sampled-field figure without opening a window.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``component``
     - ``str | None``
     - Optional
     - ``None``
     - Field component to display, such as Ey; electrostatics also accepts potential or mesh.
   * - ``quantity``
     - ``str``
     - Optional
     - ``'real'``
     - Displayed field quantity: real, imag, magnitude/abs, or phase; static fields support real or magnitude.

Returns: a Matplotlib Figure.

``ScatteringResult.show``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.show(*, block: 'bool' = True)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``block``
     - ``bool``
     - Optional
     - ``True``
     - Wait for the interactive viewer to close when true.

Returns: the interactive viewer window.

``ScatteringResult.save``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.save(path)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``path``
     - ``str | PathLike``
     - Required
     - ``—``
     - Destination/source HDF5 path. Saving is atomic; loading does not run a solver.

Returns: the written Path.

``ScatteringResult.S``
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.S(side: 'str', *, out_mode: 'int' = 0, in_mode: 'int' = 0) -> 'complex'

Return an indexed outgoing modal amplitude.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``side``
     - ``str``
     - Required
     - ``—``
     - Port label, left or right.
   * - ``out_mode``
     - ``int``
     - Optional
     - ``0``
     - Zero-based outgoing mode index.
   * - ``in_mode``
     - ``int``
     - Optional
     - ``0``
     - Zero-based incident mode index.

Returns: the selected data or diagnostic report.

``ScatteringResult.check``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.check(*, power_balance_tolerance: 'float' = 0.001, projection_condition_warning: 'float' = 10000000000.0, projection_residual_warning: 'float' = 0.001, incoming_projection_warning: 'float' = 0.001, port_gram_diagonal_warning: 'float' = 0.01, s_parameter_power_tolerance: 'float' = 1e-06) -> 'DiagnosticReport'

Return structured solver-quality diagnostics without printing.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``power_balance_tolerance``
     - ``float``
     - Optional
     - ``0.001``
     - Maximum acceptable absolute power-balance error in diagnostics.
   * - ``projection_condition_warning``
     - ``float``
     - Optional
     - ``10000000000.0``
     - Projection condition-number threshold for a diagnostic warning.
   * - ``projection_residual_warning``
     - ``float``
     - Optional
     - ``0.001``
     - Projection relative-residual threshold for a diagnostic warning.
   * - ``incoming_projection_warning``
     - ``float``
     - Optional
     - ``0.001``
     - Incoming-wave projection relative-error threshold for a diagnostic warning.
   * - ``port_gram_diagonal_warning``
     - ``float``
     - Optional
     - ``0.01``
     - Port Gram-matrix diagonal-error threshold for a diagnostic warning.
   * - ``s_parameter_power_tolerance``
     - ``float``
     - Optional
     - ``1e-06``
     - Tolerance for disagreement between S-parameter and integrated power ratios.

Returns: the selected data or diagnostic report.

``ScatteringResult.deembed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.deembed(*, left: 'float', right: 'float') -> "'ScatteringResult'"

Move left/right reference planes for a left-incident result.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``left``
     - ``float``
     - Required
     - ``—``
     - Left monitor/reference-plane position in metres.
   * - ``right``
     - ``float``
     - Required
     - ``—``
     - Right monitor/reference-plane position in metres.

Returns: a result with updated reference planes.

``ScatteringResult.far_field``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringResult.far_field(theta: 'ArrayLike') -> 'FarFieldResult'

Closed-contour radiation and 2D antenna patterns at angles in radians.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``theta``
     - ``ArrayLike``
     - Required
     - ``—``
     - Observation angles in radians for far-field evaluation.

Returns: the selected data or diagnostic report.

``ScatteringSweepResult.plot``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSweepResult.plot(*, component: 'str | None' = None, quantity: 'str' = 'magnitude')

Return an S11/S21 spectrum figure; frequency is measured in hertz.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``component``
     - ``str | None``
     - Optional
     - ``None``
     - Field component to display, such as Ey; electrostatics also accepts potential or mesh.
   * - ``quantity``
     - ``str``
     - Optional
     - ``'magnitude'``
     - Displayed field quantity: real, imag, magnitude/abs, or phase; static fields support real or magnitude.

Returns: a Matplotlib Figure.

``ScatteringSweepResult.show``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSweepResult.show(*, block: 'bool' = True)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``block``
     - ``bool``
     - Optional
     - ``True``
     - Wait for the interactive viewer to close when true.

Returns: the interactive viewer window.

``ScatteringSweepResult.save``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSweepResult.save(path)

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``path``
     - ``str | PathLike``
     - Required
     - ``—``
     - Destination/source HDF5 path. Saving is atomic; loading does not run a solver.

Returns: the written Path.

``ScatteringSweepResult.S``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ScatteringSweepResult.S(side: 'str', *, out_mode: 'int' = 0, in_mode: 'int' = 0) -> 'ComplexArray'

Return one indexed modal amplitude across the sweep.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``side``
     - ``str``
     - Required
     - ``—``
     - Port label, left or right.
   * - ``out_mode``
     - ``int``
     - Optional
     - ``0``
     - Zero-based outgoing mode index.
   * - ``in_mode``
     - ``int``
     - Optional
     - ``0``
     - Zero-based incident mode index.

Returns: the selected data or diagnostic report.

Result data and diagnostics
---------------------------

``mesh_data.coordinates`` stores physical nodes in metres; ``elements`` stores
zero-based connectivity. ``axes`` identifies physical coordinate order.
``mesh_data.metadata['context']`` records material and boundary configuration.
The result is an inspection snapshot; editing it cannot restart a solver.

``E_total``, ``E_incident``, ``E_scattered`` and the matching ``H_*``
arrays have shape (3, samples), in V/m and A/m. ``coordinates`` uses (x, z),
in metres. ``S(side, out_mode, in_mode)`` returns a complex modal amplitude;
``S11`` and ``S21`` select the fundamental ports. ``reflection``,
``transmission``, ``absorption`` and ``power_balance_error`` are power ratios.
``reference_planes`` stores positions in m; ``port_betas`` stores complex
wavenumbers in rad/m. ``solve_info`` retains projection, algebraic, and
adaptive diagnostics. Sweep ``results[index]`` loads one case;
``frequencies`` is the ordered frequency array. Sweep plotting accepts
S11/S21 and quantity real, imag, phase, magnitude, abs, or db.

Geometry and material values
----------------------------

Define reusable materials and shapes with ``fem.common`` before assigning them.
Use ``Material(name=..., epsilon=..., mu=...)`` for bulk media,
``materials.PEC`` or ``materials.PMC`` for ideal boundaries, and the
documented ``materials.copper``-style presets where SIBC is supported.
Continuous primitives and Boolean/transformed shapes live in
``fem.common.shapes``. Solver packages also re-export Material, materials, and shapes.

``fem.load``
~~~~~~~~~~~~

.. code-block:: python

    fem.load(path: 'str | os.PathLike[str]')

Load any FEM result archive; the archive records which solver wrote it.

.. list-table:: Arguments
   :header-rows: 1
   :widths: 16 20 12 16 36

   * - Argument
     - Type / units
     - Required / optional
     - Default
     - Meaning
   * - ``path``
     - ``str | os.PathLike[str]``
     - Required
     - ``—``
     - Destination/source HDF5 path. Saving is atomic; loading does not run a solver.

Returns: a typed result; multi-case archives provide lazy case access.

Errors
------

Invalid inputs raise ``ConfigurationError`` or ``GeometryError`` where available.
Mesh and numerical failures raise the corresponding ``MeshError`` or
``SolverError``. ``NoResultError`` requires a successful solve first.
``PersistenceError`` identifies an incompatible or unreadable archive.
Viewer errors include the executable path or installation setting needed to
correct a launch failure. Saving and loading do not require an active GUI.

Backend selection
-----------------

``backend="auto"`` selects GPU when available, otherwise SciPy CPU. ``backend="gpu"`` requests
CuPy/cuDSS explicitly. See the solver guide for supported algorithms.

``sweep(backend="gpu")`` selects both device and lead-mode solves;
``mode_options["backend"]`` overrides the lead-mode backend.
