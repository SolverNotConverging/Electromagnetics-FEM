"""Symmetry and modal identity for the full-vector 2D propagation pencil.

Symmetry is a property of the continuous scene.  Detection uses deterministic
physical probes, not mesh connectivity.  Eigenvectors are rotated only when
the resulting vectors still satisfy the original pencil and Gauss law.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import re

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix

from .exceptions import ConfigurationError
from fem.common.validation import positive_real


def tracking_options(enabled, options):
    """Validate an optional partial configuration without mutating the caller."""
    if not isinstance(enabled, (bool, np.bool_)):
        raise ConfigurationError('track_modes must be boolean.')
    defaults = dict(symmetry='auto', degeneracy_tolerance=1e-5,
                    overlap_threshold=.5, symmetry_tolerance=.25)
    if options is not None:
        if not isinstance(options, Mapping):
            raise ConfigurationError('track_options must be a mapping or None.')
        unknown = set(options)-set(defaults)
        if unknown:
            raise ConfigurationError(f'Unknown track_options keys: {sorted(unknown, key=repr)!r}.')
        defaults.update(options)
    symmetry = defaults['symmetry']
    if symmetry is not None and (not isinstance(symmetry, str) or not (
            symmetry == 'auto' or re.fullmatch(r'C[1-8]v', symmetry))):
        raise ConfigurationError("track_options['symmetry'] must be None, 'auto', or 'C1v' through 'C8v'.")
    for name in ('degeneracy_tolerance', 'overlap_threshold', 'symmetry_tolerance'):
        defaults[name] = positive_real(defaults[name], f'track_options[{name!r}]')
    if defaults['overlap_threshold'] > 1:
        raise ConfigurationError("track_options['overlap_threshold'] must be in (0, 1].")
    if not enabled:
        defaults['symmetry'] = None
    return defaults


@dataclass(frozen=True)
class Symmetry:
    name: str
    n: int
    rotation: np.ndarray
    mirror: np.ndarray | None


def _rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _transform(points, matrix, center):
    return center[:, None] + matrix @ (points - center[:, None])


def _scene_probes(geometry):
    # Feature-local probes keep small objects from being lost in a global grid.
    bounds = [(*geometry.x_span, *geometry.y_span)]
    bounds += [item.shape.bounds for item in (*geometry.regions, *geometry.boundaries)]
    probes = []
    for xmin, xmax, ymin, ymax in bounds:
        t = (np.arange(13) + .371) / 13
        x, y = np.meshgrid(xmin + t*(xmax-xmin), ymin + t*(ymax-ymin))
        probes.append(np.array([x.ravel(), y.ravel()]))
    return np.concatenate(probes, axis=1)


def _scene_values(geometry, points):
    x, y = points
    inside = ((x > geometry.x_span[0]) & (x < geometry.x_span[1])
              & (y > geometry.y_span[0]) & (y < geometry.y_span[1]))
    code = np.where(inside, 0, 1 if geometry.outer_boundary == 'pec' else 2)
    impedance = np.zeros(x.size, complex)
    for item in geometry.boundaries:
        mask = inside & item.shape.contains(x, y)
        code[mask] = {'pec': 1, 'pmc': 2, 'impedance': 3}[item.kind]
        impedance[mask] = 0 if item.impedance is None else item.impedance
    # Avoid evaluating a PML outside its physical bounding rectangle.
    clipped = np.array([np.clip(x, *geometry.x_span), np.clip(y, *geometry.y_span)])
    epsilon, mu = geometry.transformed_material_at(*clipped)
    return code, impedance, epsilon, mu


def detect_symmetry(solver, requested):
    """Find an axis-aligned Cnv candidate (1 <= n <= 8), cached per scene.

    This is sampled verification, not an exact CAD proof.  Explicit groups
    undergo the same verification; arbitrary group centres/orientations are
    deliberately not inferred.  The mirror reverses x about the box centre.
    """
    if requested is None:
        return Symmetry('C1', 1, np.eye(2), None)
    if not isinstance(requested, str) or not (requested == 'auto' or
            re.fullmatch(r'C[1-8]v', requested)):
        raise ConfigurationError("symmetry must be None, 'auto', or 'C1v' through 'C8v'.")
    key = (solver.geometry.revision, solver.frequency, requested)
    cached = getattr(solver, '_symmetry_cache', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    geometry = solver.geometry
    points = _scene_probes(geometry)
    center = np.array([np.mean(geometry.x_span), np.mean(geometry.y_span)])
    code, impedance, eps, mu = _scene_values(geometry, points)

    def invariant(matrix):
        other_code, other_impedance, other_eps, other_mu = _scene_values(
            geometry, _transform(points, matrix, center))
        if not np.array_equal(code, other_code) or not np.allclose(
                impedance, other_impedance, rtol=1e-10, atol=1e-12):
            return False
        active = code == 0
        r = np.eye(3)
        r[:2, :2] = matrix
        for original, other in ((eps, other_eps), (mu, other_mu)):
            # Compare full rotated tensors: diagonal anisotropy can forbid
            # rotations even when the material outlines have that symmetry.
            transformed = np.einsum('ij,jp,kj->ikp', r, original[:, active], r)
            diagonal = np.einsum('ip,ij->ijp', other[:, active], np.eye(3))
            if not np.allclose(transformed, diagonal, rtol=1e-9, atol=1e-11):
                return False
        return True

    found = Symmetry('C1', 1, np.eye(2), None)
    candidates = range(8, 0, -1) if requested == 'auto' else (int(requested[1]),)
    for n in candidates:
        rotation = _rotation(2*np.pi/n)
        mirrors = (np.diag([-1., 1.]), np.diag([1., -1.])) if requested == 'auto' else (np.diag([-1., 1.]),)
        for mirror in mirrors:
            if all(invariant(np.linalg.matrix_power(rotation, j)) and
                   invariant(mirror @ np.linalg.matrix_power(rotation, j)) for j in range(n)):
                found = Symmetry(f'C{n}v', n, rotation, mirror)
                break
        if found.mirror is not None:
            break
    if requested != 'auto' and found.name != requested:
        raise ConfigurationError(f'The continuous materials/boundaries do not support {requested} symmetry.')
    solver._symmetry_cache = (key, found)
    return found


def _probe_matrix(system, points):
    """Electric-field evaluation on common physical points, including edge signs."""
    transverse, scalar = system.basis.split_bases()
    matrices = []
    count = points.shape[1]
    for basis, indices, offset in ((transverse, system.transverse_indices, 0),
                                   (scalar, system.longitudinal_indices, 2*count)):
        probe = basis.probes(points*system.k0).tocoo()
        matrices.append(coo_matrix((probe.data, (probe.row+offset, indices[probe.col])),
                                  shape=(3*count, system.full_size)).tocsr())
    return (matrices[0]+matrices[1])[:, system.free_dofs]


def _sample_operators(solver, system, symmetry):
    from matplotlib.tri import Triangulation
    nodes, weights = np.polynomial.legendre.leggauss(16)
    spans = (solver.x_span, solver.y_span)
    axes = [(lo+hi)/2 + nodes*(hi-lo)/2 for lo, hi in spans]
    x, y = np.meshgrid(*axes)
    points = np.array([x.ravel(), y.ravel()])
    weights = np.outer(weights, weights).ravel()
    center = np.array([np.mean(span) for span in spans])
    transforms = {'identity': np.eye(2)}
    if symmetry.n > 1:
        transforms.update(rotation=symmetry.rotation, inverse=symmetry.rotation.T)
    if symmetry.mirror is not None:
        transforms['mirror'] = symmetry.mirror
    mesh = system.physical_mesh
    finder = Triangulation(*mesh.p, mesh.t.T).get_trifinder()
    valid = solver._physical_integration_mask(points)
    positions = {}
    for name, matrix in transforms.items():
        positions[name] = _transform(points, matrix.T, center)
        valid &= finder(*positions[name]) >= 0
    ids = np.flatnonzero(valid)
    if len(ids) < 8:
        raise ConfigurationError('Too few common interior points for mode identification.')
    root_weights = np.sqrt(weights[ids])
    operators = {}
    for name, matrix in transforms.items():
        probe = _probe_matrix(system, positions[name][:, ids])
        r = np.eye(3)
        r[:2, :2] = matrix

        def apply(vectors, probe=probe, r=r):
            values = (probe @ vectors).reshape(3, len(ids), -1)
            return (np.einsum('ij,jpk->ipk', r, values)*root_weights[None, :, None]).reshape(3*len(ids), -1)

        operators[name] = apply
    radius_squared = np.sum((points[:, ids]-center[:, None])**2, axis=0)
    radius_squared /= max(hi-lo for lo, hi in spans)**2
    def radial_moment(vectors):
        fields = operators['identity'](vectors).reshape(3, len(ids), -1)
        return (fields*radius_squared[None, :, None]).reshape(3*len(ids), -1)
    operators['radial_moment'] = radial_moment
    return ids, operators


def _clusters(values, tolerance):
    # Bounded-diameter clusters avoid chaining several distinct branches.
    remaining = set(range(len(values)))
    groups = []
    while remaining:
        first = min(remaining)
        group = [first]
        for j in sorted(remaining - {first}):
            if all(abs(values[j]-values[k]) <= tolerance*max(1., abs(values[j]), abs(values[k])) for k in group):
                group.append(j)
        remaining.difference_update(group)
        groups.append(group)
    return groups


def _normalized(samples):
    return samples / np.maximum(np.linalg.norm(samples, axis=0), np.finfo(float).tiny)


def _common_samples(ids, samples, reference):
    _, left, right = np.intersect1d(ids, reference['point_ids'], return_indices=True)
    if len(left) < 8:
        return None
    current = samples.reshape(3, len(ids), -1)[:, left].reshape(3*len(left), -1)
    previous = reference['samples'].reshape(3, len(reference['point_ids']), -1)[:, right].reshape(3*len(right), -1)
    return _normalized(current), _normalized(previous)


def _reference_cluster(ids, samples, reference, threshold):
    """Match whole subspaces before resolving individual partner vectors."""
    common = _common_samples(ids, samples, reference)
    if common is None:
        return None, None
    current, previous = common
    current, _ = np.linalg.qr(current)
    best, best_score = None, 0.
    for group in dict.fromkeys(reference['clusters']):
        indices = [j for j, value in enumerate(reference['clusters']) if value == group]
        if len(indices) != samples.shape[1]:
            continue
        old, _ = np.linalg.qr(previous[:, indices])
        singular = np.linalg.svd(current.conj().T @ old, compute_uv=False)
        score = float(min(1., singular[-1]**2))
        if score > best_score:
            best, best_score = indices, score
    return (best, best_score) if best_score >= threshold else (None, best_score)


def _longitudinal_sectors(solver, system, basis, neff, sample):
    """Recover pure TE/TM kernels from hybrid combinations when both exist.

    Both Ez and Hz are needed: maximizing Ez alone need not produce a pure
    TM vector under an approximate common-point overlap metric.
    """
    electric, magnetic = [], []
    e_norm, h_norm = 0., 0.
    for vector in basis.T:
        data = solver._field_data(system, system.expand(vector), neff)
        weights = system.basis.dx*solver._physical_integration_mask(data['coordinates'])
        fields = data['fields']
        electric.append((np.sqrt(weights)*fields[..., 2]).ravel())
        magnetic.append((np.sqrt(weights)*fields[..., 5]).ravel())
        e_norm += float(np.sum(weights[..., None]*abs(fields[..., :3])**2))
        h_norm += float(np.sum(weights[..., None]*abs(fields[..., 3:])**2))
    kernels = []
    for columns, norm in ((electric, e_norm), (magnetic, h_norm)):
        longitudinal = np.column_stack(columns)
        d, u = eigh(longitudinal.conj().T @ longitudinal)
        kernels.append(u[:, d <= 1e-13*max(norm, 1e-300)])
    if not all(kernel.shape[1] for kernel in kernels):
        return basis, [list(range(basis.shape[1]))]
    change = np.column_stack(kernels)
    if change.shape[1] != basis.shape[1] or np.linalg.svd(change, compute_uv=False)[-1] < 1e-6:
        return basis, [list(range(basis.shape[1]))]
    basis = basis @ change
    cut = kernels[0].shape[1]
    sectors = [list(range(cut)), list(range(cut, basis.shape[1]))]
    for sector in sectors:
        columns = np.array(sector)
        values = sample(basis[:, columns])
        d, u = eigh(values.conj().T @ values)
        basis[:, columns] = basis[:, columns] @ (u/np.sqrt(d))
    return basis, sectors


def identify_modes(solver, system, values, vectors, *, symmetry, tolerance,
                   residual_tolerance, divergence_tolerance, track, reference, directions,
                   overlap_threshold=.5, symmetry_tolerance=.25):
    """Return residual-checked vectors, identities and a compact mesh-free reference."""
    values = np.array(values, dtype=complex, copy=True)
    vectors = np.array(vectors, dtype=complex, copy=True)
    ids, operators = _sample_operators(solver, system, symmetry)
    samples = operators['identity'](vectors)
    info = [{} for _ in values]
    groups = []
    for spectral_cluster in _clusters(values, tolerance):
        # Counter-propagating or differently decaying branches can meet near
        # cutoff, but must not be mixed into standing/indeterminate fields.
        for direction in dict.fromkeys(directions[j] for j in spectral_cluster):
            groups.append([j for j in spectral_cluster if directions[j] == direction])
    for group_id, group in enumerate(groups):
        ix = np.array(group)
        original = vectors[:, ix]
        f = samples[:, ix]
        gram = f.conj().T @ f
        d, u = eigh(gram)
        if d[0] <= max(d[-1], 1e-300)*1e-10:
            raise ConfigurationError('The candidate modal subspace is rank deficient at the identification probes.')
        original_polarizations = [solver._polarization(solver._field_data(
            system, system.expand(original[:, j]), values[ix[j]])['fields'])
            for j in range(len(group))]
        # Do not contaminate already separated TE/TM invariant spaces through
        # quadrature error in the common-point overlap metric. If the input
        # contains hybrid combinations, use the whole subspace to resolve them.
        sectors = [list(range(len(group)))]
        if set(original_polarizations) == {'TE', 'TM'}:
            sectors = [[j for j, p in enumerate(original_polarizations) if p == kind]
                       for kind in ('TE', 'TM')]
        basis = np.zeros_like(original)
        for sector in sectors:
            indices = np.array(sector)
            d, u = eigh(gram[np.ix_(indices, indices)])
            basis[:, indices] = original[:, indices] @ (u / np.sqrt(d))
        if 'hybrid' in original_polarizations and len(group) > 1:
            basis, sectors = _longitudinal_sectors(
                solver, system, basis, np.mean(values[ix]), operators['identity'])
        q = operators['identity'](basis)
        blocks = [(sector, 0, 0) for sector in sectors]

        def diagonalize(action, blocks, kind):
            nonlocal basis, q
            next_blocks = []
            for members, angular, parity in blocks:
                indices = np.array(members)
                block = q[:, indices].conj().T @ action(basis[:, indices])
                eigenvalues, rotation = eigh((block+block.conj().T)/2)
                basis[:, indices] = basis[:, indices] @ rotation
                q[:, indices] = q[:, indices] @ rotation
                if kind == 'rotation':
                    targets = np.cos(2*np.pi*np.arange(symmetry.n//2+1)/symmetry.n)
                    labels = np.argmin(abs(eigenvalues[:, None]-targets), axis=1)
                elif kind == 'mirror':
                    labels = np.where(eigenvalues >= 0, 1, -1)
                else:
                    # Preserve resolved longitudinal/spatial descriptors when
                    # choosing a gauge for repeated copies of the same irrep.
                    labels = np.empty(len(indices), dtype=int)
                    for label, cluster in enumerate(_clusters(eigenvalues, 1e-7)):
                        labels[cluster] = label
                for label in sorted(set(labels), reverse=kind == 'mirror'):
                    selected = indices[labels == label].tolist()
                    next_blocks.append((selected, int(label) if kind == 'rotation' else angular,
                                        int(label) if kind == 'mirror' else parity))
            return next_blocks

        if symmetry.n > 1:
            blocks = diagonalize(lambda v: (operators['rotation'](v)+operators['inverse'](v))/2,
                                 blocks, 'rotation')
        if symmetry.mirror is not None:
            blocks = diagonalize(operators['mirror'], blocks, 'mirror')
        def longitudinal(v):
            z = operators['identity'](v).reshape(3, len(ids), -1)
            z[:2] = 0
            return z.reshape(3*len(ids), -1)
        blocks = diagonalize(longitudinal, blocks, 'longitudinal')
        blocks = diagonalize(operators['radial_moment'], blocks, 'spatial')

        labels = [None]*len(group)
        partners = [None]*len(group)
        canonical_order = []
        for members, angular, parity in sorted(blocks, key=lambda item: (item[1], -item[2])):
            # For Cnv, angular 0 (and n/2 when n is even) are 1D irreps;
            # other angular sectors are 2D E multiplets, with mirror partners.
            if symmetry.mirror is not None:
                doublet = angular != 0 and not (symmetry.n % 2 == 0 and angular == symmetry.n//2)
                label = f'{symmetry.name}:E{angular}' if doublet else f'{symmetry.name}:m{angular}:{parity:+d}'
                for member in members:
                    labels[member], partners[member] = label, parity
            canonical_order.extend(members)
        basis, q = basis[:, canonical_order], q[:, canonical_order]
        labels = [labels[j] for j in canonical_order]
        partners = [partners[j] for j in canonical_order]
        polarizations = [solver._polarization(solver._field_data(
            system, system.expand(basis[:, j]), values[ix[0]])['fields'])
            for j in range(len(group))]
        old_cluster, subspace_overlap = (None, None) if reference is None else _reference_cluster(
            ids, q, reference, overlap_threshold)

        # Resolve repeated copies of the same irrep using the previous subspace.
        # Procrustes alignment acts only within an identical mirror sector.
        if reference is not None and old_cluster is not None:
            common = _common_samples(ids, q, reference)
            if common is not None:
                current, previous = common
                for label, partner, polarization in dict.fromkeys(zip(labels, partners, polarizations)):
                    new = [j for j in range(len(group)) if (labels[j], partners[j], polarizations[j]) == (label, partner, polarization)]
                    old = [j for j in old_cluster if reference['labels'][j] == (label, partner)
                           and reference['polarizations'][j] == polarization]
                    if len(new) > 1 and len(new) == len(old):
                        cross = current[:, new].conj().T @ previous[:, old]
                        left, singular, right = np.linalg.svd(cross)
                        if singular[-1]**2 >= overlap_threshold:
                            alignment = left @ right
                            basis[:, new] = basis[:, new] @ alignment
                            q[:, new] = q[:, new] @ alignment

        # Associate projected partners with the original eigenpairs before
        # deciding whether rotations are accurate enough for returned fields.
        original_unit = _normalized(f)
        row, col = linear_sum_assignment(-abs(original_unit.conj().T @ _normalized(q))**2)
        assignment = np.empty(len(group), int)
        assignment[col] = row
        new_values = values[ix[assignment]].copy()
        # The best scalar neff near each assigned root follows a few Newton
        # steps minimizing ||Q(neff)v||; this does not average distinct roots.
        for j in range(len(group)):
            v = basis[:, j]
            for _ in range(3):
                r = system.polynomial(new_values[j]) @ v
                derivative = (system.A1 + 2*new_values[j]*system.A2) @ v
                denom = np.vdot(derivative, derivative).real
                if denom > np.finfo(float).tiny:
                    new_values[j] -= np.vdot(derivative, r)/denom
        residuals = [system.relative_residual(basis[:, j], z) for j, z in enumerate(new_values)]
        divergence = [system.divergence_residual(system.expand(basis[:, j]), z) for j, z in enumerate(new_values)]
        rotate = (np.isfinite(residuals).all() and np.isfinite(divergence).all()
                  and max(residuals) <= residual_tolerance and max(divergence) <= divergence_tolerance)
        if rotate:
            vectors[:, ix], values[ix], samples[:, ix] = basis, new_values, q
        else:
            vectors[:, ix] = original[:, assignment]
            values[ix] = values[ix[assignment]].copy()
            samples[:, ix] = f[:, assignment]
        span, _ = np.linalg.qr(q)
        closure = np.zeros(len(group))
        for name in ('rotation', 'mirror'):
            if name in operators:
                transformed = operators[name](basis)
                closure = np.maximum(closure, np.linalg.norm(
                    transformed-span @ (span.conj().T @ transformed), axis=0))
        for j, index in enumerate(group):
            symmetry_residual = float(closure[j])
            if symmetry.mirror is not None:
                reflected = operators['mirror'](basis[:, j:j+1])[:, 0]
                symmetry_residual = max(symmetry_residual, float(np.linalg.norm(reflected-partners[j]*q[:, j])))
            if symmetry.n > 1:
                cosine = (operators['rotation'](basis[:, j:j+1])[:, 0]
                          + operators['inverse'](basis[:, j:j+1])[:, 0])/2
                angular = float(np.vdot(q[:, j], cosine).real)
                targets = np.cos(2*np.pi*np.arange(symmetry.n//2+1)/symmetry.n)
                symmetry_residual = max(symmetry_residual, float(
                    np.linalg.norm(cosine-targets[np.argmin(abs(targets-angular))]*q[:, j])))
            # An incomplete multiplet cannot be assigned a reliable symmetry
            # partner merely by taking the sign of a small expectation value.
            resolved = symmetry.mirror is not None and symmetry_residual <= symmetry_tolerance
            # Label confidence measures actual returned fields, not only the
            # ideal symmetry-adapted identification vectors.
            confidence = float(abs(np.vdot(_normalized(samples[:, index:index+1])[:, 0],
                                           _normalized(q[:, j:j+1])[:, 0]))**2)
            info[index] = dict(degenerate_cluster=group_id, cluster_size=len(group),
                symmetry_irrep=labels[j] if resolved else None,
                symmetry_partner=partners[j] if resolved else None,
                symmetry_resolved=resolved, symmetry_subspace_residual=symmetry_residual,
                subspace_overlap=subspace_overlap,
                symmetry_projection='applied' if rotate and len(group)>1 else
                                    'residual_rejected' if not rotate and len(group)>1 else 'not_needed',
                symmetry_overlap=min(1., confidence))

    samples = _normalized(samples)
    mode_ids = [None]*len(values)
    overlaps = [None]*len(values)
    previous_columns = {}
    if track and reference is not None:
        common = _common_samples(ids, samples, reference)
        if common is not None:
            current, previous = common
            score = abs(previous.conj().T @ current)**2
            for old, label in enumerate(reference['labels']):
                for new in range(len(values)):
                    if label != (info[new]['symmetry_irrep'], info[new]['symmetry_partner']):
                        score[old, new] = 0
            old_indices, new_indices = linear_sum_assignment(-score)
            for old, new in zip(old_indices, new_indices):
                if score[old, new] >= overlap_threshold:
                    mode_ids[new] = reference['mode_ids'][old]
                    overlaps[new] = float(score[old, new])
                    previous_columns[new] = (current[:, new], previous[:, old])
    next_id = max(reference['mode_ids'], default=-1)+1 if track and reference is not None else 0
    for j in range(len(values)):
        if mode_ids[j] is None:
            mode_ids[j], next_id = next_id, next_id+1
        if j in previous_columns:
            current, previous = previous_columns[j]
            phase = np.exp(-1j*np.angle(np.vdot(previous, current)))
        else:
            phase = np.exp(-1j*np.angle(samples[np.argmax(abs(samples[:, j])), j]))
        vectors[:, j] *= phase
        samples[:, j] *= phase
        info[j].update(mode_id=mode_ids[j], tracking_overlap=overlaps[j])
    order = np.argsort(mode_ids, kind='stable')
    info = [info[j] for j in order]
    reference = dict(point_ids=ids, samples=samples[:, order],
                     labels=[(item['symmetry_irrep'], item['symmetry_partner']) for item in info],
                     polarizations=[solver._polarization(solver._field_data(
                         system, system.expand(vectors[:, j]), values[j])['fields']) for j in order],
                     clusters=[item['degenerate_cluster'] for item in info],
                     mode_ids=[mode_ids[j] for j in order])
    return values[order], vectors[:, order], info, reference
