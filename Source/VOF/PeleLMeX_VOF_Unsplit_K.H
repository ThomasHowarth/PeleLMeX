#ifndef PELELM_VOF_UNSPLIT_K_H
#define PELELM_VOF_UNSPLIT_K_H

// Unsplit geometric flux kernels for the PLIC volume of fluid, after Owkes &
// Desjardins (JCP 270, 2014).
//
// The liquid crossing a face in a step is the liquid inside the flux volume,
// the region bounded by the face and its corners traced back along the flow.
// Everything here works in cell units: cell (i,j,k) is the unit cube
// [i,i+1] x [j,j+1] x [k,k+1], and face velocities are Courant numbers.  In 2D
// the mesh is one cell thick in z, so that one kernel serves both dimensions.
//
// Differences from the paper, none of which change the method:
// - The flux volume is the signed sum of the tetrahedra joining a corner of the
//   face to the triangles of its boundary, rather than the tabulated partition
//   of its Fig. 9.  The two are equivalent: any closed oriented triangulated
//   surface gives the right signed integral, including crossed flux volumes.
// - The side faces are shared by the flux volumes of neighboring faces, so they
//   must be triangulated identically by each of them.  The diagonal joins the
//   traced lower node of the mesh edge to its upper node, which depends only
//   on the mesh edge.  This is what makes the fluxes tile without overlaps.
// - Simplices are cut using an explicit stack, for GPU use, and the normals
//   are those of pelelmex::vof::fit_plane rather than ELVIRA, as the paper
//   allows.

#include <PeleLMeX_VOF_K.H>

namespace pelelmex::vof::unsplit {

using namespace amrex::literals;

struct Vec3
{
  amrex::Real x[3];
};

struct Tet
{
  Vec3 v[4];
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Vec3
lerp(const Vec3& a, const Vec3& b, const amrex::Real t) noexcept
{
  return Vec3{
    {a.x[0] + t * (b.x[0] - a.x[0]), a.x[1] + t * (b.x[1] - a.x[1]),
     a.x[2] + t * (b.x[2] - a.x[2])}};
}

/**
 * \brief Signed volume of the tetrahedron with apex o over the triangle
 *        (a,b,c): positive when the triangle's right-handed normal points away
 *        from o.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
signed_volume(
  const Vec3& o, const Vec3& a, const Vec3& b, const Vec3& c) noexcept
{
  const amrex::Real ax = a.x[0] - o.x[0], ay = a.x[1] - o.x[1],
                    az = a.x[2] - o.x[2];
  const amrex::Real bx = b.x[0] - o.x[0], by = b.x[1] - o.x[1],
                    bz = b.x[2] - o.x[2];
  const amrex::Real cx = c.x[0] - o.x[0], cy = c.x[1] - o.x[1],
                    cz = c.x[2] - o.x[2];
  return (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) +
          az * (bx * cy - by * cx)) /
         6.0_rt;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
abs_volume(const Tet& t) noexcept
{
  return std::abs(signed_volume(t.v[0], t.v[1], t.v[2], t.v[3]));
}

/** \brief Tetrahedra filling the prism with triangles p, q and edges p-q */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
prism_to_tets(
  const Vec3& p0,
  const Vec3& p1,
  const Vec3& p2,
  const Vec3& q0,
  const Vec3& q1,
  const Vec3& q2,
  Tet* out) noexcept
{
  out[0] = Tet{{p0, p1, p2, q2}};
  out[1] = Tet{{p0, p1, q2, q1}};
  out[2] = Tet{{p0, q0, q1, q2}};
}

/**
 * \brief Split a tetrahedron by the plane where the vertex values d vanish.
 *
 * Pieces with d <= 0 go to neg and those with d > 0 to pos.
 *
 * \param n_neg,n_pos number of pieces written, at most 3 each
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
split_tet(
  const Tet& t,
  const amrex::Real* d,
  Tet* neg,
  int& n_neg,
  Tet* pos,
  int& n_pos) noexcept
{
  int in[4];
  int out[4];
  int ni = 0;
  int no = 0;
  for (int v = 0; v < 4; ++v) {
    if (d[v] <= 0.0_rt) {
      in[ni++] = v;
    } else {
      out[no++] = v;
    }
  }
  n_neg = 0;
  n_pos = 0;
  if (no == 0) {
    neg[0] = t;
    n_neg = 1;
    return;
  }
  if (ni == 0) {
    pos[0] = t;
    n_pos = 1;
    return;
  }

  // Point where the edge joining an inside and an outside vertex crosses
  auto cross = [&](const int a, const int b) {
    return lerp(t.v[a], t.v[b], d[a] / (d[a] - d[b]));
  };
  if (ni == 1) {
    const int a = in[0];
    const Vec3 p0 = cross(a, out[0]);
    const Vec3 p1 = cross(a, out[1]);
    const Vec3 p2 = cross(a, out[2]);
    neg[0] = Tet{{t.v[a], p0, p1, p2}};
    n_neg = 1;
    prism_to_tets(t.v[out[0]], t.v[out[1]], t.v[out[2]], p0, p1, p2, pos);
    n_pos = 3;
  } else if (no == 1) {
    const int a = out[0];
    const Vec3 p0 = cross(in[0], a);
    const Vec3 p1 = cross(in[1], a);
    const Vec3 p2 = cross(in[2], a);
    pos[0] = Tet{{t.v[a], p0, p1, p2}};
    n_pos = 1;
    prism_to_tets(t.v[in[0]], t.v[in[1]], t.v[in[2]], p0, p1, p2, neg);
    n_neg = 3;
  } else {
    const int a = in[0];
    const int b = in[1];
    const int c = out[0];
    const int e = out[1];
    const Vec3 ac = cross(a, c);
    const Vec3 ae = cross(a, e);
    const Vec3 bc = cross(b, c);
    const Vec3 be = cross(b, e);
    prism_to_tets(t.v[a], ac, ae, t.v[b], bc, be, neg);
    n_neg = 3;
    prism_to_tets(t.v[c], ac, bc, t.v[e], ae, be, pos);
    n_pos = 3;
  }
}

/**
 * \brief Volume of the part of a tetrahedron on the liquid side of the plane
 *        m.(x - x_lo) <= alpha of the cell with low corner x_lo.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
tet_liquid_volume(
  const Tet& t,
  const int i,
  const int j,
  const int k,
  const amrex::Real* plane) noexcept
{
  amrex::Real d[4];
  for (int v = 0; v < 4; ++v) {
    d[v] = plane[0] * (t.v[v].x[0] - i) + plane[1] * (t.v[v].x[1] - j) +
           plane[2] * (t.v[v].x[2] - k) - plane[3];
  }
  Tet neg[3];
  Tet pos[3];
  int n_neg = 0;
  int n_pos = 0;
  split_tet(t, d, neg, n_neg, pos, n_pos);
  amrex::Real vol = 0.0_rt;
  for (int n = 0; n < n_neg; ++n) {
    vol += abs_volume(neg[n]);
  }
  return vol;
}

// Tetrahedra held while cutting by the mesh planes.  Each cut replaces one
// tetrahedron by at most 6 and a depth-first path crosses few planes for a
// Courant number below 1: 64 is well above what can be reached.
static constexpr int cut_stack_size = 64;

/**
 * \brief Unsigned liquid volume of a tetrahedron, cut along the mesh planes
 *        into one piece per cell and then by the plane of each cell.
 *
 * \param F     volume fraction
 * \param plane (mx, my, mz, alpha) of each cell, in cell coordinates
 * \param bx    cells where F and plane are valid; pieces beyond it are
 *              charged to the nearest cell inside it
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
tet_liquid_in_mesh(
  const Tet& tet,
  amrex::Array4<amrex::Real const> const& F,
  amrex::Array4<amrex::Real const> const& plane,
  const amrex::Box& bx) noexcept
{
  constexpr amrex::Real eps = 1.0e-12_rt;
  Tet stack[cut_stack_size];
  int top = 0;
  stack[top++] = tet;
  amrex::Real liquid = 0.0_rt;
  const auto lo = amrex::lbound(bx);
  const auto hi = amrex::ubound(bx);

  while (top > 0) {
    const Tet t = stack[--top];

    // The first mesh plane crossing this tetrahedron, if any.  The z planes
    // play no part in 2D, whose mesh is one cell thick.
    int cut_dir = -1;
    amrex::Real cut_at = 0.0_rt;
    for (int d = 0; d < AMREX_SPACEDIM && cut_dir < 0; ++d) {
      amrex::Real tmin = t.v[0].x[d];
      amrex::Real tmax = t.v[0].x[d];
      for (int v = 1; v < 4; ++v) {
        tmin = amrex::min(tmin, t.v[v].x[d]);
        tmax = amrex::max(tmax, t.v[v].x[d]);
      }
      const amrex::Real p = std::floor(tmin + eps) + 1.0_rt;
      if (tmax > p + eps) {
        cut_dir = d;
        cut_at = p;
      }
    }

    if (cut_dir >= 0 && top + 6 <= cut_stack_size) {
      const amrex::Real d[4] = {
        t.v[0].x[cut_dir] - cut_at, t.v[1].x[cut_dir] - cut_at,
        t.v[2].x[cut_dir] - cut_at, t.v[3].x[cut_dir] - cut_at};
      Tet neg[3];
      Tet pos[3];
      int n_neg = 0;
      int n_pos = 0;
      split_tet(t, d, neg, n_neg, pos, n_pos);
      for (int n = 0; n < n_neg; ++n) {
        stack[top++] = neg[n];
      }
      for (int n = 0; n < n_pos; ++n) {
        stack[top++] = pos[n];
      }
      continue;
    }

    // Within a single cell, found from the centroid
    const amrex::Real vol = abs_volume(t);
    if (vol <= 0.0_rt) {
      continue;
    }
    int c[3] = {0, 0, 0};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      c[d] = static_cast<int>(std::floor(
        0.25_rt * (t.v[0].x[d] + t.v[1].x[d] + t.v[2].x[d] + t.v[3].x[d])));
    }
    const int ci = amrex::Clamp(c[0], lo.x, hi.x);
    const int cj = amrex::Clamp(c[1], lo.y, hi.y);
    const int ck = amrex::Clamp(c[2], lo.z, hi.z);
    const amrex::Real f = F(ci, cj, ck);
    if (f <= vof_tol) {
      continue;
    }
    if (f >= 1.0_rt - vof_tol) {
      liquid += vol;
      continue;
    }
    const amrex::Real pl[4] = {
      plane(ci, cj, ck, 0), plane(ci, cj, ck, 1), plane(ci, cj, ck, 2),
      plane(ci, cj, ck, 3)};
    liquid += tet_liquid_volume(t, ci, cj, ck, pl);
  }
  return liquid;
}

/**
 * \brief Apex o of the two tetrahedra (o,a,c,e) and (o,a,b,c) added on a
 *        traced face (a,b,c,e) to give them a total signed volume vcor.  o lies
 *        over the barycenter of the face; the volume is linear in o along dir.
 *
 * \return false if the face has no extent normal to dir (no correction)
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE bool
correction_apex(
  const Vec3& a,
  const Vec3& b,
  const Vec3& c,
  const Vec3& e,
  const int dir,
  const amrex::Real vcor,
  Vec3& o) noexcept
{
  const Vec3* tri[2][3] = {{&a, &c, &e}, {&a, &b, &c}}; // +dir normals
  for (int d = 0; d < 3; ++d) {
    o.x[d] = 0.25_rt * (a.x[d] + b.x[d] + c.x[d] + e.x[d]);
  }
  amrex::Real s0 = 0.0_rt; // 6 V at o_dir = 0
  amrex::Real sd = 0.0_rt; // d(6 V)/d(o_dir)
  for (auto& n : tri) {
    const Vec3 &p = *n[0], &q = *n[1], &r = *n[2];
    const amrex::Real u[3] = {
      q.x[0] - p.x[0], q.x[1] - p.x[1], q.x[2] - p.x[2]};
    const amrex::Real w[3] = {
      r.x[0] - p.x[0], r.x[1] - p.x[1], r.x[2] - p.x[2]};
    const amrex::Real nrm[3] = {
      u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
      u[0] * w[1] - u[1] * w[0]};
    // V = (p - o).N / 6
    for (int d = 0; d < 3; ++d) {
      const amrex::Real od = (d == dir) ? 0.0_rt : o.x[d];
      s0 += (p.x[d] - od) * nrm[d];
    }
    sd -= nrm[dir];
  }
  if (std::abs(sd) <= 1.0e-10_rt) {
    return false;
  }
  o.x[dir] = (6.0_rt * vcor - s0) / sd;
  return true;
}

/**
 * \brief Signed liquid volume, in cell volumes, crossing the low face of cell
 *        (i,j,k) in direction dir, positive along +dir.
 *
 * \param back    traced position of each mesh node at the start of the step,
 *                in cell units (3 components)
 * \param courant the face velocity times dt/dx, which the flux volume is
 *                corrected to match
 * \param bx      cells where F and plane are valid
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
face_liquid_flux(
  const int i,
  const int j,
  const int k,
  const int dir,
  const amrex::Real courant,
  amrex::Array4<amrex::Real const> const& F,
  amrex::Array4<amrex::Real const> const& plane,
  amrex::Array4<amrex::Real const> const& back,
  const amrex::Box& bx) noexcept
{
  // Right-handed tangential directions.  The face spans 0..1 along each, and
  // in 2D the node arrays have no extent in z: the k+1 nodes are the k nodes
  // shifted by one cell.
  const int t1 = (dir + 1) % 3;
  const int t2 = (dir + 2) % 3;

  int off1[3] = {0, 0, 0};
  int off2[3] = {0, 0, 0};
  off1[t1] = 1;
  off2[t2] = 1;

  auto node = [&](const int s1, const int s2) {
    return Vec3{
      {static_cast<amrex::Real>(i + s1 * off1[0] + s2 * off2[0]),
       static_cast<amrex::Real>(j + s1 * off1[1] + s2 * off2[1]),
       static_cast<amrex::Real>(k + s1 * off1[2] + s2 * off2[2])}};
  };
  auto traced = [&](const int s1, const int s2) {
#if AMREX_SPACEDIM == 2
    // Nodes of a 2D mesh: z stays as it was, and only the in-plane offset
    // selects the node
    const int ni = i + s1 * off1[0] + s2 * off2[0];
    const int nj = j + s1 * off1[1] + s2 * off2[1];
    const amrex::Real z = static_cast<amrex::Real>(k + s1 * off1[2] + s2 * off2[2]);
    return Vec3{{back(ni, nj, 0, 0), back(ni, nj, 0, 1), z}};
#else
    const int ni = i + s1 * off1[0] + s2 * off2[0];
    const int nj = j + s1 * off1[1] + s2 * off2[1];
    const int nk = k + s1 * off1[2] + s2 * off2[2];
    return Vec3{{back(ni, nj, nk, 0), back(ni, nj, nk, 1), back(ni, nj, nk, 2)}};
#endif
  };

  // Face corners counterclockwise about +dir, and their traced positions
  const Vec3 a = node(0, 0);
  const Vec3 b = node(1, 0);
  const Vec3 c = node(1, 1);
  const Vec3 e = node(0, 1);
  const Vec3 ap = traced(0, 0);
  const Vec3 bp = traced(1, 0);
  const Vec3 cp = traced(1, 1);
  const Vec3 ep = traced(0, 1);

  // Fan of the boundary from a.  With outward triangles the signed volumes
  // add up to the flux volume, positive for a flow along +dir.  Triangles
  // touching a vanish: the face itself and the side faces through a.  The
  // diagonals of the side faces depend only on the mesh edge (see above).
  constexpr int ntri = 8;
  const Vec3* tri[ntri][3] = {
    {&ap, &bp, &b},  // side a-b, diagonal a' - b
    {&e, &ep, &ap},  // side e-a, diagonal e - a'
    {&b, &bp, &c},   // side b-c, diagonal b' - c
    {&bp, &cp, &c},  //
    {&c, &cp, &ep},  // side c-e, diagonal e' - c
    {&c, &ep, &e},   //
    {&ap, &ep, &cp}, // traced face, diagonal a' - c'
    {&ap, &cp, &bp}};

  amrex::Real vol = 0.0_rt;
  for (int n = 0; n < ntri; ++n) {
    vol += signed_volume(a, *tri[n][0], *tri[n][1], *tri[n][2]);
  }

  // Extra volume so that the flux volume matches the face velocity exactly:
  // two tetrahedra from the traced face to a point o above its barycenter, in
  // the direction of dir (Eq. 27-28).  Their volume is linear in o along dir.
  const Vec3* ttri[2][3] = {{&ap, &cp, &ep}, {&ap, &bp, &cp}}; // +dir normals
  Vec3 o;
  const bool has_cor = correction_apex(ap, bp, cp, ep, dir, courant - vol, o);

  // Signed liquid volume: each tetrahedron carries the sign of its volume,
  // and its pieces are unsigned (Sec. 3.2.3).  A flux volume entirely within
  // pure cells needs no cutting.
  amrex::Real pmin[3];
  amrex::Real pmax[3];
  for (int d = 0; d < 3; ++d) {
    pmin[d] = amrex::min(
      amrex::min(a.x[d], b.x[d], c.x[d], e.x[d]),
      amrex::min(ap.x[d], bp.x[d], cp.x[d], ep.x[d]));
    pmax[d] = amrex::max(
      amrex::max(a.x[d], b.x[d], c.x[d], e.x[d]),
      amrex::max(ap.x[d], bp.x[d], cp.x[d], ep.x[d]));
    if (has_cor) {
      pmin[d] = amrex::min(pmin[d], o.x[d]);
      pmax[d] = amrex::max(pmax[d], o.x[d]);
    }
  }
  const auto blo = amrex::lbound(bx);
  const auto bhi = amrex::ubound(bx);
  const int ilo = amrex::Clamp(static_cast<int>(std::floor(pmin[0])), blo.x, bhi.x);
  const int ihi = amrex::Clamp(static_cast<int>(std::floor(pmax[0])), blo.x, bhi.x);
  const int jlo = amrex::Clamp(static_cast<int>(std::floor(pmin[1])), blo.y, bhi.y);
  const int jhi = amrex::Clamp(static_cast<int>(std::floor(pmax[1])), blo.y, bhi.y);
#if AMREX_SPACEDIM == 2
  const int klo = 0;
  const int khi = 0;
#else
  const int klo = amrex::Clamp(static_cast<int>(std::floor(pmin[2])), blo.z, bhi.z);
  const int khi = amrex::Clamp(static_cast<int>(std::floor(pmax[2])), blo.z, bhi.z);
#endif
  bool all_gas = true;
  bool all_liq = true;
  for (int kk = klo; kk <= khi; ++kk) {
    for (int jj = jlo; jj <= jhi; ++jj) {
      for (int ii = ilo; ii <= ihi; ++ii) {
        const amrex::Real f = F(ii, jj, kk);
        all_gas = all_gas && (f <= vof_tol);
        all_liq = all_liq && (f >= 1.0_rt - vof_tol);
      }
    }
  }
  if (all_gas) {
    return 0.0_rt;
  }
  if (all_liq) {
    return courant;
  }

  amrex::Real liquid = 0.0_rt;
  for (int n = 0; n < ntri; ++n) {
    const Tet t{{a, *tri[n][0], *tri[n][1], *tri[n][2]}};
    const amrex::Real v = signed_volume(t.v[0], t.v[1], t.v[2], t.v[3]);
    if (v != 0.0_rt) {
      const amrex::Real l = tet_liquid_in_mesh(t, F, plane, bx);
      liquid += (v > 0.0_rt) ? l : -l;
    }
  }
  if (has_cor) {
    for (auto& n : ttri) {
      const Tet t{{o, *n[0], *n[1], *n[2]}};
      const amrex::Real v = signed_volume(t.v[0], t.v[1], t.v[2], t.v[3]);
      if (v != 0.0_rt) {
        const amrex::Real l = tet_liquid_in_mesh(t, F, plane, bx);
        liquid += (v > 0.0_rt) ? l : -l;
      }
    }
  }
  return liquid;
}

/**
 * \brief Velocity at a position, in Courant numbers, by multilinear
 *        interpolation of the node velocities.  Positions beyond the node box
 *        use its nearest nodes.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
interp_node_velocity(
  const amrex::Real* x,
  amrex::Array4<amrex::Real const> const& vel,
  const amrex::Box& nbx,
  amrex::Real* u) noexcept
{
  const auto lo = amrex::lbound(nbx);
  const auto hi = amrex::ubound(nbx);
  const int nlo[3] = {lo.x, lo.y, lo.z};
  const int nhi[3] = {hi.x, hi.y, hi.z};
  int base[3] = {0, 0, 0};
  amrex::Real w[3] = {0.0_rt, 0.0_rt, 0.0_rt};
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    const int b = static_cast<int>(std::floor(x[d]));
    base[d] = amrex::Clamp(b, nlo[d], amrex::max(nlo[d], nhi[d] - 1));
    w[d] = amrex::Clamp(x[d] - base[d], 0.0_rt, 1.0_rt);
  }
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    u[d] = 0.0_rt;
  }
#if AMREX_SPACEDIM == 2
  for (int s1 = 0; s1 <= 1; ++s1) {
    for (int s0 = 0; s0 <= 1; ++s0) {
      const amrex::Real wt =
        (s0 != 0 ? w[0] : 1.0_rt - w[0]) * (s1 != 0 ? w[1] : 1.0_rt - w[1]);
      for (int d = 0; d < 2; ++d) {
        u[d] += wt * vel(base[0] + s0, base[1] + s1, 0, d);
      }
    }
  }
#else
  for (int s2 = 0; s2 <= 1; ++s2) {
    for (int s1 = 0; s1 <= 1; ++s1) {
      for (int s0 = 0; s0 <= 1; ++s0) {
        const amrex::Real wt = (s0 != 0 ? w[0] : 1.0_rt - w[0]) *
                               (s1 != 0 ? w[1] : 1.0_rt - w[1]) *
                               (s2 != 0 ? w[2] : 1.0_rt - w[2]);
        for (int d = 0; d < 3; ++d) {
          u[d] += wt * vel(base[0] + s0, base[1] + s1, base[2] + s2, d);
        }
      }
    }
  }
#endif
}

} // namespace pelelmex::vof::unsplit

#endif
