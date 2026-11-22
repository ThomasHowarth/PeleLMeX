#ifndef PELELM_MLMGUTILS_H
#define PELELM_MLMGUTILS_H

#include <AMReX_MLMG.H>
#include <string>

// MLMG options shared by the diffusion and the plasma preconditioner solves
namespace MLMGUtils {
// Map an input bottom_solver name onto MLMG's BottomSolver; aborts on an
// unknown name so that typos are not silently ignored.
inline amrex::MLMG::BottomSolver
parseBottomSolver(const std::string& name, const std::string& prefix)
{
  using BS = amrex::MLMG::BottomSolver;
  if (name == "smoother") {
    return BS::smoother;
  }
  if (name == "bicgstab" || name == "bicg") {
    return BS::bicgstab;
  }
  if (name == "cg") {
    return BS::cg;
  }
  if (name == "bicgcg") {
    return BS::bicgcg;
  }
  if (name == "cgbicg") {
    return BS::cgbicg;
  }
  if (name == "hypre") {
#ifdef AMREX_USE_HYPRE
    return BS::hypre;
#else
    amrex::Abort(
      prefix + ".bottom_solver = hypre requires PeleLMeX built with hypre "
               "(GNU make USE_HYPRE = TRUE, CMake PELE_ENABLE_HYPRE = ON)");
#endif
  }
  if (name == "petsc") {
#ifdef AMREX_USE_PETSC
    return BS::petsc;
#else
    amrex::Abort(
      prefix + ".bottom_solver = petsc requires PeleLMeX built with PETSc "
               "(GNU make USE_PETSC = TRUE)");
#endif
  }
  if (name == "algmg") {
    return BS::algmg;
  }
  amrex::Abort(
    prefix + ".bottom_solver: unknown value '" + name +
    "' (smoother, bicgstab, cg, bicgcg, cgbicg, hypre, petsc, algmg)");
  return BS::Default; // not reached; amrex::Abort is not [[noreturn]]
}

// Bottom solvers that assemble a matrix: single-component systems only
inline bool
isMatrixBottomSolver(amrex::MLMG::BottomSolver bs)
{
  using BS = amrex::MLMG::BottomSolver;
  return bs == BS::hypre || bs == BS::petsc || bs == BS::algmg;
}

// Apply the user-selected bottom solver; an empty name keeps the default
inline void
setBottomSolver(
  amrex::MLMG& mlmg,
  const std::string& name,
  const std::string& prefix,
  const std::string& hypre_namespace,
  const amrex::Real bottom_rtol)
{
  if (name.empty()) {
    return;
  }
  mlmg.setBottomSolver(parseBottomSolver(name, prefix));
#ifdef AMREX_USE_HYPRE
  if (mlmg.getBottomSolver() == amrex::MLMG::BottomSolver::hypre) {
    mlmg.setHypreOptionsNamespace(hypre_namespace);
  }
#else
  amrex::ignore_unused(hypre_namespace);
#endif
  if (bottom_rtol > 0.0) {
    mlmg.setBottomTolerance(bottom_rtol);
  }
}
} // namespace MLMGUtils

#endif
