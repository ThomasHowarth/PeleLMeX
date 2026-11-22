#ifndef PRECOND_OP_H
#define PRECOND_OP_H

#include <AMReX_MLMG.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLABecCecLaplacian.H>

class PeleLM;

class PrecondOp
{
public:
  PrecondOp(PeleLM* a_pelelm);

  // Data and parameters
  PeleLM* m_pelelm;

  void setDiffOpScalars(
    const amrex::Real& aScal,
    const amrex::Real& bScal,
    const amrex::Real& cScal);
  void setDriftOpScalars(const amrex::Real& aScal, const amrex::Real& bScal);
  void setStildaOpScalars(const amrex::Real& aScal, const amrex::Real& bScal);

  void setDiffOpBCs(const amrex::BCRec& a_bcrec);
  void setDriftOpBCs(const amrex::BCRec& a_bcrec);
  void setStildaOpBCs(const amrex::BCRec& a_bcrec);

  void setDiffOpLevelBC(int lev, amrex::MultiFab* a_MF);
  void setDriftOpLevelBC(int lev, amrex::MultiFab* a_MF);
  void setStildaOpLevelBC(int lev, amrex::MultiFab* a_MF);

  void setDiffOpRelaxation(const amrex::Real& a_omega);
  void setDiffOpACoeff(int lev, const amrex::Real& acoeff);
  void setDiffOpBCoeff(
    int lev,
    const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bCoeff);
  void setDiffOpCCoeff(
    int lev,
    const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_cCoeff);

  void getDiffOpDiagonal(int lev, amrex::MultiFab& a_diffOpDiag);

  void setDriftOpBCoeff(
    int lev,
    const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bCoeff);
  void setStildaOpBCoeff(
    int lev,
    const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bCoeff);

  void diffOpSolve(
    const amrex::Vector<amrex::MultiFab*>& a_sol,
    const amrex::Vector<const amrex::MultiFab*>& a_rhs);

  void StildaOpSolve(
    const amrex::Vector<amrex::MultiFab*>& a_sol,
    const amrex::Vector<const amrex::MultiFab*>& a_rhs);

  void driftOpApply(
    const amrex::Vector<amrex::MultiFab*>& a_Ax,
    const amrex::Vector<amrex::MultiFab*>& a_x);

  void readParameters();

  amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>
  getOpBC(amrex::Orientation::Side a_side, const amrex::BCRec& a_bc);

  // PC linear ops
  std::unique_ptr<amrex::MLABecCecLaplacian> m_diff;
  std::unique_ptr<amrex::MLABecLaplacian> m_drift;
  std::unique_ptr<amrex::MLABecLaplacian> m_Stilda;

  // Associated MLMGs
  std::unique_ptr<amrex::MLMG> m_mlmg_diff;
  std::unique_ptr<amrex::MLMG> m_mlmg_drift;
  std::unique_ptr<amrex::MLMG> m_mlmg_Stilda;

  // MLMG options (ef.precond.*). The defaults are MLMG's.
  int m_diff_verbose = 0;
  int m_drift_verbose = 0;
  int m_Stilda_verbose = 0;
  int m_bottom_verbose = 0;
  int m_fixed_mg_it = -1;
  int m_mg_max_iter = 200;
  int m_mg_bottom_maxiter = 200;
  int m_mg_max_coarsening_level_diff = 100;
  int m_mg_max_coarsening_level_Stilda = 100;
  int m_mg_maxorder = 2;
  int m_num_pre_smooth = 2;
  int m_num_post_smooth = 2;

  amrex::Real m_mg_rtol = 1.0e-11;
  amrex::Real m_mg_atol = 1.0e-14;
  amrex::Real m_mg_bottom_rtol = -1.0;

  // The electron diffusion-drift operator has no matrix: iterative bottom
  // solvers only. The Stilda (Poisson-like) solve can use any of them.
  std::string m_diff_bottom_solver;
  std::string m_Stilda_bottom_solver;
  std::string m_hypre_namespace = "ef_precond_hypre";
  amrex::MultigridType m_Stilda_mg_type = amrex::MultigridType::geometric;

private:
  void setMLMGOptions(
    amrex::MLMG& a_mlmg,
    int a_verbose,
    const std::string& a_bottom_solver,
    const std::string& a_prefix) const;
};
#endif
