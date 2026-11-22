#include <PeleLMeX.H>
#include <PeleLMeX_MLMGUtils.H>
#include <PrecondOp.H>
#include <AMReX_ParmParse.H>

//---------------------------------------------------------------------------------------
// Preconditioner Operator
PrecondOp::PrecondOp(PeleLM* a_pelelm) : m_pelelm(a_pelelm)
{
  readParameters();

  // Solve LPInfo
  amrex::LPInfo info_diff;
  info_diff.setMaxCoarseningLevel(m_mg_max_coarsening_level_diff);
  amrex::LPInfo info_Stilda;
  info_Stilda.setMaxCoarseningLevel(m_mg_max_coarsening_level_Stilda);

  // Apply LPInfo (no coarsening)
  amrex::LPInfo info_apply;
  info_apply.setMaxCoarseningLevel(0);

  // Diff/Drift Op
  m_diff.reset(new amrex::MLABecCecLaplacian(
    m_pelelm->Geom(0, m_pelelm->finestLevel()),
    m_pelelm->boxArray(0, m_pelelm->finestLevel()),
    m_pelelm->DistributionMap(0, m_pelelm->finestLevel()), info_diff));
  m_diff->setMaxOrder(m_mg_maxorder);

  // Drift Op
  m_drift.reset(new amrex::MLABecLaplacian(
    m_pelelm->Geom(0, m_pelelm->finestLevel()),
    m_pelelm->boxArray(0, m_pelelm->finestLevel()),
    m_pelelm->DistributionMap(0, m_pelelm->finestLevel()), info_apply));
  m_drift->setMaxOrder(m_mg_maxorder);

  // Stilda Op
  m_Stilda.reset(new amrex::MLABecLaplacian(
    m_pelelm->Geom(0, m_pelelm->finestLevel()),
    m_pelelm->boxArray(0, m_pelelm->finestLevel()),
    m_pelelm->DistributionMap(0, m_pelelm->finestLevel()), info_Stilda));
  m_Stilda->setMaxOrder(m_mg_maxorder);

  // MLMGs
  m_mlmg_diff = std::make_unique<amrex::MLMG>(*m_diff);
  setMLMGOptions(
    *m_mlmg_diff, m_diff_verbose, m_diff_bottom_solver,
    "ef.precond.diff_bottom_solver");
  // The smoothing options are the electron solve's only, as before
  m_mlmg_diff->setPreSmooth(m_num_pre_smooth);
  m_mlmg_diff->setPostSmooth(m_num_post_smooth);
  m_mlmg_drift = std::make_unique<amrex::MLMG>(*m_drift);
  m_mlmg_drift->setVerbose(m_drift_verbose);
  m_mlmg_Stilda = std::make_unique<amrex::MLMG>(*m_Stilda);
  setMLMGOptions(
    *m_mlmg_Stilda, m_Stilda_verbose, m_Stilda_bottom_solver,
    "ef.precond.Stilda_bottom_solver");
  m_mlmg_Stilda->setMultigridType(m_Stilda_mg_type);
}

void
PrecondOp::setMLMGOptions(
  amrex::MLMG& a_mlmg,
  int a_verbose,
  const std::string& a_bottom_solver,
  const std::string& a_prefix) const
{
  a_mlmg.setVerbose(a_verbose);
  a_mlmg.setBottomVerbose(m_bottom_verbose);
  a_mlmg.setMaxIter(m_mg_max_iter);
  a_mlmg.setBottomMaxIter(m_mg_bottom_maxiter);
  if (m_fixed_mg_it > 0) {
    a_mlmg.setFixedIter(m_fixed_mg_it);
  }
  MLMGUtils::setBottomSolver(
    a_mlmg, a_bottom_solver, a_prefix, m_hypre_namespace, m_mg_bottom_rtol);
}

void
PrecondOp::setDiffOpScalars(
  const amrex::Real& aScal, const amrex::Real& bScal, const amrex::Real& cScal)
{
  m_diff->setScalars(aScal, bScal, cScal);
}

void
PrecondOp::setDiffOpRelaxation(const amrex::Real& a_omega)
{
  m_diff->setRelaxation(a_omega);
}

void
PrecondOp::setDriftOpScalars(const amrex::Real& aScal, const amrex::Real& bScal)
{
  m_drift->setScalars(aScal, bScal);
}

void
PrecondOp::setStildaOpScalars(
  const amrex::Real& aScal, const amrex::Real& bScal)
{
  m_Stilda->setScalars(aScal, bScal);
}

void
PrecondOp::setDiffOpBCs(const amrex::BCRec& a_bcrec)
{
  m_diff->setDomainBC(
    getOpBC(amrex::Orientation::low, a_bcrec),
    getOpBC(amrex::Orientation::high, a_bcrec));
}

void
PrecondOp::setDriftOpBCs(const amrex::BCRec& a_bcrec)
{
  m_drift->setDomainBC(
    getOpBC(amrex::Orientation::low, a_bcrec),
    getOpBC(amrex::Orientation::high, a_bcrec));
}

void
PrecondOp::setStildaOpBCs(const amrex::BCRec& a_bcrec)
{
  m_Stilda->setDomainBC(
    getOpBC(amrex::Orientation::low, a_bcrec),
    getOpBC(amrex::Orientation::high, a_bcrec));
}

void
PrecondOp::setDiffOpLevelBC(int lev, amrex::MultiFab* a_MF)
{
  m_diff->setLevelBC(lev, a_MF);
}

void
PrecondOp::setDriftOpLevelBC(int lev, amrex::MultiFab* a_MF)
{
  m_drift->setLevelBC(lev, a_MF);
}

void
PrecondOp::setStildaOpLevelBC(int lev, amrex::MultiFab* a_MF)
{
  m_Stilda->setLevelBC(lev, a_MF);
}

void
PrecondOp::setDiffOpACoeff(int lev, const amrex::Real& acoeff)
{
  m_diff->setACoeffs(lev, acoeff);
}

void
PrecondOp::setDiffOpBCoeff(
  int lev, const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bCoeff)
{
  m_diff->setBCoeffs(lev, a_bCoeff);
}

void
PrecondOp::setDiffOpCCoeff(
  int lev, const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_cCoeff)
{
  m_diff->setCCoeffs(lev, a_cCoeff);
}

void
PrecondOp::getDiffOpDiagonal(int lev, amrex::MultiFab& a_diffOpDiag)
{
  m_diff->getDiagonal(lev, a_diffOpDiag);
}

void
PrecondOp::setDriftOpBCoeff(
  int lev, const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bCoeff)
{
  m_drift->setBCoeffs(lev, a_bCoeff);
}

void
PrecondOp::setStildaOpBCoeff(
  int lev, const amrex::Array<const amrex::MultiFab*, AMREX_SPACEDIM>& a_bCoeff)
{
  m_Stilda->setBCoeffs(lev, a_bCoeff);
}

void
PrecondOp::diffOpSolve(
  const amrex::Vector<amrex::MultiFab*>& a_sol,
  const amrex::Vector<const amrex::MultiFab*>& a_rhs)
{
  m_mlmg_diff->solve(a_sol, a_rhs, m_mg_rtol, m_mg_atol);
}

void
PrecondOp::StildaOpSolve(
  const amrex::Vector<amrex::MultiFab*>& a_sol,
  const amrex::Vector<const amrex::MultiFab*>& a_rhs)
{
  m_mlmg_Stilda->solve(a_sol, a_rhs, m_mg_rtol, m_mg_atol);
}

void
PrecondOp::driftOpApply(
  const amrex::Vector<amrex::MultiFab*>& a_Ax,
  const amrex::Vector<amrex::MultiFab*>& a_x)
{
  m_mlmg_drift->apply(a_Ax, a_x);
}

amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM>
PrecondOp::getOpBC(amrex::Orientation::Side a_side, const amrex::BCRec& a_bc)
{
  amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> r;
  for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    if (m_pelelm->Geom(0).isPeriodic(idim)) {
      r[idim] = amrex::LinOpBCType::Periodic;
    } else {
      auto amrexbc =
        (a_side == amrex::Orientation::low) ? a_bc.lo(idim) : a_bc.hi(idim);
      if (amrexbc == amrex::BCType::ext_dir) {
        r[idim] = amrex::LinOpBCType::Dirichlet;
      } else if (
        amrexbc == amrex::BCType::foextrap ||
        amrexbc == amrex::BCType::hoextrap ||
        amrexbc == amrex::BCType::reflect_even) {
        r[idim] = amrex::LinOpBCType::Neumann;
      } else if (amrexbc == amrex::BCType::reflect_odd) {
        r[idim] = amrex::LinOpBCType::reflect_odd;
      } else {
        r[idim] = amrex::LinOpBCType::bogus;
      }
    }
  }
  return r;
}

void
PrecondOp::readParameters()
{
  amrex::ParmParse pp("ef.precond");

  pp.query("diff_verbose", m_diff_verbose);
  pp.query("drift_verbose", m_drift_verbose);
  pp.query("Stilda_verbose", m_Stilda_verbose);
  pp.query("bottom_verbose", m_bottom_verbose);
  pp.query("fixedIter", m_fixed_mg_it);
  pp.query("rtol", m_mg_rtol);
  pp.query("atol", m_mg_atol);
  pp.query("max_iter", m_mg_max_iter);
  pp.query("bottom_maxiter", m_mg_bottom_maxiter);
  pp.query("bottom_rtol", m_mg_bottom_rtol);
  pp.query("max_order", m_mg_maxorder);
  pp.query("max_coarsening_level_diff", m_mg_max_coarsening_level_diff);
  pp.query("max_coarsening_level_Stilda", m_mg_max_coarsening_level_Stilda);
  pp.query("num_pre_smooth", m_num_pre_smooth);
  pp.query("num_post_smooth", m_num_post_smooth);
  pp.query("diff_bottom_solver", m_diff_bottom_solver);
  pp.query("Stilda_bottom_solver", m_Stilda_bottom_solver);
  pp.query("hypre_namespace", m_hypre_namespace);
  pp.query_enum_case_insensitive("Stilda_multigrid_type", m_Stilda_mg_type);

  AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
    m_diff_bottom_solver.empty() ||
      !MLMGUtils::isMatrixBottomSolver(
        MLMGUtils::parseBottomSolver(
          m_diff_bottom_solver, "ef.precond.diff_bottom_solver")),
    "ef.precond.diff_bottom_solver: the electron drift-diffusion operator "
    "has no matrix (smoother, bicgstab, cg, bicgcg or cgbicg)");
}
