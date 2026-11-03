#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm() // NOLINT(readability-make-member-function-const)
{
  amrex::ParmParse pp("prob");
  amrex::Vector<amrex::Real> center;
  if (pp.queryarr("center", center)) {
    AMREX_ALWAYS_ASSERT(center.size() == AMREX_SPACEDIM);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      prob_parm->center[d] = center[d];
    }
  }
  pp.query("diameter", prob_parm->diameter);
  pp.query("velocity", prob_parm->velocity);
}

void
PeleLM::freeProbParm()
{
}
