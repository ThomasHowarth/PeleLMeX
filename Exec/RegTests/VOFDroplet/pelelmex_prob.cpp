#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm() // NOLINT(readability-make-member-function-const)
{
  amrex::ParmParse pp("prob");
  std::string type = "static";
  pp.query("type", type);
  if (type == "static") {
    prob_parm->type = DropletType::Static;
  } else if (type == "oscillate") {
    prob_parm->type = DropletType::Oscillate;
  } else if (type == "breakup") {
    prob_parm->type = DropletType::Breakup;
  } else {
    amrex::Abort("prob.type must be static, oscillate or breakup");
  }
  amrex::Vector<amrex::Real> center;
  if (pp.queryarr("center", center)) {
    AMREX_ALWAYS_ASSERT(center.size() == AMREX_SPACEDIM);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      prob_parm->center[d] = center[d];
    }
  }
  pp.query("radius", prob_parm->radius);
  pp.query("amplitude", prob_parm->amplitude);
  pp.query("velocity", prob_parm->velocity);
}

void
PeleLM::freeProbParm()
{
}
