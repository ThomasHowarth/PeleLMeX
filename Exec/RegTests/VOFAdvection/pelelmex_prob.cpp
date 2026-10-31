#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm() // NOLINT(readability-make-member-function-const)
{
  amrex::ParmParse pp("prob");
  std::string type = "zalesak";
  pp.query("type", type);
  if (type == "zalesak") {
    prob_parm->type = VOFTestType::Zalesak;
  } else if (type == "vortex") {
    prob_parm->type = VOFTestType::Vortex;
  } else if (type == "translate") {
    prob_parm->type = VOFTestType::Translate;
  } else {
    amrex::Abort("prob.type must be zalesak, vortex or translate");
  }
  pp.query("radius", prob_parm->radius);
  amrex::Vector<amrex::Real> center(AMREX_SPACEDIM);
  if (pp.queryarr("center", center, 0, AMREX_SPACEDIM) != 0) {
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      prob_parm->center[d] = center[d];
    }
  }
  amrex::Vector<amrex::Real> vel(AMREX_SPACEDIM);
  if (pp.queryarr("velocity", vel, 0, AMREX_SPACEDIM) != 0) {
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      prob_parm->vel[d] = vel[d];
    }
  }
  pp.query("slot_width", prob_parm->slot_width);
  pp.query("slot_length", prob_parm->slot_length);
  pp.query("period", prob_parm->period);
}

void
PeleLM::freeProbParm()
{
}
