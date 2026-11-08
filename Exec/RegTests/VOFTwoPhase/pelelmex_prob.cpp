#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm() // NOLINT(readability-make-member-function-const)
{
  amrex::ParmParse pp("prob");
  std::string type = "pool";
  pp.query("type", type);
  if (type == "pool") {
    prob_parm->type = TwoPhaseType::Pool;
  } else if (type == "rt") {
    prob_parm->type = TwoPhaseType::RT;
  } else if (type == "dambreak") {
    prob_parm->type = TwoPhaseType::DamBreak;
  } else if (type == "translate") {
    prob_parm->type = TwoPhaseType::Translate;
  } else {
    amrex::Abort("prob.type must be pool, rt, dambreak or translate");
  }
  pp.query("depth", prob_parm->depth);
  pp.query("tilt", prob_parm->tilt);
  pp.query("amplitude", prob_parm->amplitude);
  pp.query("wavelength", prob_parm->wavelength);
  pp.query("width", prob_parm->width);
  pp.query("height", prob_parm->height);
  pp.query("radius", prob_parm->radius);
  amrex::Vector<amrex::Real> center(AMREX_SPACEDIM);
  if (pp.queryarr("center", center, 0, AMREX_SPACEDIM) != 0) {
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      prob_parm->center[d] = center[d];
    }
  }
  amrex::Vector<amrex::Real> velocity(AMREX_SPACEDIM);
  if (pp.queryarr("velocity", velocity, 0, AMREX_SPACEDIM) != 0) {
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
      prob_parm->velocity[d] = velocity[d];
    }
  }
}

void
PeleLM::freeProbParm()
{
}
