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
  } else {
    amrex::Abort("prob.type must be pool, rt or dambreak");
  }
  pp.query("depth", prob_parm->depth);
  pp.query("tilt", prob_parm->tilt);
  pp.query("amplitude", prob_parm->amplitude);
  pp.query("wavelength", prob_parm->wavelength);
  pp.query("width", prob_parm->width);
  pp.query("height", prob_parm->height);
}

void
PeleLM::freeProbParm()
{
}
