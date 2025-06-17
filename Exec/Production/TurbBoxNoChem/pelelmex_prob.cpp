#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>


void
PeleLM::readProbParm()
{
  amrex::ParmParse pp("prob");
  pp.query("P_mean", PeleLM::prob_parm->P_mean);
  pp.query("T_mean", PeleLM::prob_parm->T_mean); // only used if not incompressible
}
