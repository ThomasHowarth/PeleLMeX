#ifndef PELELM_PROB_PARM_H
#define PELELM_PROB_PARM_H

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>
#include <PeleLMeX_FlowControllerData.H>

using namespace amrex::literals;

struct ProbParm
{
  amrex::Real P_mean = 101325.0_rt;
  amrex::Real standoff = 0.0_rt;
  amrex::Real pertmag = 0.0004_rt;
  amrex::Real pertlength = -1.0;
  amrex::Real V_in = 0.2;
  FlowControllerData FCData;
};
#endif
