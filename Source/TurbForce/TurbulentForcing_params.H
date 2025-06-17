#ifndef IAMR_TurbulentForcing_params_H_
#define IAMR_TurbulentForcing_params_H_

namespace TurbulentForcing {

    // function to generate the parameters
    void init_turbulent_forcing (amrex::GeometryData const& geomdata);

    // function to make the forcing term
    void addTurbVelForces(amrex::GeometryData const& geomdata,
			  const amrex::Box& bx,
			  const amrex::Real& a_time,
			  amrex::Array4<amrex::Real> const& force,
			  amrex::Array4<const amrex::Real> const& rho);

    void deriveForcing(PeleLM* a_pelelm,
		       const amrex::Box& bx,
		       amrex::FArrayBox& derfab,
		       int dcomp,
		       int ncomp,
		       const amrex::FArrayBox& statefab,
		       const amrex::FArrayBox& reactfab,
		       const amrex::FArrayBox& pressfab,
		       const amrex::Geometry& geom,
		       amrex::Real time,
		       const amrex::Vector<amrex::BCRec>& bcrec,
		       int level);

    extern AMREX_GPU_MANAGED int verbose;
    // factor by which to reduce sampling for faster performance
    extern AMREX_GPU_MANAGED int ff_factor;
    // how many modes to use
    extern AMREX_GPU_MANAGED int nmodes;
    // don't use any modes below mode_start. We probably don't need this
    extern AMREX_GPU_MANAGED int mode_start;
    // fine tune force scale
    extern AMREX_GPU_MANAGED amrex::Real force_scale_fudge;
    // suppress symmetry breaking modes
    extern AMREX_GPU_MANAGED amrex::Real forcing_epsilon;
    // time offset for forcing function
    extern AMREX_GPU_MANAGED amrex::Real time_offset;

    constexpr int array_size = 33;
    constexpr int num_fdarray = 17;
    // forcedata will contain num_fdarray arrays of size (0,0,0)(array_size-1,array_size-1,array_size-1)
    extern amrex::Real* forcedata;

}
#endif
