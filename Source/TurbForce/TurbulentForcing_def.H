#include <TurbulentForcing_params.H>
#include <depRand.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Arena.H>

using namespace amrex;

// factor by which to reduce sampling for faster performance
AMREX_GPU_MANAGED int TurbulentForcing::ff_factor;
// how many modes to use
AMREX_GPU_MANAGED int TurbulentForcing::nmodes;
// don't use any modes below mode_start. We probably don't need this
AMREX_GPU_MANAGED int TurbulentForcing::mode_start;
// Diagnostic print outs
AMREX_GPU_MANAGED int TurbulentForcing::verbose;
// suppress symmetry breaking modes
AMREX_GPU_MANAGED Real TurbulentForcing::force_scale_fudge;
// suppress symmetry breaking modes
AMREX_GPU_MANAGED Real TurbulentForcing::forcing_epsilon;
// time offset for forcing function
AMREX_GPU_MANAGED Real TurbulentForcing::time_offset;

Real* TurbulentForcing::forcedata;

//
//
//

void
TurbulentForcing::init_turbulent_forcing (GeometryData const& geomdata)
{
    const Real* problo = geomdata.ProbLo();
    const Real* probhi = geomdata.ProbHi();

    constexpr Real Pi = M_PI;
    constexpr Real TwoPi = 2.*Pi;

    // Start with checks.
    // This is for 3D only.
    AMREX_ALWAYS_ASSERT(AMREX_SPACEDIM==3);

    // Forcing requires that Lx==Ly, Lz can be longer
    Real Lx = probhi[0]-problo[0];
    Real Ly = probhi[1]-problo[1];
    Real Lz = probhi[2]-problo[2];
    AMREX_ALWAYS_ASSERT(Lx==Ly);

    // Read in parameters
    ParmParse pp("turb");

    verbose = 0;
    pp.query("v", verbose);        

    // first mode to force
    mode_start = 0;
    pp.query("mode_start", mode_start);

    // largest mode to force
    nmodes = 4;
    pp.query("nmodes", nmodes);

    // fast force coarsening factor
    ff_factor = 4;
    pp.query("ff_factor", ff_factor);

    // fine scale tuning of the forcing
    force_scale_fudge = 1.0;
    pp.query("force_scale_fudge", force_scale_fudge);

    // reduce amplitude of modes used for breaking symmetry
    forcing_epsilon = 0.1;
    pp.query("forcing_epsilon", forcing_epsilon);

    // target velocity fluctuation
    Real urms;
    pp.get("urms",urms);

    // shape the spectrum of the forcing
    int spectrum_type = 2;
    pp.query("spectrum_type",spectrum_type);

    // reduce the impact of any zero mode 
    int moderate_zero_modes = 1;
    pp.query("moderate_zero_modes",moderate_zero_modes);

    // time offset (e.g. for starting from precursor turb sim)
    time_offset = 0.0;
    pp.query("time_offset",time_offset);

    // allow peridodic reproduction in z
    // can be used as a flag i.e. =1 for a factor 2
    // or as the factor itself
    int hack_lz = 0;
    pp.query("hack_lz",hack_lz);

    if (hack_lz>0) {
      if (hack_lz==1) {
        Lz = Lz/2.0;
      } else {
        Lz = Lz/hack_lz;
      }
    }

    if (verbose)
    {
        Print() << "Lx = " << Lx << std::endl;
        Print() << "Ly = " << Ly << std::endl;
        Print() << "Lz = " << Lz << std::endl;
    }

    //
    // calculate turbulent forcing scales (using andy's reference values)
    //
    const Real flr(2.64e-2), fvr(3.27), fts_min(1.21e-3), fts_max(2.42e-3);
    Real fsr;
    if (Lx == Lz) { 
	fsr = 1.47e7; // cubic reference value
    } else {          
	fsr = 1.41e7; // non-cubic reference value
    }
    const Real force_scale = force_scale_fudge * fsr * pow(urms / fvr,2) * pow(flr/Lx,3);
    const Real forcing_time_scale_min = fts_min * (fvr/flr) * (Lx/urms);
    const Real forcing_time_scale_max = fts_max * (fvr/flr) * (Lx/urms);

    Real Lmin = std::min(Lx,std::min(Ly,Lz));
    Real kappaMax = ((Real)nmodes)/Lmin + 1.0e-8;
    int nxmodes = nmodes*(int)(0.5+Lx/Lmin);
    int nymodes = nmodes*(int)(0.5+Ly/Lmin);
    int nzmodes = nmodes*(int)(0.5+Lz/Lmin);

    if (verbose)
    {
        Print() << "Lmin = " << Lmin << std::endl;
        Print() << "kappaMax = " << kappaMax << std::endl;
        Print() << "nxmodes = " << nxmodes << std::endl;
        Print() << "nymodes = " << nxmodes << std::endl;
        Print() << "nzmodes = " << nxmodes << std::endl;
    }

    Real freqMin = 1.0/forcing_time_scale_max;
    Real freqMax = 1.0/forcing_time_scale_min;
    Real freqDiff= freqMax-freqMin;

    if (verbose)
    {
	Print() << "force_scale = " << force_scale << "\n";
        Print() << "forcing_time_scale_min = " << forcing_time_scale_min << std::endl;
        Print() << "forcing_time_scale_max = " << forcing_time_scale_max << std::endl;
        Print() << "freqMin = " << freqMin << std::endl;
        Print() << "freqMax = " << freqMax << std::endl;
        Print() << "freqDiff = " << freqDiff << std::endl;
    }

    // tmp CPU storage that holds everything in one flat array
    const int num_elmts = array_size*array_size*array_size;
    const int tmp_size  = num_fdarray*num_elmts;
    Real tmp[tmp_size];

    // Separate out forcing data into individual Array4's
    int i_arr = 0;
    int fd_ncomp = 1;
    Dim3 fd_begin{0,0,0};
    Dim3 fd_end{array_size,array_size,array_size};

    Array4<Real> FTX(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> TAT(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FPX(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FPY(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FPZ(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FAX(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FAY(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FAZ(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FPXX(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FPXY(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FPXZ(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FPYX(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FPYY(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FPYZ(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FPZX(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FPZY(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    Array4<Real> FPZZ(&tmp[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
    
    // initiate the magic
    DepRand::InitRandom((unsigned long)111397);
    
    int mode_count = 0;

    int xstep = (int)(Lx/Lmin+0.5);
    int ystep = (int)(Ly/Lmin+0.5);
    int zstep = (int)(Lz/Lmin+0.5);

    if (verbose)
        Print() << "Mode step = " << xstep << " " << ystep << " " << zstep << std::endl;

    for (int kz = mode_start*zstep; kz <= nzmodes; kz += zstep ) {
      Real kzd = (Real)kz;
      for (int ky = mode_start*ystep; ky <= nymodes; ky += ystep ) {
        Real kyd = (Real)ky;
        for (int kx = mode_start*xstep; kx <= nxmodes; kx += xstep ) {
          Real kxd = (Real)kx;

          Real kappa = sqrt( (kxd*kxd)/(Lx*Lx) + (kyd*kyd)/(Ly*Ly) + (kzd*kzd)/(Lz*Lz) );

          if (kappa<=kappaMax) {
            FTX(kx,ky,kz) = (freqMin + freqDiff*DepRand::Random() )*TwoPi;
	    DepRand::Random(); // dummy FTY (don't remove)
	    DepRand::Random(); // dummy FTZ (don't remove)
            // Translation angles, theta=0..2Pi and phi=0..Pi
            TAT(kx,ky,kz) = DepRand::Random()*TwoPi;
	    DepRand::Random(); // dummy TAP (don't remove)
            // Phases
            FPX(kx,ky,kz) = DepRand::Random()*TwoPi;
            FPY(kx,ky,kz) = DepRand::Random()*TwoPi;
            FPZ(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPXX(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPYX(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPZX(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPXY(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPYY(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPZY(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPXZ(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPYZ(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPZZ(kx,ky,kz) = DepRand::Random()*TwoPi;

            // Amplitudes (alpha)
            Real thetaTmp      = DepRand::Random()*TwoPi;
            Real cosThetaTmp   = cos(thetaTmp);
            Real sinThetaTmp   = sin(thetaTmp);

            Real phiTmp        = DepRand::Random()*Pi;
            Real cosPhiTmp     = cos(phiTmp);
            Real sinPhiTmp     = sin(phiTmp);

            Real px = cosThetaTmp * sinPhiTmp;
            Real py = sinThetaTmp * sinPhiTmp;
            Real pz =               cosPhiTmp;

            Real mp2 = px*px + py*py + pz*pz;
            if (kappa < 0.000001) {
	      if (verbose)
		Print() << "ZERO AMPLITUDE MODE " << kx << ky << kz << std::endl;
              FAX(kx,ky,kz) = 0.;
              FAY(kx,ky,kz) = 0.;
              FAZ(kx,ky,kz) = 0.;
            } else {
              // Count modes that contribute
              mode_count++;
              // Set amplitudes
              Real Ekh;
              if (spectrum_type==1) {
                Ekh = 1. / kappa;
              } else if (spectrum_type==2) {
                Ekh = 1. / (kappa*kappa);
              } else {
                Ekh = 1.;
              }
	      // div_free_forcing (assumed) needs another
	      Ekh /= kappa;
	      
              if (moderate_zero_modes==1) {
                if (kx==0) Ekh /= 2.;
                if (ky==0) Ekh /= 2.;
                if (kz==0) Ekh /= 2.;
              }
              if (force_scale>0.) {
                FAX(kx,ky,kz) = force_scale * px * Ekh / mp2;
                FAY(kx,ky,kz) = force_scale * py * Ekh / mp2;
                FAZ(kx,ky,kz) = force_scale * pz * Ekh / mp2;
              } else {
                FAX(kx,ky,kz) = px * Ekh / mp2;
                FAY(kx,ky,kz) = py * Ekh / mp2;
                FAZ(kx,ky,kz) = pz * Ekh / mp2;
              }

              if (verbose)
              {
                  Print() << "Mode";
                  Print() << "kappa = " << kx << " " << ky << " " << kz << " " << kappa << " "
                          << sqrt(FAX(kx,ky,kz)*FAX(kx,ky,kz)+FAY(kx,ky,kz)*FAY(kx,ky,kz)+FAZ(kx,ky,kz)*FAZ(kx,ky,kz)) << std::endl;
                  Print() << "Amplitudes - A" << std::endl;
                  Print() << FAX(kx,ky,kz) << " " << FAY(kx,ky,kz) << " " << FAZ(kx,ky,kz) << std::endl;
                  Print() << "Frequencies" << std::endl;
                  Print() << FTX(kx,ky,kz) << std::endl;
                  Print() << "TAT" << std::endl;
                  Print() << TAT(kx,ky,kz) << std::endl;
                  Print() << "Amplitudes - AA" << std::endl;
                  Print() << FPXX(kx,ky,kz) << " " << FPYX(kx,ky,kz) << " " << FPZX(kx,ky,kz) << std::endl;
                  Print() << FPXY(kx,ky,kz) << " " << FPYY(kx,ky,kz) << " " << FPZY(kx,ky,kz) << std::endl;
                  Print() << FPXZ(kx,ky,kz) << " " << FPYZ(kx,ky,kz) << " " << FPZZ(kx,ky,kz) << std::endl;
              }
            }
          }
        }
      }
    }

    // Now let's break symmetry, have to assume high aspect ratio in z for now
    int reduced_mode_count = 0;

    for (int kz = 1; kz < zstep; kz++ ) {
      Real kzd = (Real)kz;
      for (int ky = mode_start; ky <= nymodes; ky += ystep ) {
        Real kyd = (Real)ky;
        for (int kx = mode_start; kx <= nxmodes; kx += xstep ) {
          Real kxd = (Real)kx;

          Real kappa = sqrt( (kxd*kxd)/(Lx*Lx) + (kyd*kyd)/(Ly*Ly) + (kzd*kzd)/(Lz*Lz) );

          if (kappa<=kappaMax) {
            FTX(kx,ky,kz) = (freqMin + freqDiff*DepRand::Random() )*TwoPi;
	    DepRand::Random(); // dummy FTY (don't remove)
	    DepRand::Random(); // dummy FTZ (don't remove)
            // Translation angles, theta=0..2Pi and phi=0..Pi
            TAT(kx,ky,kz) = DepRand::Random()*TwoPi;
	    DepRand::Random(); // dummy TAP (don't remove)
            // Phases
            FPX(kx,ky,kz) = DepRand::Random()*TwoPi;
            FPY(kx,ky,kz) = DepRand::Random()*TwoPi;
            FPZ(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPXX(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPYX(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPZX(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPXY(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPYY(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPZY(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPXZ(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPYZ(kx,ky,kz) = DepRand::Random()*TwoPi;
	    FPZZ(kx,ky,kz) = DepRand::Random()*TwoPi;

            // Amplitudes (alpha)
            Real thetaTmp      = DepRand::Random()*TwoPi;
            Real cosThetaTmp   = cos(thetaTmp);
            Real sinThetaTmp   = sin(thetaTmp);

            Real phiTmp        = DepRand::Random()*Pi;
            Real cosPhiTmp     = cos(phiTmp);
            Real sinPhiTmp     = sin(phiTmp);

            Real px = cosThetaTmp * sinPhiTmp;
            Real py = sinThetaTmp * sinPhiTmp;
            Real pz =               cosPhiTmp;

            Real mp2 = px*px + py*py + pz*pz;
            if (kappa < 0.000001) {
              Print() << "ZERO AMPLITUDE MODE " << kx << ky << kz << std::endl;
              FAX(kx,ky,kz) = 0.;
              FAY(kx,ky,kz) = 0.;
              FAZ(kx,ky,kz) = 0.;
            } else {
              // Count modes that contribute
              reduced_mode_count++;
              // Set amplitudes
              Real Ekh;
              if (spectrum_type==1) {
                Ekh = 1. / kappa;
              } else if (spectrum_type==2) {
                Ekh = 1. / (kappa*kappa);
              } else {
                Ekh = 1.;
              }
	      // div_free_forcing (assumed) needs another
	      Ekh /= kappa;
	      
              if (moderate_zero_modes==1) {
                if (kx==0) Ekh /= 2.;
                if (ky==0) Ekh /= 2.;
                if (kz==0) Ekh /= 2.;
              }
              if (force_scale>0.) {
                FAX(kx,ky,kz) = forcing_epsilon * force_scale * px * Ekh / mp2;
                FAY(kx,ky,kz) = forcing_epsilon * force_scale * py * Ekh / mp2;
                FAZ(kx,ky,kz) = forcing_epsilon * force_scale * pz * Ekh / mp2;
              } else {
                FAX(kx,ky,kz) = forcing_epsilon * px * Ekh / mp2;
                FAY(kx,ky,kz) = forcing_epsilon * py * Ekh / mp2;
                FAZ(kx,ky,kz) = forcing_epsilon * pz * Ekh / mp2;
              }

              if (verbose)
              {
                  Print() << "Mode";
                  Print() << "kappa = " << kx << " " << ky << " " << kz << " " << kappa << " "
                          << sqrt(FAX(kx,ky,kz)*FAX(kx,ky,kz)+FAY(kx,ky,kz)*FAY(kx,ky,kz)+FAZ(kx,ky,kz)*FAZ(kx,ky,kz)) << std::endl;
                  Print() << "Amplitudes - A" << std::endl;
                  Print() << FAX(kx,ky,kz) << " " << FAY(kx,ky,kz) << " " << FAZ(kx,ky,kz) << std::endl;
                  Print() << "Frequencies" << std::endl;
                  Print() << FTX(kx,ky,kz) << std::endl;
                  Print() << "TAT" << std::endl;
                  Print() << TAT(kx,ky,kz) << std::endl;
                  Print() << "Amplitudes - AA" << std::endl;
                  Print() << FPXX(kx,ky,kz) << " " << FPYX(kx,ky,kz) << " " << FPZX(kx,ky,kz) << std::endl;
                  Print() << FPXY(kx,ky,kz) << " " << FPYY(kx,ky,kz) << " " << FPZY(kx,ky,kz) << std::endl;
                  Print() << FPXZ(kx,ky,kz) << " " << FPYZ(kx,ky,kz) << " " << FPZZ(kx,ky,kz) << std::endl;
              }
            }
          }
        }
      }
    }

    Print() << "mode_count = " << mode_count << std::endl;
    Print() << "reduced_mode_count = " << reduced_mode_count << std::endl;
    if (spectrum_type==1) {
      Print() << "Spectrum type 1" << std::endl;
    } else if (spectrum_type==2) {
      Print() << "Spectrum type 2" << std::endl;
    } else {
      Print() << "Spectrum type OTHER" << std::endl;
    }

// Now allocate forcedata and copy in tmp array.
#ifdef AMREX_USE_GPU
    if (Gpu::inLaunchRegion())
    {
        forcedata = static_cast<Real*>(The_Arena()->alloc(tmp_size*sizeof(Real)));
        Gpu::htod_memcpy_async(forcedata, tmp, tmp_size*sizeof(Real));
    }
    else
#endif
    {
        forcedata = static_cast<Real*>(The_Pinned_Arena()->alloc(tmp_size*sizeof(Real)));
        std::memcpy(forcedata, tmp, tmp_size*sizeof(Real));
    }
}

//
//
//

void
TurbulentForcing::addTurbVelForces(GeometryData const& geomdata,
				   const Box& bx,
				   const Real& time,
				   Array4<Real> const& force,
				   Array4<const Real> const& rho)
{
     // For now, only works in 3D and without tiling
     AMREX_ASSERT(AMREX_SPACEDIM==3);

     constexpr Real Pi = M_PI;
     constexpr Real TwoPi = 2.*Pi;
     
     // Physical coordinates
     auto const& problo = geomdata.ProbLo();
     auto const& probhi = geomdata.ProbHi();

     Real Lx = probhi[0]-problo[0];
     Real Ly = probhi[1]-problo[1];
     Real Lz = probhi[2]-problo[2];
     Real Lmin = min(Lx,Ly,Lz);

     const int* f_lo = bx.loVect();
     const int* f_hi = bx.hiVect();
    
     int xstep = static_cast<int>(Lx/Lmin+0.5);
     int ystep = static_cast<int>(Ly/Lmin+0.5);
     int zstep = static_cast<int>(Lz/Lmin+0.5);

     Real kappaMax = TurbulentForcing::nmodes/Lmin + 1.0e-8;

     const Real forcetime = time + time_offset;

     auto const& dx = geomdata.CellSize();

     // Separate out forcing data into individual Array4's
     int i_arr = 0;
     int fd_ncomp = 1;
     int num_elmts=TurbulentForcing::array_size*TurbulentForcing::array_size*TurbulentForcing::array_size;
     Dim3 fd_begin{0,0,0};
     Dim3 fd_end{TurbulentForcing::array_size,TurbulentForcing::array_size,TurbulentForcing::array_size};

     Array4<Real> FTX(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> TAT(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FPX(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FPY(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FPZ(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FAX(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FAY(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FAZ(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FPXX(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FPXY(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FPXZ(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FPYX(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FPYY(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FPYZ(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FPZX(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FPZY(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);
     Array4<Real> FPZZ(&TurbulentForcing::forcedata[(i_arr++)*num_elmts], fd_begin, fd_end, fd_ncomp);

     RealBox loc = RealBox(bx,geomdata.CellSize(),geomdata.ProbLo());
     const auto& loc_lo = loc.lo();
     GpuArray<Real, AMREX_SPACEDIM> xlo = {AMREX_D_DECL(loc_lo[0],loc_lo[1],loc_lo[2])};

     //
     // Construct force at fewer points and then interpolate.
     // This is much faster on CPU.
     //

     Real hx = dx[0];
     Real hy = dx[1];
     Real hz = dx[2];

     int ilo = f_lo[0];
     int jlo = f_lo[1];
     int klo = f_lo[2];

     int ihi = f_hi[0];
     int jhi = f_hi[1];
     int khi = f_hi[2];

     // coarse cell size
     Real ff_hx = hx*TurbulentForcing::ff_factor;
     Real ff_hy = hy*TurbulentForcing::ff_factor;
     Real ff_hz = hz*TurbulentForcing::ff_factor;

     // coarse bounds (without accounting for ghost cells)
     // FIXME -- think about how bx (the box we want to fill) may not be the same as the force box!
     int ff_ilo = ilo/TurbulentForcing::ff_factor;
     int ff_jlo = jlo/TurbulentForcing::ff_factor;
     int ff_klo = klo/TurbulentForcing::ff_factor;

     int ff_ihi = (ihi+1)/TurbulentForcing::ff_factor;
     int ff_jhi = (jhi+1)/TurbulentForcing::ff_factor;
     int ff_khi = (khi+1)/TurbulentForcing::ff_factor;

     // adjust for ghost cells
     if (ilo < (ff_ilo*TurbulentForcing::ff_factor)) {
         ff_ilo=ff_ilo-1;
     }
     if (jlo < (ff_jlo*TurbulentForcing::ff_factor)) {
         ff_jlo=ff_jlo-1;
     }
     if (klo < (ff_klo*TurbulentForcing::ff_factor)) {
         ff_klo=ff_klo-1;
     }
     if (ihi == (ff_ihi*TurbulentForcing::ff_factor)) {
         ff_ihi=ff_ihi+1;
     }
     if (jhi == (ff_jhi*TurbulentForcing::ff_factor)) {
         ff_jhi=ff_jhi+1;
     }
     if (khi == (ff_khi*TurbulentForcing::ff_factor)) {
         ff_khi=ff_khi+1;
     }

     // allocate coarse force array
     Box ffbx(IntVect(ff_ilo, ff_jlo, ff_klo), IntVect(ff_ihi, ff_jhi, ff_khi));
     // not sure if want elixir, gpu::sync, or async_arena here...
     FArrayBox ff_force(ffbx,AMREX_SPACEDIM);
     const auto& ffarr = ff_force.array();

     // Construct node-based coarse forcing
     ParallelFor(ffbx, [ = ]
     AMREX_GPU_DEVICE (int i, int j, int k ) noexcept
     {
         Real z = xlo[2] + ff_hz*(k-ff_klo);
         Real y = xlo[1] + ff_hy*(j-ff_jlo);
         Real x = xlo[0] + ff_hx*(i-ff_ilo);

         for (int n = 0; n < AMREX_SPACEDIM; n++)
             ffarr(i,j,k,n) = 0.0;

         // forcedata (and Array4) has column-major layout
         for (int kz = TurbulentForcing::mode_start*zstep; kz <= TurbulentForcing::nmodes*zstep; kz += zstep) {
             for (int ky = TurbulentForcing::mode_start*ystep; ky <= TurbulentForcing::nmodes*ystep; ky += ystep) {
                 for (int kx = TurbulentForcing::mode_start*xstep; kx <= TurbulentForcing::nmodes*xstep; kx += xstep)
                 {
                     Real kappa = sqrt( (kx*kx)/(Lx*Lx) + (ky*ky)/(Ly*Ly) + (kz*kz)/(Lz*Lz) );

                     if (kappa <= kappaMax)
                     {
                         Real xT = cos(FTX(kx,ky,kz)*forcetime + TAT(kx,ky,kz));

                             ffarr(i,j,k,0) += xT *
                                 ( FAZ(kx,ky,kz)*TwoPi*(ky/Ly)
                                   *   sin(TwoPi*kx*x/Lx+FPZX(kx,ky,kz))
                                   *   cos(TwoPi*ky*y/Ly+FPZY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPZZ(kx,ky,kz))
                                   - FAY(kx,ky,kz)*TwoPi*(kz/Lz)
                                   *   sin(TwoPi*kx*x/Lx+FPYX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPYY(kx,ky,kz))
                                   *   cos(TwoPi*kz*z/Lz+FPYZ(kx,ky,kz)) );

                             ffarr(i,j,k,1) += xT *
                                 ( FAX(kx,ky,kz)*TwoPi*(kz/Lz)
                                   *   sin(TwoPi*kx*x/Lx+FPXX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPXY(kx,ky,kz))
                                   *   cos(TwoPi*kz*z/Lz+FPXZ(kx,ky,kz))
                                   - FAZ(kx,ky,kz)*TwoPi*(kx/Lx)
                                   *   cos(TwoPi*kx*x/Lx+FPZX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPZY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPZZ(kx,ky,kz)) );

                             ffarr(i,j,k,2) += xT *
                                 ( FAY(kx,ky,kz)*TwoPi*(kx/Lx)
                                   *   cos(TwoPi*kx*x/Lx+FPYX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPYY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPYZ(kx,ky,kz))
                                   - FAX(kx,ky,kz)*TwoPi*(ky/Ly)
                                   *   sin(TwoPi*kx*x/Lx+FPXX(kx,ky,kz))
                                   *   cos(TwoPi*ky*y/Ly+FPXY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPXZ(kx,ky,kz)) );
                     }
                 }
             }
         }

         //
         // For high aspect ratio domain, add more modes to break symmetry at a low level.
         // We assume Lz is longer, Lx = Ly.
         //
         for ( int kz = 1; kz <= zstep-1; kz++) {
             for ( int ky = TurbulentForcing::mode_start; ky <= TurbulentForcing::nmodes*ystep; ky++) {
                 for ( int kx = TurbulentForcing::mode_start; kx <= TurbulentForcing::nmodes*xstep; kx++)
                 {
                     Real kappa = sqrt( (kx*kx)/(Lx*Lx) + (ky*ky)/(Ly*Ly) + (kz*kz)/(Lz*Lz) );

                     if (kappa <= kappaMax)
                     {
                         Real xT = cos(FTX(kx,ky,kz)*forcetime + TAT(kx,ky,kz));

                             ffarr(i,j,k,0) += xT *
                                 ( FAZ(kx,ky,kz)*TwoPi*(ky/Ly)
                                   *   sin(TwoPi*kx*x/Lx+FPZX(kx,ky,kz))
                                   *   cos(TwoPi*ky*y/Ly+FPZY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPZZ(kx,ky,kz))
                                   - FAY(kx,ky,kz)*TwoPi*(kz/Lz)
                                   *   sin(TwoPi*kx*x/Lx+FPYX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPYY(kx,ky,kz))
                                   *   cos(TwoPi*kz*z/Lz+FPYZ(kx,ky,kz)) );

                             ffarr(i,j,k,1) += xT *
                                 ( FAX(kx,ky,kz)*TwoPi*(kz/Lz)
                                   *   sin(TwoPi*kx*x/Lx+FPXX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPXY(kx,ky,kz))
                                   *   cos(TwoPi*kz*z/Lz+FPXZ(kx,ky,kz))
                                   - FAZ(kx,ky,kz)*TwoPi*(kx/Lx)
                                   *   cos(TwoPi*kx*x/Lx+FPZX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPZY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPZZ(kx,ky,kz)) );

                             ffarr(i,j,k,2) += xT *
                                 ( FAY(kx,ky,kz)*TwoPi*(kx/Lx)
                                   *   cos(TwoPi*kx*x/Lx+FPYX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPYY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPYZ(kx,ky,kz))
                                   - FAX(kx,ky,kz)*TwoPi*(ky/Ly)
                                   *   sin(TwoPi*kx*x/Lx+FPXX(kx,ky,kz))
                                   *   cos(TwoPi*ky*y/Ly+FPXY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPXZ(kx,ky,kz)) );
                     }
                 }
             }
         }

     });

     // Need all of ffarr filled for next lambda
     Gpu::synchronize();

     // Now interpolate onto fine grid
     ParallelFor(bx, AMREX_SPACEDIM, [ = ]
     AMREX_GPU_DEVICE (int i, int j, int k, int n ) noexcept
     {
         int ff_k = k/TurbulentForcing::ff_factor;
         int ff_j = j/TurbulentForcing::ff_factor;
         int ff_i = i/TurbulentForcing::ff_factor;

         Real zd = ( hz*(k-klo + 0.5) - ff_hz*(ff_k-ff_klo) )/ff_hz;
         Real yd = ( hy*(j-jlo + 0.5) - ff_hy*(ff_j-ff_jlo) )/ff_hy;
         Real xd = ( hx*(i-ilo + 0.5) - ff_hx*(ff_i-ff_ilo) )/ff_hx;

         Real ff00 =  ffarr(ff_i  ,ff_j  ,ff_k  ,n) * (1. - xd)
             + ffarr(ff_i+1,ff_j  ,ff_k  ,n) * xd;
         Real ff01 =  ffarr(ff_i  ,ff_j  ,ff_k+1,n) * (1. - xd)
             + ffarr(ff_i+1,ff_j  ,ff_k+1,n) * xd;
         Real ff10 =  ffarr(ff_i  ,ff_j+1,ff_k  ,n) * (1. - xd)
             + ffarr(ff_i+1,ff_j+1,ff_k  ,n) * xd;
         Real ff11 =  ffarr(ff_i  ,ff_j+1,ff_k+1,n) * (1. - xd)
             + ffarr(ff_i+1,ff_j+1,ff_k+1,n) * xd;

         Real ff =  ( ff00*(1.-yd)+ff10*yd ) * (1. - zd)
             + ( ff01*(1.-yd)+ff11*yd ) * zd;

         force(i,j,k,n) += rho(i,j,k) * ff;

     });
}

//
// Add derived variables to plotfile
//
void
TurbulentForcing::deriveForcing(
  PeleLM* a_pelelm,
  const Box& bx,
  FArrayBox& derfab,
  int dcomp,
  int ncomp,
  const FArrayBox& statefab,
  const FArrayBox& /*reactfab*/,
  const FArrayBox& /*pressfab*/,
  const Geometry& geom,
  Real time,
  const Vector<BCRec>& /*bcrec*/,
  int /*level*/)

{
     amrex::ignore_unused(a_pelelm, ncomp);
     AMREX_ASSERT(derfab.box().contains(bx));
     AMREX_ASSERT(statefab.box().contains(bx));
     AMREX_ASSERT(derfab.nComp() >= dcomp + ncomp);
     AMREX_ASSERT(!a_pelelm->m_incompressible);

     // Need geom for forcing
     GeometryData const& geomdata = geom.data();

     // addTurbVelForces "adds", so need to zero this to start
     Array4<Real> const& der = derfab.array(dcomp);
     derfab.setVal<RunOn::Device>(0., bx, dcomp, ncomp);

     // if using turbforce and is incompressible
     // then make a fab with constant density m_rho
     FArrayBox DensityFab(bx);
     DensityFab.setVal(a_pelelm->m_rho);
     Array4<const Real> const& rho = (a_pelelm->m_incompressible) != 0
       ? DensityFab.array()
       : statefab.array(DENSITY);

     // call the function above to construct the forcing
     addTurbVelForces(geomdata,bx,time,der,rho);
}



































// Edward made me take this out

#if 0
     //
     // Original implementation using all 33 points in k-space.
     // May be fast enough on GPU.
     //

     auto const& dens = Scal.array(scalScomp);

     // Construct cell-centered forcing
     ParallelFor(bx, [ = ]
     AMREX_GPU_DEVICE (int i, int j, int k ) noexcept
     {
         Real z = xlo[2] + hz*(k-klo + 0.5);
         Real y = xlo[1] + hy*(j-jlo + 0.5);
         Real x = xlo[0] + hx*(i-ilo + 0.5);

         Real f1 = 0;
         Real f2 = 0;
         Real f3 = 0;

         // forcedata (and Array4) has column-major layout
         for (int kz = TurbulentForcing::mode_start*zstep; kz <= TurbulentForcing::nmodes*zstep; kz += zstep) {
             for (int ky = TurbulentForcing::mode_start*ystep; ky <= TurbulentForcing::nmodes*ystep; ky += ystep) {
                 for (int kx = TurbulentForcing::mode_start*xstep; kx <= TurbulentForcing::nmodes*xstep; kx += xstep)
                 {
                     Real kappa = sqrt( (kx*kx)/(Lx*Lx) + (ky*ky)/(Ly*Ly) + (kz*kz)/(Lz*Lz) );

                     if (kappa <= kappaMax)
                     {
                         Real xT = cos(FTX(kx,ky,kz)*forcetime + TAT(kx,ky,kz));

                         // if ( i==0 && j==0 && k==0 && kx==0 && ky==0 && kx==0){
                         //     printf("(0,0,0) : xT : %15.13e %15.13e %15.13e %15.13e\n",
                         //         FTX(kx,ky,kz), time, TAT(kx,ky,kz), xT);
                         //     Abort();
                         // }


                             f1 += xT *
                                 ( FAZ(kx,ky,kz)*TwoPi*(ky/Ly)
                                   *   sin(TwoPi*kx*x/Lx+FPZX(kx,ky,kz))
                                   *   cos(TwoPi*ky*y/Ly+FPZY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPZZ(kx,ky,kz))
                                   - FAY(kx,ky,kz)*TwoPi*(kz/Lz)
                                   *   sin(TwoPi*kx*x/Lx+FPYX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPYY(kx,ky,kz))
                                   *   cos(TwoPi*kz*z/Lz+FPYZ(kx,ky,kz)) );

                             f2 += xT *
                                 ( FAX(kx,ky,kz)*TwoPi*(kz/Lz)
                                   *   sin(TwoPi*kx*x/Lx+FPXX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPXY(kx,ky,kz))
                                   *   cos(TwoPi*kz*z/Lz+FPXZ(kx,ky,kz))
                                   - FAZ(kx,ky,kz)*TwoPi*(kx/Lx)
                                   *   cos(TwoPi*kx*x/Lx+FPZX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPZY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPZZ(kx,ky,kz)) );

                             f3 += xT *
                                 ( FAY(kx,ky,kz)*TwoPi*(kx/Lx)
                                   *   cos(TwoPi*kx*x/Lx+FPYX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPYY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPYZ(kx,ky,kz))
                                   - FAX(kx,ky,kz)*TwoPi*(ky/Ly)
                                   *   sin(TwoPi*kx*x/Lx+FPXX(kx,ky,kz))
                                   *   cos(TwoPi*ky*y/Ly+FPXY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPXZ(kx,ky,kz)) );
                     }
                 }
             }
         }

         //
         // For high aspect ratio domain, add more modes to break symmetry at a low level.
         // We assume Lz is longer, Lx = Ly.
         //
         for ( int kz = 1; kz <= zstep-1; kz++) {
             for ( int ky = TurbulentForcing::mode_start; ky <= TurbulentForcing::nmodes*ystep; ky++) {
                 for ( int kx = TurbulentForcing::mode_start; kx <= TurbulentForcing::nmodes*xstep; kx++)
                 {
                     Real kappa = sqrt( (kx*kx)/(Lx*Lx) + (ky*ky)/(Ly*Ly) + (kz*kz)/(Lz*Lz) );

                     if (kappa <= kappaMax)
                     {
                         Real xT = cos(FTX(kx,ky,kz)*forcetime + TAT(kx,ky,kz));

                             f1 += xT *
                                 ( FAZ(kx,ky,kz)*TwoPi*(ky/Ly)
                                   *   sin(TwoPi*kx*x/Lx+FPZX(kx,ky,kz))
                                   *   cos(TwoPi*ky*y/Ly+FPZY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPZZ(kx,ky,kz))
                                   - FAY(kx,ky,kz)*TwoPi*(kz/Lz)
                                   *   sin(TwoPi*kx*x/Lx+FPYX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPYY(kx,ky,kz))
                                   *   cos(TwoPi*kz*z/Lz+FPYZ(kx,ky,kz)) );

                             f2 += xT *
                                 ( FAX(kx,ky,kz)*TwoPi*(kz/Lz)
                                   *   sin(TwoPi*kx*x/Lx+FPXX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPXY(kx,ky,kz))
                                   *   cos(TwoPi*kz*z/Lz+FPXZ(kx,ky,kz))
                                   - FAZ(kx,ky,kz)*TwoPi*(kx/Lx)
                                   *   cos(TwoPi*kx*x/Lx+FPZX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPZY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPZZ(kx,ky,kz)) );

                             f3 += xT *
                                 ( FAY(kx,ky,kz)*TwoPi*(kx/Lx)
                                   *   cos(TwoPi*kx*x/Lx+FPYX(kx,ky,kz))
                                   *   sin(TwoPi*ky*y/Ly+FPYY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPYZ(kx,ky,kz))
                                   - FAX(kx,ky,kz)*TwoPi*(ky/Ly)
                                   *   sin(TwoPi*kx*x/Lx+FPXX(kx,ky,kz))
                                   *   cos(TwoPi*ky*y/Ly+FPXY(kx,ky,kz))
                                   *   sin(TwoPi*kz*z/Lz+FPXZ(kx,ky,kz)) );
                     }
                 }
             }
         }

         frc(i,j,k,0) += dens(i,j,k,0) * f1;
         frc(i,j,k,1) += dens(i,j,k,0) * f2;
         frc(i,j,k,2) += dens(i,j,k,0) * f3;

         // if ( i==0 && j==0 && k==0 ){
         //   printf("(0,0,0) : %15.13e %15.13e %15.13e\n",
         //          f1, f2, f3);
         // }
         // if ( i==16 && j==16 && k==16 ){
         //   printf("(16,16,16) : %15.13e %15.13e %15.13e\n",
         //          f1, f2, f3);
         // }
         // if ( i==25 && j==12 && k==3 ){
         //   printf("(25,12,3) : %15.13e %15.13e %15.13e\n",
         //          f1, f2, f3);
         // }

     });
#endif // if 0

