#ifndef PELELM_UTILS_H
#define PELELM_UTILS_H

#include <utility>
#include <AMReX_MultiFabUtil.H>
#ifdef AMREX_USE_EB
#include <AMReX_EBMultiFabUtil.H>
#endif

template <class T>
amrex::Gpu::DeviceVector<T>
convertToDeviceVector(const amrex::Vector<T>& v)
{
  static_assert(
    std::is_trivially_copyable_v<T>, "T must be trivially copyable");
  int ncomp = v.size();
  amrex::Gpu::DeviceVector<T> v_d(ncomp);
#ifdef AMREX_USE_GPU
  amrex::Gpu::htod_memcpy(v_d.data(), v.data(), sizeof(T) * ncomp);
#else
  std::memcpy(v_d.data(), v.data(), sizeof(T) * ncomp);
#endif
  return v_d;
}

// average_down / average_down_faces / set_covered that use the EB-aware
// versions in EB builds. eb_set_covered is a no-op without EB.
template <typename... Args>
void
eb_average_down(Args&&... args)
{
#ifdef AMREX_USE_EB
  amrex::EB_average_down(std::forward<Args>(args)...);
#else
  amrex::average_down(std::forward<Args>(args)...);
#endif
}

template <typename... Args>
void
eb_average_down_faces(Args&&... args)
{
#ifdef AMREX_USE_EB
  amrex::EB_average_down_faces(std::forward<Args>(args)...);
#else
  amrex::average_down_faces(std::forward<Args>(args)...);
#endif
}

template <typename... Args>
void
eb_set_covered([[maybe_unused]] Args&&... args)
{
#ifdef AMREX_USE_EB
  amrex::EB_set_covered(std::forward<Args>(args)...);
#endif
}

#endif
