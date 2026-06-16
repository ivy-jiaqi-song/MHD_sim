// Athena++ problem generator for this repository's MHDFlows-style comparison run.
//
// This file is repo-owned glue code. The runner copies it into a disposable
// Athena++ build tree as src/pgen/mhdflows_turbulence.cpp before configuring
// Athena with --prob=mhdflows_turbulence.

#include <algorithm>
#include <cmath>
#include <sstream>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

#if !MAGNETIC_FIELDS_ENABLED
#error "mhdflows_turbulence requires Athena++ magnetic fields; configure with -b"
#endif

namespace {

Real sqr(Real x) {
  return x*x;
}

} // namespace

void Mesh::InitUserMeshData(ParameterInput *pin) {
  return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real sound_speed = pin->GetOrAddReal("problem", "sound_speed", 1.0);
  const Real pressure = pin->GetOrAddReal("problem", "pressure",
                                          rho0*sound_speed*sound_speed);

  const Real bx0 = pin->GetOrAddReal("problem", "mean_field_x", 1.0);
  const Real by0 = pin->GetOrAddReal("problem", "mean_field_y", 0.0);
  const Real bz0 = pin->GetOrAddReal("problem", "mean_field_z", 0.0);

  const Real initial_velocity_power =
      pin->GetOrAddReal("problem", "initial_velocity_power", 5.0e-4);
  const Real initial_velocity_amplitude =
      pin->GetOrAddReal("problem", "initial_velocity_amplitude",
                        std::sqrt(std::max(static_cast<Real>(0.0),
                                           initial_velocity_power)));
  const Real initial_velocity_wavenumber =
      pin->GetOrAddReal("problem", "initial_velocity_wavenumber", 1.0);

  const Real x1min = pmy_mesh->mesh_size.x1min;
  const Real x1max = pmy_mesh->mesh_size.x1max;
  const Real box_size = x1max - x1min;
  if (box_size <= 0.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in mhdflows_turbulence ProblemGenerator" << std::endl
        << "x1 domain has non-positive size" << std::endl;
    ATHENA_ERROR(msg);
  }

  const Real k0 = 2.0*PI*initial_velocity_wavenumber/box_size;

  for (int k=ks; k<=ke; ++k) {
    const Real z = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      const Real y = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        const Real x = pcoord->x1v(i);

        // A compact divergence-free velocity seed. Athena's turbulence driver
        // supplies the ongoing stochastic forcing after initialization.
        const Real ux = initial_velocity_amplitude*(std::cos(k0*y) - std::cos(k0*z));
        const Real uy = initial_velocity_amplitude*(std::cos(k0*z) - std::cos(k0*x));
        const Real uz = initial_velocity_amplitude*(std::cos(k0*x) - std::cos(k0*y));

        phydro->u(IDN,k,j,i) = rho0;
        phydro->u(IM1,k,j,i) = rho0*ux;
        phydro->u(IM2,k,j,i) = rho0*uy;
        phydro->u(IM3,k,j,i) = rho0*uz;

        if (NON_BAROTROPIC_EOS) {
          const Real gm1 = peos->GetGamma() - 1.0;
          phydro->u(IEN,k,j,i) = pressure/gm1
              + 0.5*rho0*(sqr(ux) + sqr(uy) + sqr(uz))
              + 0.5*(sqr(bx0) + sqr(by0) + sqr(bz0));
        }
      }
    }
  }

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie+1; ++i) {
        pfield->b.x1f(k,j,i) = bx0;
      }
    }
  }
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
      for (int i=is; i<=ie; ++i) {
        pfield->b.x2f(k,j,i) = by0;
      }
    }
  }
  for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        pfield->b.x3f(k,j,i) = bz0;
      }
    }
  }

  pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord,
                                     is, ie, js, je, ks, ke);
  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  return;
}
