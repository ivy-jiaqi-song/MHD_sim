//========================================================================================
// Athena++ problem generator for a Kolmogorov-flow hydrodynamic sanity check.
//
// This is kept outside the Athena++ checkout and copied into an ignored working tree by
// backends/athena/run_athena_kolmogorov_hd.py. The upstream Athena++ checkout is not
// modified in place.
//========================================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

namespace {

Real force_amplitude;
Real force_mode_y;
Real initial_velocity_rms;
Real iso_sound_speed;
std::string initial_condition;
int initial_modes;
int random_seed;

Real UnitPhase(int seed, int mx, int my, int which) {
  Real x = std::sin(static_cast<Real>(seed * 12 + mx * 78 + my * 37 + which * 19)
                    * 12.9898) * 43758.5453;
  return x - std::floor(x);
}

Real SignedCoeff(int seed, int mx, int my) {
  return UnitPhase(seed, mx, my, 0) - 0.5;
}

Real CenteredNoise(int seed, int i, int j, int which) {
  return 2.0 * UnitPhase(seed, i, j, which) - 1.0;
}

Real SineForce(MeshBlock *pmb, Real x2) {
  const Real y_min = pmb->pmy_mesh->mesh_size.x2min;
  const Real ly = pmb->pmy_mesh->mesh_size.x2max - y_min;
  return force_amplitude * std::sin(2.0 * PI * force_mode_y * (x2 - y_min) / ly);
}

void KolmogorovForcing(MeshBlock *pmb, const Real time, const Real dt,
                       const AthenaArray<Real> &prim,
                       const AthenaArray<Real> &prim_scalar,
                       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                       AthenaArray<Real> &cons_scalar) {
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      const Real fx = SineForce(pmb, pmb->pcoord->x2v(j));
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real rho = prim(IDN, k, j, i);
        cons(IM1, k, j, i) += dt * rho * fx;
        if (NON_BAROTROPIC_EOS) {
          cons(IEN, k, j, i) += dt * rho * prim(IVX, k, j, i) * fx;
        }
      }
    }
  }
}

Real HistoryEnstrophy(MeshBlock *pmb, int iout) {
  const Real lx = pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min;
  const Real ly = pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min;
  const Real dx = lx / pmb->pmy_mesh->mesh_size.nx1;
  const Real dy = ly / pmb->pmy_mesh->mesh_size.nx2;
  AthenaArray<Real> &w = pmb->phydro->w;
  AthenaArray<Real> volume;
  volume.NewAthenaArray(pmb->ncells1);

  Real sum = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real dvydx = (w(IVY, k, j, i + 1) - w(IVY, k, j, i - 1)) / (2.0 * dx);
        const Real dvxdy = (w(IVX, k, j + 1, i) - w(IVX, k, j - 1, i)) / (2.0 * dy);
        const Real omega = dvydx - dvxdy;
        sum += volume(i) * 0.5 * SQR(omega);
      }
    }
  }
  return sum;
}

Real HistoryDivergenceSquared(MeshBlock *pmb, int iout) {
  const Real lx = pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min;
  const Real ly = pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min;
  const Real dx = lx / pmb->pmy_mesh->mesh_size.nx1;
  const Real dy = ly / pmb->pmy_mesh->mesh_size.nx2;
  AthenaArray<Real> &w = pmb->phydro->w;
  AthenaArray<Real> volume;
  volume.NewAthenaArray(pmb->ncells1);

  Real sum = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real dvxdx = (w(IVX, k, j, i + 1) - w(IVX, k, j, i - 1)) / (2.0 * dx);
        const Real dvydy = (w(IVY, k, j + 1, i) - w(IVY, k, j - 1, i)) / (2.0 * dy);
        sum += volume(i) * SQR(dvxdx + dvydy);
      }
    }
  }
  return sum;
}

Real HistoryForcingPower(MeshBlock *pmb, int iout) {
  AthenaArray<Real> &w = pmb->phydro->w;
  AthenaArray<Real> volume;
  volume.NewAthenaArray(pmb->ncells1);

  Real sum = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
      const Real fx = SineForce(pmb, pmb->pcoord->x2v(j));
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        sum += volume(i) * w(IDN, k, j, i) * w(IVX, k, j, i) * fx;
      }
    }
  }
  return sum;
}

Real HistoryMaxVelocitySquared(MeshBlock *pmb, int iout) {
  AthenaArray<Real> &w = pmb->phydro->w;
  Real max_speed2 = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        max_speed2 = std::max(max_speed2,
                              SQR(w(IVX, k, j, i)) + SQR(w(IVY, k, j, i))
                                  + SQR(w(IVZ, k, j, i)));
      }
    }
  }
  return max_speed2;
}

Real HistoryDensityMin(MeshBlock *pmb, int iout) {
  AthenaArray<Real> &w = pmb->phydro->w;
  Real rho_min = std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        rho_min = std::min(rho_min, w(IDN, k, j, i));
      }
    }
  }
  return rho_min;
}

Real HistoryDensityMax(MeshBlock *pmb, int iout) {
  AthenaArray<Real> &w = pmb->phydro->w;
  Real rho_max = -std::numeric_limits<Real>::max();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        rho_max = std::max(rho_max, w(IDN, k, j, i));
      }
    }
  }
  return rho_max;
}

Real HistoryDensitySquared(MeshBlock *pmb, int iout) {
  AthenaArray<Real> &w = pmb->phydro->w;
  AthenaArray<Real> volume;
  volume.NewAthenaArray(pmb->ncells1);

  Real sum = 0.0;
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      pmb->pcoord->CellVolume(k, j, pmb->is, pmb->ie, volume);
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        sum += volume(i) * SQR(w(IDN, k, j, i));
      }
    }
  }
  return sum;
}

Real HistoryMaxMach(MeshBlock *pmb, int iout) {
  AthenaArray<Real> &w = pmb->phydro->w;
  Real max_mach = 0.0;
  const Real cs = std::max(iso_sound_speed, std::numeric_limits<Real>::min());
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        const Real speed = std::sqrt(SQR(w(IVX, k, j, i)) + SQR(w(IVY, k, j, i))
                                     + SQR(w(IVZ, k, j, i)));
        max_mach = std::max(max_mach, speed / cs);
      }
    }
  }
  return max_mach;
}

} // namespace

void Mesh::InitUserMeshData(ParameterInput *pin) {
  if (MAGNETIC_FIELDS_ENABLED) {
    std::stringstream msg;
    msg << "### FATAL ERROR in kolmogorov_hd.cpp" << std::endl
        << "Configure Athena++ without magnetic fields for this HD sanity check."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  force_amplitude = pin->GetOrAddReal("problem", "force_amplitude", 0.1);
  force_mode_y = pin->GetOrAddReal("problem", "force_mode_y", 2.0);
  initial_condition = pin->GetOrAddString("problem", "initial_condition", "fourier_divfree");
  initial_velocity_rms = pin->GetOrAddReal("problem", "initial_velocity_rms", 1.0e-3);
  iso_sound_speed = pin->GetOrAddReal("hydro", "iso_sound_speed", 10.0);
  initial_modes = pin->GetOrAddInteger("problem", "initial_modes", 4);
  random_seed = pin->GetOrAddInteger("problem", "seed", 1234);

  AllocateUserHistoryOutput(8);
  EnrollUserHistoryOutput(0, HistoryEnstrophy, "enstrophy");
  EnrollUserHistoryOutput(1, HistoryDivergenceSquared, "div2");
  EnrollUserHistoryOutput(2, HistoryForcingPower, "forcepow");
  EnrollUserHistoryOutput(3, HistoryMaxVelocitySquared, "maxvel2",
                          UserHistoryOperation::max);
  EnrollUserHistoryOutput(4, HistoryDensityMin, "rho_min", UserHistoryOperation::min);
  EnrollUserHistoryOutput(5, HistoryDensityMax, "rho_max", UserHistoryOperation::max);
  EnrollUserHistoryOutput(6, HistoryDensitySquared, "rho2");
  EnrollUserHistoryOutput(7, HistoryMaxMach, "maxmach", UserHistoryOperation::max);
  EnrollUserExplicitSourceFunction(KolmogorovForcing);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  const Real density = pin->GetOrAddReal("problem", "density", 1.0);
  const Real pressure = pin->GetOrAddReal("problem", "pressure", 1.0);
  const Real gamma = peos->GetGamma();
  const Real gm1 = gamma - 1.0;
  const Real lx = pmy_mesh->mesh_size.x1max - pmy_mesh->mesh_size.x1min;
  const Real ly = pmy_mesh->mesh_size.x2max - pmy_mesh->mesh_size.x2min;
  Real sum_vx = 0.0;
  Real sum_vy = 0.0;
  int ncell = 0;
  bool use_grid_noise = false;
  bool use_fourier_divfree = false;

  if (initial_condition == "grid_noise" || initial_condition == "grid-noise"
      || initial_condition == "noise") {
    use_grid_noise = true;
  } else if (initial_condition == "fourier_divfree"
             || initial_condition == "fourier-divfree"
             || initial_condition == "divfree"
             || initial_condition == "divergence_free") {
    use_fourier_divfree = true;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in kolmogorov_hd.cpp ProblemGenerator" << std::endl
        << "initial_condition must be 'fourier_divfree' or 'grid_noise'; got '"
        << initial_condition << "'" << std::endl;
    ATHENA_ERROR(msg);
  }

  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        const Real x = pcoord->x1v(i) - pmy_mesh->mesh_size.x1min;
        const Real y = pcoord->x2v(j) - pmy_mesh->mesh_size.x2min;
        Real vx = 0.0;
        Real vy = 0.0;

        if (use_grid_noise) {
          int noise_i = static_cast<int>(std::floor((x / lx) * pmy_mesh->mesh_size.nx1)) + 1;
          int noise_j = static_cast<int>(std::floor((y / ly) * pmy_mesh->mesh_size.nx2)) + 1;
          noise_i = std::max(1, std::min(noise_i, pmy_mesh->mesh_size.nx1));
          noise_j = std::max(1, std::min(noise_j, pmy_mesh->mesh_size.nx2));
          vx = CenteredNoise(random_seed, noise_i, noise_j, 1);
          vy = CenteredNoise(random_seed, noise_i, noise_j, 2);
        } else if (use_fourier_divfree) {
          for (int mx = 1; mx <= initial_modes; ++mx) {
            for (int my = 1; my <= initial_modes; ++my) {
              const Real kx = 2.0 * PI * static_cast<Real>(mx) / lx;
              const Real ky = 2.0 * PI * static_cast<Real>(my) / ly;
              const Real phase_x = 2.0 * PI * UnitPhase(random_seed, mx, my, 1);
              const Real phase_y = 2.0 * PI * UnitPhase(random_seed, mx, my, 2);
              const Real coeff = SignedCoeff(random_seed, mx, my)
                                 / std::sqrt(static_cast<Real>(mx * mx + my * my));
              vx += coeff * ky * std::sin(kx * x + phase_x) * std::cos(ky * y + phase_y);
              vy -= coeff * kx * std::cos(kx * x + phase_x) * std::sin(ky * y + phase_y);
            }
          }
        }

        phydro->u(IDN, k, j, i) = density;
        phydro->u(IM1, k, j, i) = density * vx;
        phydro->u(IM2, k, j, i) = density * vy;
        phydro->u(IM3, k, j, i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i) = pressure / gm1 + 0.5 * density * (SQR(vx) + SQR(vy));
        }
        sum_vx += vx;
        sum_vy += vy;
        ncell += 1;
      }
    }
  }

  const Real mean_vx = ncell > 0 ? sum_vx / ncell : 0.0;
  const Real mean_vy = ncell > 0 ? sum_vy / ncell : 0.0;
  Real sum_speed2 = 0.0;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        Real vx = phydro->u(IM1, k, j, i) / phydro->u(IDN, k, j, i) - mean_vx;
        Real vy = phydro->u(IM2, k, j, i) / phydro->u(IDN, k, j, i) - mean_vy;
        phydro->u(IM1, k, j, i) = phydro->u(IDN, k, j, i) * vx;
        phydro->u(IM2, k, j, i) = phydro->u(IDN, k, j, i) * vy;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN, k, j, i) = pressure / gm1 + 0.5 * density * (SQR(vx) + SQR(vy));
        }
        sum_speed2 += SQR(vx) + SQR(vy);
      }
    }
  }

  const Real rms = std::sqrt(sum_speed2 / std::max(ncell, 1));
  const Real scale = (rms > 0.0) ? initial_velocity_rms / rms : 0.0;
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        phydro->u(IM1, k, j, i) *= scale;
        phydro->u(IM2, k, j, i) *= scale;
        if (NON_BAROTROPIC_EOS) {
          const Real vx = phydro->u(IM1, k, j, i) / phydro->u(IDN, k, j, i);
          const Real vy = phydro->u(IM2, k, j, i) / phydro->u(IDN, k, j, i);
          phydro->u(IEN, k, j, i) = pressure / gm1 + 0.5 * density * (SQR(vx) + SQR(vy));
        }
      }
    }
  }
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  return;
}

void MeshBlock::UserWorkInLoop() {
  return;
}

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  return;
}
