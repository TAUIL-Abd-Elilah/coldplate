! Copyright 2026 Coldplate contributors.
! SPDX-License-Identifier: Apache-2.0
!
! Residual of the steady advection-diffusion equation for temperature:
!
!     div(u T) - div(k grad T) - source = 0
!
! This is the same physics as the JAX thermal Tesseract, written independently
! in Fortran. Nothing here computes a derivative: Enzyme differentiates this
! routine at the LLVM IR level to produce exact JVPs and VJPs, so the whole
! component's derivative information comes from compiler AD rather than from
! an AD library or from hand-written adjoint code.
!
! Arrays are flat and indexed row-major to match numpy exactly (Fortran is
! column-major by default, so every index is computed explicitly):
!     T, k  : (Ny, Nx)     idx = j*Nx + i + 1
!     u     : (Ny, Nx+1)   idx = j*(Nx+1) + i + 1
!     v     : (Ny+1, Nx)   idx = j*Nx + i + 1
!
! Advection uses the same smooth Peclet-weighted face value as the JAX block:
! upwind for large |Pe|, central for small. A hard upwind switch would be
! non-differentiable wherever a face velocity crosses zero, which is fatal for
! both the Newton solve and the gradient.

! bind(C) fixes the exported symbol name to exactly `thermal_residual`.
! Without it flang applies the usual Fortran mangling and emits
! `thermal_residual_`, so the C wrapper's declaration resolves to a different,
! bodyless symbol -- and Enzyme then reports "failed to find fn to
! differentiate ... declare void @thermal_residual", because after linking
! there is a declaration but no definition to differentiate. Arguments stay
! pass-by-reference (no `value` attribute), which is what the wrapper expects.
subroutine thermal_residual(Nx, Ny, T, u, v, k, q_chip, chip_frac, &
                            bc_mode, t_hot, R) &
    bind(C, name="thermal_residual")
  use iso_c_binding, only: c_double, c_int
  implicit none

  ! Bind straight to libm's tanh instead of using the Fortran intrinsic.
  ! LFortran lowers `tanh(x)` to a call into its own runtime, `_lfortran_dtanh`,
  ! which Enzyme sees as an opaque external symbol and refuses to
  ! differentiate ("No forward mode derivative found"). Enzyme does carry a
  ! rule for libm `tanh`, so naming it explicitly makes the call
  ! differentiable while leaving the arithmetic identical to the JAX block.
  interface
    pure function c_tanh(x) bind(C, name="tanh")
      import :: c_double
      real(c_double), value :: x
      real(c_double) :: c_tanh
    end function c_tanh
  end interface

  integer(c_int), intent(in) :: Nx, Ny
  real(c_double), intent(in) :: T(Nx*Ny), u(Ny*(Nx+1)), v((Ny+1)*Nx), k(Nx*Ny)
  real(c_double), intent(in) :: q_chip, chip_frac, bc_mode, t_hot
  real(c_double), intent(out) :: R(Nx*Ny)

  integer :: i, j, c
  real(c_double) :: h, lo, hi
  real(c_double) :: fe, fw, fn, fs, qe, qw, qn, qs
  real(c_double) :: kf, uu, vv, w, TL, TR, TB, TT, mask

  h = 1.0d0 / dble(Nx)
  lo = 0.5d0 * (1.0d0 - chip_frac) * dble(Nx)
  hi = 0.5d0 * (1.0d0 + chip_frac) * dble(Nx)

  do j = 0, Ny - 1
    do i = 0, Nx - 1
      c = j*Nx + i + 1

      ! ---- east x-face, index i+1 ----
      if (i + 1 == Nx) then
        kf = k(j*Nx + i + 1)
      else
        kf = 0.5d0 * (k(j*Nx + i + 1) + k(j*Nx + i + 2))
      end if
      uu = u(j*(Nx+1) + i + 2)
      w = 0.5d0 * (1.0d0 + c_tanh(0.5d0 * uu * h / max(kf, 1.0d-12)))
      TL = T(j*Nx + i + 1)
      if (i + 1 == Nx) then
        TR = T(j*Nx + Nx)
      else
        TR = T(j*Nx + i + 2)
      end if
      fe = uu * (w * TL + (1.0d0 - w) * TR)
      if (i + 1 == Nx) then
        qe = 0.0d0
      else
        qe = -kf * (T(j*Nx + i + 2) - T(j*Nx + i + 1)) / h
      end if

      ! ---- west x-face, index i ----
      if (i == 0) then
        kf = k(j*Nx + 1)
      else
        kf = 0.5d0 * (k(j*Nx + i) + k(j*Nx + i + 1))
      end if
      uu = u(j*(Nx+1) + i + 1)
      w = 0.5d0 * (1.0d0 + c_tanh(0.5d0 * uu * h / max(kf, 1.0d-12)))
      if (i == 0) then
        TL = T(j*Nx + 1)
      else
        TL = T(j*Nx + i)
      end if
      TR = T(j*Nx + i + 1)
      fw = uu * (w * TL + (1.0d0 - w) * TR)
      if (i == 0) then
        qw = 0.0d0
      else
        qw = -kf * (T(j*Nx + i + 1) - T(j*Nx + i)) / h
      end if

      ! ---- north y-face, index j+1 ----
      if (j + 1 == Ny) then
        kf = k(j*Nx + i + 1)
      else
        kf = 0.5d0 * (k(j*Nx + i + 1) + k((j+1)*Nx + i + 1))
      end if
      vv = v((j+1)*Nx + i + 1)
      w = 0.5d0 * (1.0d0 + c_tanh(0.5d0 * vv * h / max(kf, 1.0d-12)))
      TB = T(j*Nx + i + 1)
      if (j + 1 == Ny) then
        TT = T((Ny-1)*Nx + i + 1)
      else
        TT = T((j+1)*Nx + i + 1)
      end if
      fn = vv * (w * TB + (1.0d0 - w) * TT)
      if (j + 1 == Ny) then
        ! cold sink at T = 0, half a cell away
        qn = 2.0d0 * k((Ny-1)*Nx + i + 1) * T((Ny-1)*Nx + i + 1) / h
      else
        qn = -kf * (T((j+1)*Nx + i + 1) - T(j*Nx + i + 1)) / h
      end if

      ! ---- south y-face, index j ----
      if (j == 0) then
        kf = k(i + 1)
      else
        kf = 0.5d0 * (k((j-1)*Nx + i + 1) + k(j*Nx + i + 1))
      end if
      vv = v(j*Nx + i + 1)
      w = 0.5d0 * (1.0d0 + c_tanh(0.5d0 * vv * h / max(kf, 1.0d-12)))
      if (j == 0) then
        TB = T(i + 1)
      else
        TB = T((j-1)*Nx + i + 1)
      end if
      TT = T(j*Nx + i + 1)
      fs = vv * (w * TB + (1.0d0 - w) * TT)
      if (j == 0) then
        if (bc_mode > 0.5d0) then
          ! Rayleigh-Benard: isothermal hot wall half a cell below the centre
          qs = -k(i + 1) * (T(i + 1) - t_hot) / (0.5d0 * h)
        else
          ! cold plate: chip heat flux entering through the bottom wall
          mask = 0.0d0
          if (dble(i) >= lo .and. dble(i) < hi) mask = 1.0d0
          qs = q_chip * mask
        end if
      else
        qs = -kf * (T(j*Nx + i + 1) - T((j-1)*Nx + i + 1)) / h
      end if

      R(c) = (fe - fw) / h + (fn - fs) / h + (qe - qw) / h + (qn - qs) / h
    end do
  end do
end subroutine
