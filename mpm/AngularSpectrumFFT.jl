#= ====================================================================================================
    library: AngularSpectrumFFT.jl
        - author:  Konan Yagasaki (Kyoto Fusioneering Ltd.)
        - version: beta (2025-05-12)
    ---------------------------------------------------------------------------------------------------
    library for anglur-spectrum propagation by FFTW
==================================================================================================== =#
module AngularSpectrumFFT

using FFTW, LinearAlgebra

export propagate_angular_spectrum

function propagate_angular_spectrum(u::Array{ComplexF64,2}, dz::Float64, λ::Float64, dx::Float64,
                                    θx::Float64, θy::Float64)
"""
    angluar spectrum propagation
    dz is positive only when forward propagation
"""
    N, M = size(u)
    k  = 2π / λ                       # wavelength

    # only for even number ...
    #fx = (-N÷2 : N÷2-1) ./ (N*dx)       # fx
    #fy = (-M÷2 : M÷2-1) ./ (M*dx)       # fy

    fx = (-fld(N,2) : cld(N,2)-1) ./ (N*dx)
    fy = (-fld(M,2) : cld(M,2)-1) ./ (M*dx)

    FX = reshape(fx, N, 1)
    FY = reshape(fy, 1, M)
    
    FZsq = 1 .- (λ .* FX).^2 .- (λ .* FY).^2
    FZsq[FZsq .< 0] .= 0.0
    FZ = sqrt.(FZsq)                   # propagation constan
    
    ## propagation function
    H = @. exp(1im * k * dz * FZ)

    ## consider tilt shift
    Δx = θx * dz
    Δy = θy * dz
    shift = @. exp(-2π*im * (FX*Δx + FY*Δy))

    U = fftshift(fft(u))
    return ifft(ifftshift(U .* H .* shift))
end

end