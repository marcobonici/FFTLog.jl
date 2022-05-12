
"""
    mutable struct SingleBesselPlan{T,C} <: AbstractPlan

This struct contains all the elements necessary to evaluate the integral 
with one Bessel function.
All the arguments of this struct are keyword arguments. Here we show 
the compelte list and their default values:

- `x::Vector{T}` : the LOGARITHMICALLY SPACED vector of x-axis values. You
  need always to provide this vector.

- `y::Matrix{T} = zeros(10, 10)` : the logarithmically spaced vector of the values
  where the transformed function will be evaluated. It has the same length of `x`

- `fy::Matrix{T} = zeros(10, 10)` : the y-axis of the transformed function; it is a vector
  if only one Bessel function order is provided in the functions

- `hm::Matrix{C} = zeros(ComplexF64, 10, 10)` : matrix of the coefficients
  ``h_m = c_m \\, h_{m, \\mathrm{corr}} \\, g_\\ell``, where ``c_m``s,  
  ``h_{m, \\mathrm{corr}}``s and ``g_\\ell`` are respectively stored in
  `plan.cm`, `plan.hm_corr` and `plan.gl`.
  Each column contains all the ``h_m``s for a given spherical Bessel order ``\\ell``. 

- `hm_corr::Matrix{C} = zeros(ComplexF64, 10, 10)` : matrix of the coefficients
  ``h_{m, \\mathrm{corr}} = (x_0 y_0)^{- i \\eta_m}``, where ``\\eta_m = \\frac{2 \\pi m}{N \\, \\Delta_{\\ln x}}``
  and ``x_0`` and ``y_0`` are the smallest values of `plan.x` and `plan.y`, respectively. 
  Each column contains all the ``h_{m, \\mathrm{corr}}``s for a given spherical Bessel order ``\\ell``. 

- `d_ln_x::T = log(x[2] / x[1])` : the spacing between the `x` elements.

- `fy_corr::Matrix{T} = zeros(10, 10)` : matrix of the coefficients
  ``K(y) = \\frac{\\sqrt{\\pi}}{4 y^{\\nu}}``, where ``\\nu`` is the bias paremeter
  stored in `plan.ν`.
  Each column contains all the ``h_{m, \\mathrm{corr}}``s for a given spherical Bessel order ``\\ell``. 


- `original_length::Int = length(x)` : the original inpout length of the `x` vector; 
  it is stored because, for numerical stability purposes, during the computation this
  vector is expanded at the edged, and so the input function ones. 

- `gl::Matrix{C} = zeros(ComplexF64, 100, 100)` : vector with the ``g_\\ell`` values
  for all the input spherical Bessel order.

- `ν::T = 1.01` : bias parameter.

- `n_extrap_low::Int = 0` : number of points to concatenate on the left
  of `x`, logarithmically distributed with the same ratio of the left-edge
  elements of `x`

- `n_extrap_high::Int = 0` : number of points to concatenate on the right
  of `x`, logarithmically distributed with the same ratio of the right-edge
  elements of `x`

- `c_window_width::T = 0.25` : position where the tapering by the window function 
  begins; by default `c_window_width= 0.25`, so is begins when 
  ``m = \\pm 0.75 \\times N/2``, where ``N`` is the size of the input array.

- `n_pad::Int = 0` : number of zeros to be concatenated both on the left and
  on the right of the input function.

- `n::Int = 0` : the derivative order for the spherical Bessel function.

- `N::Int = original_length + n_extrap_low + n_extrap_high + 2 * n_pad` : number
  of points where the input function is known; are considered both the "true values"
  and the fake ones, added for a more numerically stable fft.  

- `m::Vector{T} = zeros(N)` : vector with all the indexes that will be used for the
  power-law expansion of the input function

- `cm::Vector{C} = zeros(ComplexF64, N)` : vector containing all the input function 
  power-law exapnsion ``c_m`` coefficients.

- `ηm::Vector{T} = zeros(N)` : vector of all the 
  ``\\eta_m = \\frac{2 \\pi m}{N \\, \\Delta_{\\ln x}} `` coefficients.

- `plan_rfft::FFTW.rFFTWPlan = plan_rfft(randn(1024))` : a random initialized 
  fft plan of [`FFTW`](@ref)

- `plan_irfft = 
        plan_irfft(
            randn(Complex{Float64}, 2, Int((original_length + n_extrap_low + n_extrap_high + 2 * n_pad) / 2) + 1),
            original_length + n_extrap_low + n_extrap_high + 2 * n_pad, 
            2
        )` : 

See also: [`AbstractPlan`](@ref)
"""
@kwdef mutable struct SingleBesselPlan{T,C} <: AbstractPlan
    x::Vector{T}
    y::Matrix{T} = zeros(10, 10)
    fy::Matrix{T} = zeros(10, 10)
    hm::Matrix{C} = zeros(ComplexF64, 10, 10)
    hm_corr::Matrix{C} = zeros(ComplexF64, 10, 10)
    d_ln_x::T = log(x[2] / x[1])
    fy_corr::Matrix{T} = zeros(10, 10)
    original_length::Int = length(x)
    gl::Matrix{C} = zeros(ComplexF64, 100, 100)
    ν::T = 1.01
    n_extrap_low::Int = 0
    n_extrap_high::Int = 0
    c_window_width::T = 0.25
    n_pad::Int = 0
    n::Int = 0
    N::Int = original_length + n_extrap_low + n_extrap_high + 2 * n_pad
    m::Vector{T} = zeros(N)
    cm::Vector{C} = zeros(ComplexF64, N)
    ηm::Vector{T} = zeros(N)
    plan_rfft::FFTW.rFFTWPlan = plan_rfft(randn(1024))
    plan_irfft =
        plan_irfft(
            randn(Complex{Float64}, 2, Int((original_length + n_extrap_low + n_extrap_high + 2 * n_pad) / 2) + 1),
            original_length + n_extrap_low + n_extrap_high + 2 * n_pad,
            2
        )
end


"""
    mutable struct HankelPlan{T,C} <: AbstractPlan

A specific type of FFTLogPlan designed for the Hankel transform.
Its arguments are the same of `SingleBesselPlan`, checks its documentation for
more information.

See also: [`SingleBesselPlan`](@ref), [`AbstractPlan`](@ref)
"""
@kwdef mutable struct HankelPlan{T,C} <: AbstractPlan
    x::Vector{T}
    y::Matrix{T} = zeros(10, 10)
    fy::Matrix{T} = zeros(10, 10)
    hm::Matrix{C} = zeros(ComplexF64, 10, 10)
    hm_corr::Matrix{C} = zeros(ComplexF64, 10, 10)
    d_ln_x::T = log(x[2] / x[1])
    fy_corr::Matrix{T} = zeros(10, 10)
    original_length::Int = length(x)
    gl::Matrix{C} = zeros(ComplexF64, 100, 100)
    ν::T = 1.01
    n_extrap_low::Int = 0
    n_extrap_high::Int = 0
    c_window_width::T = 0.25
    n_pad::Int = 0
    n::Int = 0
    N::Int = original_length + n_extrap_low + n_extrap_high + 2 * n_pad
    m::Vector{T} = zeros(N)
    cm::Vector{C} = zeros(ComplexF64, N)
    ηm::Vector{T} = zeros(N)
    plan_rfft::FFTW.rFFTWPlan = plan_rfft(randn(1024))
    plan_irfft = plan_irfft(randn(Complex{Float64}, 2,
            Int((original_length + n_extrap_low + n_extrap_high + 2 * n_pad) / 2) + 1),
        original_length + n_extrap_low + n_extrap_high + 2 * n_pad, 2)
end



##########################################################################################92




"""
    _eval_gl(ell, z::Vector, n::Int )::Vector

Evaluate the ``g_{\\ell}`` coefficients, defined as
```math
g_{\\ell}^{(n)}(z) = (-1)^n \\, 2^{z-n} \\, \\frac{
        \\Gamma\\left(\\frac{\\ell + z - n}{2}\\right)
    }{
        \\Gamma\\left(\\frac{3 + \\ell + n - z}{2}\\right)
    }
```
"""
function _eval_gl(ell, z::Vector, n::Int)
    gl = ((-1)^n) .* 2 .^ (z .- n) .* gamma.((ell .+ z .- n) / 2) ./
         gamma.((3 .+ ell .+ n .- z) / 2)
    if n != 0
        for i in 1:n
            gl .*= (z .- i)
        end
    end
    return gl
end



"""
    _eval_y!(plan::Union{SingleBesselPlan, HankelPlan}, ell::Vector)

Given an input `plan::Union{SingleBesselPlan, HankelPlan}`, compute the `y` values where the output 
function will be evaluated and the coefficient ``K(y)`` outside the IFFT. They
are, respectively:
```math
y = \\frac{\\ell + 1}{x} \\; , \\quad\\quad K(y) = \\frac{\\sqrt{\\pi}}{4 y^{\\nu}}
```
The vector of their values are stored respectively in `plan.y` and `plan.fy_corr`,
and nothing is returned.

See also: [`SingleBesselPlan`](@ref), [`HankelPlan`](@ref)
"""
function _eval_y!(plan::Union{SingleBesselPlan, HankelPlan}, ell::Vector)

    plan.y = zeros(length(ell), length(plan.x))
    plan.fy_corr = zeros(size(plan.y))
    plan.fy = zeros(size(plan.y))

    reverse!(plan.x)

    @inbounds for myl in 1:length(ell)
        plan.y[myl, :] = (ell[myl] + 1) ./ plan.x
        plan.fy_corr[myl, :] =
            plan.y[myl, :] .^ (-plan.ν) .* sqrt(π) ./ 4
    end

    reverse!(plan.x)
end


"""
    _eval_gl_hm!(plan::Union{SingleBesselPlan, HankelPlan}, ell::Vector)

Given an input `plan::Union{SingleBesselPlan, HankelPlan}`, compute the ``g_{\\ell}`` values and 
the ``h_{m, \\mathrm{corr}}`` coefficents inside the IFFT. They are, respectively:

```math
g_{\\ell}^{(n)}(z) = (-1)^n \\, 2^{z-n} \\, \\frac{
        \\Gamma\\left(\\frac{\\ell + z - n}{2}\\right)
    }{
        \\Gamma\\left(\\frac{3 + \\ell + n - z}{2}\\right)
    } \\; , \\quad\\quad 
h_{m, \\mathrm{corr}} = (x_0 y_0)^{- i \\eta_m}
```
where ``\\eta_m = \\frac{2 \\pi m}{N \\, \\Delta_{\\ln x}}``, and ``x_0``
``y_0`` are the smallest values of `plan.x` and `plan.y`, respectively. 

The vector of their values are stored  in `plan.gl` and `plan.hy_corr`,
and nothing is returned.

See also: [`SingleBesselPlan`](@ref), [`HankelPlan`](@ref)
"""
function _eval_gl_hm!(plan::Union{SingleBesselPlan, HankelPlan}, ell::Vector)
    z = plan.ν .+ im .* plan.ηm
    plan.gl = zeros(length(ell), length(z))

    plan.hm_corr = zeros(ComplexF64, length(ell), Int(plan.N / 2 + 1))

    @inbounds for myl in 1:length(ell)
        plan.hm_corr[myl, :] =
            (plan.x[1] .* plan.y[myl, 1]) .^ (-im .* plan.ηm)
        plan.gl[myl, :] = _eval_gl(ell[myl], z, plan.n)
    end
    plan.hm = zeros(ComplexF64, size(plan.gl))

end




##########################################################################################92


"""
    prepare_FFTLog!(plan::Union{SingleBesselPlan, HankelPlan}, ell::Vector)

Given an input `plan::Union{SingleBesselPlan, HankelPlan}`, pre-plan an optimized real-input FFT for all
the Bessel function orders stored in the vector `ell`.
In other words, it computes:
- the `y` vector of values where the transformed will be evaluated (stored in `plan.y`).
- the corresponding `gl` vector of ``g_{\\ell}`` values (stored in `plan.gl`).
- the `m` vector of indexes for the ``c_m`` coefficents (stored in `plan.m`).
- the corresponding `ηm` and `hm_corr` vector of ``\\eta_m`` and ``h_{m, \\mathrm{corr}}`` 
  values (stored in `plan.ηm` and `plan.hm_corr`).

See also: [`SingleBesselPlan`](@ref), [`HankelPlan`](@ref)
"""
function prepare_FFTLog!(plan::Union{SingleBesselPlan, HankelPlan}, ell::Vector)
    plan.x = _logextrap(plan.x, plan.n_extrap_low + plan.n_pad,
        plan.n_extrap_high + plan.n_pad)

    _eval_y!(plan, ell)

    plan.m = Array(0:length(plan.x)/2)
    _eval_ηm!(plan)

    _eval_gl_hm!(plan, ell)

    plan.plan_rfft = plan_rfft(plan.x)
    plan.plan_irfft = plan_irfft(
        randn(Complex{Float64}, length(ell), Int((length(plan.x) / 2) + 1)),
        plan.original_length + plan.n_extrap_low + plan.n_extrap_high + 2 * plan.n_pad, 2)
end


"""
    prepare_Hankel!(plan::HankelPlan, ell::Vector)

Given an input `plan::HankelPlan`, pre-plan an optimized real-input FFT for all
the Bessel function orders stored in the vector `ell` concerning an Hankel transform.
Same as `prepare_FFTLog`, checks its documentation for more information.

See also: [`HankelPlan`](@ref),  [`prepare_FFTLog!`](@ref)
"""
function prepare_Hankel!(hankplan::HankelPlan, ell::Vector)
    prepare_FFTLog!(hankplan, ell .- 0.5)
end



##########################################################################################92



"""
    get_y(plan::Union{SingleBesselPlan, HankelPlan})::Vector

Return the computed `y` vector, containing the values where the transformed
function will be evaluated.

See also: [`SingleBesselPlan`](@ref), [`HankelPlan`](@ref)
"""
function get_y(plan::Union{SingleBesselPlan, HankelPlan})
    n1 = plan.n_extrap_low + plan.n_pad + 1
    n2 = plan.n_extrap_low + plan.n_pad + plan.original_length
    return plan.y[:, n1:n2]
end



"""
    evaluate_FFTLog!(fy, plan::Union{SingleBesselPlan, HankelPlan}, fx)

Given an input `plan::Union{SingleBesselPlan, HankelPlan}`, evaluate the FFT `fy` of the `fx` y-axis data
on the basis of the parameters stored in `plan`.
The result is stored both in `plan.fy` and in the input `fy`.

See also: [`SingleBesselPlan`](@ref), [`HankelPlan`](@ref)
"""
function evaluate_FFTLog!(fy, plan::Union{SingleBesselPlan, HankelPlan}, fx)
    fx = _logextrap(fx, plan.n_extrap_low, plan.n_extrap_high)
    fx = _zeropad(fx, plan.n_pad)
    _eval_cm!(plan, fx)

    @inbounds for myl in 1:length(plan.y[:, 1])
        plan.hm[myl, :] = plan.cm .* @view plan.gl[myl, :]
        plan.hm[myl, :] .*= @view plan.hm_corr[myl, :]
    end

    plan.fy[:, :] .= plan.plan_irfft * conj!(plan.hm)
    plan.fy[:, :] .*= @view plan.fy_corr[:, :]

    n1 = plan.n_extrap_low + plan.n_pad + 1
    n2 = plan.n_extrap_low + plan.n_pad + plan.original_length
    fy[:, :] .= @view plan.fy[:, n1:n2]
end



"""
    evaluate_Hankel(plan::HankelPlan, fx)::Union{Vector, Matrix}

Given an input `plan::HankelPlan`, evaluate the FFT `fy` of the `fx` y-axis data
on the basis of the parameters stored in `plan` for an Hankel transform.
The result is both stored in `plan.fy` and retuned as output.

See also: [`HankelPlan`](@ref)
"""
function evaluate_Hankel(hankplan::HankelPlan, fx)
    n1 = hankplan.n_extrap_low + hankplan.n_pad + 1
    n2 = hankplan.n_extrap_low + hankplan.n_pad + hankplan.original_length
    fy = evaluate_FFTLog(hankplan, fx .* (hankplan.x[n1:n2]) .^ (5 / 2))
    fy .*= sqrt.(2 * get_y(hankplan) / π)
    return fy
end


"""
    evaluate_Hankel!(fy, plan::HankelPlan, fx)

Given an input `plan::HankelPlan`, evaluate the FFT `fy` of the `fx` y-axis data
on the basis of the parameters stored in `plan` for an Hankel transform.
The result is stored both in `plan.fy` and in the input `fy`.

See also: [`HankelPlan`](@ref)
"""
function evaluate_Hankel!(fy, hankplan::HankelPlan, fx)
    n1 = hankplan.n_extrap_low + hankplan.n_pad + 1
    n2 = hankplan.n_extrap_low + hankplan.n_pad + hankplan.original_length
    evaluate_FFTLog!(fy, hankplan, fx .* (hankplan.x[n1:n2]) .^ (5 / 2))
    fy .*= sqrt.(2 * get_y(hankplan) / π)
    return fy
end



