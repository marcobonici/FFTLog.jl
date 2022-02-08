module FFTLog

using FFTW
using Base: @kwdef
using SpecialFunctions: gamma

export prepare_FFTLog!, evaluate_FFTLog, evaluate_FFTLog!
export prepare_Hankel!, evaluate_Hankel, evaluate_Hankel!
export mul!
export get_y

abstract type AbstractPlan end

set_num_threads(N::Int) = FFTW.set_num_threads(N)

"""
    _c_window(N::AbstractArray, NCut::Int)

This function returns the smoothing window function as defined
in Eq. (C.1) of [McEwen et al. (2016)](https://arxiv.org/abs/1603.04826).
"""
function _c_window(N::AbstractArray, NCut::Int)
    NRight = last(N) - NCut
    NR = filter(x->x>=NRight, N)
    ThetaRight = (last(N).-NR) ./ (last(N) - NRight - 1)
    W = ones(length(N))
    W[findall(x->x>=NRight, N)] = ThetaRight .- 1 ./ (2*π) .* sin.(2 .* π .*
	ThetaRight)
    return W
end


"""
    FFTLogPlan()

This struct contains all the elements necessary to evaluate the integral with one Bessel function.
"""
@kwdef mutable struct FFTLogPlan{T,C} <: AbstractPlan
    x::Vector{T}
    y::Matrix{T} = zeros(10,10)
    fy::Matrix{T} = zeros(10,10)
    hm::Matrix{C} = zeros(10,10) .+im
    hm_corr::Matrix{C} = zeros(10,10) .+im
    d_ln_x::T = log(x[2]/x[1])
    fy_corr::Matrix{T} = zeros(10,10)
    original_length::Int = length(x)
    gl::Matrix{C} = zeros(100,100) .+im
    ν::T = 1.01
    n_extrap_low::Int = 0
    n_extrap_high::Int = 0
    c_window_width::T = 0.25
    n_pad::Int = 0
    n::Int = 0
    N::Int = original_length+n_extrap_low+n_extrap_high+2*n_pad
    m::Vector{T} = zeros(N)
    cm::Vector{C} = zeros(N) .+im
    ηm::Vector{T} = zeros(N)
    plan_rfft::FFTW.rFFTWPlan = plan_rfft(randn(1024))
    plan_irfft = plan_irfft(randn(Complex{Float64}, 2,
    Int((original_length+n_extrap_low+n_extrap_high+2*n_pad)/2) +1),
    original_length+n_extrap_low+n_extrap_high+2*n_pad, 2)
end

@kwdef mutable struct HankelPlan{T,C} <: AbstractPlan
    x::Vector{T}
    y::Matrix{T} = zeros(10,10)
    fy::Matrix{T} = zeros(10,10)
    hm::Matrix{C} = zeros(10,10) .+im
    hm_corr::Matrix{C} = zeros(10,10) .+im
    d_ln_x::T = log(x[2]/x[1])
    fy_corr::Matrix{T} = zeros(10,10)
    original_length::Int = length(x)
    gl::Matrix{C} = zeros(100,100) .+im
    ν::T = 1.01
    n_extrap_low::Int = 0
    n_extrap_high::Int = 0
    c_window_width::T = 0.25
    n_pad::Int = 0
    n::Int = 0
    N::Int = original_length+n_extrap_low+n_extrap_high+2*n_pad
    m::Vector{T} = zeros(N)
    cm::Vector{C} = zeros(N) .+im
    ηm::Vector{T} = zeros(N)
    plan_rfft::FFTW.rFFTWPlan = plan_rfft(randn(1024))
    plan_irfft = plan_irfft(randn(Complex{Float64}, 2,
    Int((original_length+n_extrap_low+n_extrap_high+2*n_pad)/2) +1),
    original_length+n_extrap_low+n_extrap_high+2*n_pad, 2)
end

function _eval_cm!(plan::AbstractPlan, fx)
    plan.cm = plan.plan_rfft * (fx .* plan.x .^ (-plan.ν))
    plan.cm .*= _c_window(plan.m, floor(Int, plan.c_window_width*plan.N/2))
end

#TODO: adjust definition, not mutating function!
function _eval_ηm!(plan::AbstractPlan)
    #TODO: #9 since we know the length of the initial array, we could use this info here to
    #remove the necessity of DLnX
    plan.ηm = 2 .* π ./ (plan.N .* plan.d_ln_x) .* plan.m
    return plan.ηm
end

function _gl(ell, z::Vector, n::Int)
    gl = ((-1)^n) .* 2 .^ (z .-n) .* gamma.((ell .+ z .- n)/2) ./
    gamma.((3 .+ ell .+ n .- z)/2)
    if n != 0
        for i in 1:n
            gl .*= (z-i)
        end
    end
    return gl
end

function _logextrap(x::Vector, n_extrap_low::Int, n_extrap_high::Int)
    d_ln_x_low = log(x[2]/x[1])
    d_ln_x_high= log(reverse(x)[1]/reverse(x)[2])
    if n_extrap_low != 0
        #TODO: check if you can directly vcat the array!
        low_x = x[1] .* exp.(d_ln_x_low .* Array(-n_extrap_low:-1))
        X = vcat(low_x, x)
    end
    if n_extrap_high != 0
        high_x = last(X) .* exp.(d_ln_x_high .* Array(1:n_extrap_high))
        X = vcat(X, high_x)
    end
    return X
end

function _zeropad(x::Vector, n_pad::Int)
    #TODO: check if you can directly vcat the array!
    zeros_array = zeros(n_pad)
    return vcat(zeros_array, x, zeros_array)
end

function _evaluate_y(plan::AbstractPlan, ell::Vector)
    reverse!(plan.x)

    plan.y = zeros(length(ell), length(plan.x))
    plan.fy_corr = zeros(size(plan.y))

    @inbounds for myl in 1:length(ell)
        plan.y[myl,:] = (ell[myl] + 1) ./ plan.x
        plan.fy_corr[myl,:] =
        plan.y[myl,:] .^ (-plan.ν) .* sqrt(π) ./4
    end

    plan.fy = zeros(size(plan.y))

    reverse!(plan.x)
end

function evaluate_gl_hm(plan::AbstractPlan, ell::Vector)
    z = plan.ν .+ im .* plan.ηm
    plan.gl = zeros(length(ell), length(z))

    plan.hm_corr = zeros(ComplexF64, length(ell), Int(plan.N/2+1))

    @inbounds for myl in 1:length(ell)
        plan.hm_corr[myl,:] =
        (plan.x[1] .* plan.y[myl,1] ) .^ (-im .*plan.ηm)
        plan.gl[myl,:] = _gl(ell[myl], z, plan.n)
    end
    plan.hm = zeros(ComplexF64, size(plan.gl))

end

function prepare_FFTLog!(plan::AbstractPlan, ell::Vector)
    plan.x = _logextrap(plan.x, plan.n_extrap_low + plan.n_pad,
	plan.n_extrap_high + plan.n_pad)

    _evaluate_y(plan, ell)

    plan.m = Array(0:length(plan.x)/2)
    _eval_ηm!(plan)

    evaluate_gl_hm(plan, ell)

    plan.plan_rfft = plan_rfft(plan.x)
    plan.plan_irfft = plan_irfft(randn(Complex{Float64}, length(ell),
    Int((length(plan.x)/2) +1)),
    plan.original_length+plan.n_extrap_low+plan.n_extrap_high+2*plan.n_pad, 2)
end

function prepare_Hankel!(hankplan::HankelPlan, ell::Vector)
    prepare_FFTLog!(hankplan, ell .-0.5)
end

function get_y(plan::AbstractPlan)
    #TODO: check this n_extrap_low
    return plan.y[:,plan.n_extrap_low+plan.n_pad+1:plan.n_extrap_low+
	plan.n_pad+plan.original_length]
end

function evaluate_FFTLog(plan::AbstractPlan, fx)
    fy = zeros(size(get_y(plan)))

    evaluate_FFTLog!(fy, plan, fx)

    return fy
end

function evaluate_FFTLog!(fy, plan::AbstractPlan, fx)
    fx = _logextrap(fx, plan.n_extrap_low,
	plan.n_extrap_high)
    fx = _zeropad(fx, plan.n_pad)
    _eval_cm!(plan, fx)

    @inbounds for myl in 1:length(plan.y[:,1])
        plan.hm[myl,:] = plan.cm .* @view plan.gl[myl,:]
        plan.hm[myl,:] .*= @view plan.hm_corr[myl, :]
    end

    plan.fy[:,:] .= plan.plan_irfft * conj!(plan.hm)
    plan.fy[:,:] .*= @view plan.fy_corr[:,:]

    #TODO: check if this is really n_extrap_low!
    fy[:,:] .= @view plan.fy[:,plan.n_extrap_low+plan.n_pad+
	1:plan.n_extrap_low+plan.n_pad+plan.original_length]
end


function evaluate_Hankel(hankplan::HankelPlan, fx)
    fy = evaluate_FFTLog(hankplan, fx .*(
        hankplan.x[hankplan.n_extrap_low+hankplan.n_pad+1:hankplan.n_extrap_low+
	hankplan.n_pad+hankplan.original_length]).^(5/2) )
    fy .*= sqrt.(2*get_y(hankplan)/π)
    return fy
end

function evaluate_Hankel!(fy, hankplan::HankelPlan, fx)
    #TODO: check if this is really n_extrap_low!
    evaluate_FFTLog!(fy, hankplan, fx .*(
        hankplan.x[hankplan.n_extrap_low+hankplan.n_pad+1:hankplan.n_extrap_low+
	hankplan.n_pad+hankplan.original_length]).^(5/2) )
    fy .*= sqrt.(2*get_y(hankplan)/π)
    return fy
end

function mul!(Y, Q::FFTLogPlan, A)
    evaluate_FFTLog!(Y, Q, A)
end

function mul!(Y, Q::HankelPlan, A)
    Y[:,:] .= evaluate_Hankel!(Y, Q, A)
end

end # module
