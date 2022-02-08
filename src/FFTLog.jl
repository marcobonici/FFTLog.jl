module FFTLog

using FFTW
using Base: @kwdef
using SpecialFunctions: gamma
export mul!

abstract type AbstractPlan end

set_num_threads(N::Int) = FFTW.set_num_threads(N)

"""
    _cwindow(N::AbstractArray, NCut::Int)

This function returns the smoothing window function as defined
in Eq. (C.1) of [McEwen et al. (2016)](https://arxiv.org/abs/1603.04826).
"""
function _cwindow(N::AbstractArray, NCut::Int)
    NRight = last(N) - NCut
    NR = filter(x->x>=NRight, N)
    ThetaRight = (last(N).-NR) ./ (last(N) - NRight - 1)
    W = ones(length(N))
    W[findall(x->x>=NRight, N)] = ThetaRight .- 1 ./ (2*π) .* sin.(2 .* π .*
	ThetaRight)
    return W
end

@kwdef mutable struct FFTLogPlan{T,C} <: AbstractPlan 
    XArray::Vector{T}
    YArray::Matrix{T} = zeros(10,10)
    FYArray::Matrix{T} = zeros(10,10)
    HMArray::Matrix{C} = zeros(10,10) .+im
    HMArrayCorr::Matrix{C} = zeros(10,10) .+im
    DLnX::T = log(XArray[2]/XArray[1])
    FYArrayCorr::Matrix{T} = zeros(10,10)
    OriginalLenght::Int = length(XArray)
    GLArray::Matrix{C} = zeros(100,100) .+im
    ν::T = 1.01
    NExtrapLow::Int = 0
    NExtrapHigh::Int = 0
    CWindowWidth::T = 0.25
    NPad::Int = 0
    n::Int = 0
    N::Int = OriginalLenght+NExtrapHigh+NExtrapLow+2*NPad
    M::Vector{T} = zeros(N)
    CM::Vector{C} = zeros(N) .+im
    ηM::Vector{T} = zeros(N)
    PlanFFT::FFTW.rFFTWPlan = plan_rfft(randn(1024))
    PlanIFFT = plan_irfft(randn(Complex{Float64}, 2,
    Int((OriginalLenght+NExtrapLow+NExtrapHigh+2*NPad)/2) +1),
    OriginalLenght+NExtrapLow+NExtrapHigh+2*NPad, 2)
end

@kwdef mutable struct HankelPlan{T,C} <: AbstractPlan
    XArray::Vector{T}
    YArray::Matrix{T} = zeros(10,10)
    FYArray::Matrix{T} = zeros(10,10)
    HMArray::Matrix{C} = zeros(10,10) .+im
    HMArrayCorr::Matrix{C} = zeros(10,10) .+im
    DLnX::T = log(XArray[2]/XArray[1])
    FYArrayCorr::Matrix{T} = zeros(10,10)
    OriginalLenght::Int = length(XArray)
    GLArray::Matrix{C} = zeros(100,100) .+im
    ν::T = 1.01
    NExtrapLow::Int = 0
    NExtrapHigh::Int = 0
    CWindowWidth::T = 0.25
    NPad::Int = 0
    n::Int = 0
    N::Int = OriginalLenght+NExtrapHigh+NExtrapLow+2*NPad
    M::Vector{T} = zeros(N)
    CM::Vector{C} = zeros(N) .+im
    ηM::Vector{T} = zeros(N)
    PlanFFT::FFTW.rFFTWPlan = plan_rfft(randn(1024))
    PlanIFFT = plan_irfft(randn(Complex{Float64}, 2,
    Int((OriginalLenght+NExtrapLow+NExtrapHigh+2*NPad)/2) +1),
    OriginalLenght+NExtrapLow+NExtrapHigh+2*NPad, 2)
end

function _evalcm!(plan::AbstractPlan, FXArray)
    plan.CM = plan.PlanFFT * (FXArray .* plan.XArray .^ (-plan.ν))
    plan.CM .*= _cwindow(plan.M, floor(Int, plan.CWindowWidth*plan.N/2))
end

function _evalηm!(plan::AbstractPlan)
    #TODO: #9 since we know the length of the initial array, we could use this info here to
    #remove the necessity of DLnX
    plan.ηM = 2 .* π ./ (plan.N .* plan.DLnX) .* plan.M
    return plan.ηM
end

function _gl(Ell, ZArray::Vector, n::Int)
    GL = ((-1)^n) .* 2 .^ (ZArray .-n) .* gamma.((Ell .+ ZArray .- n)/2) ./
    gamma.((3 .+ Ell .+ n .- ZArray)/2)
    if n != 0
        for i in 1:n
            GL .*= (ZArray-i)
        end
    end
    return GL
end

function _logextrap(X::Vector, NExtrapLow::Int, NExtrapHigh::Int)
    DLnXLow = log(X[2]/X[1])
    DLnXHigh= log(reverse(X)[1]/reverse(X)[2])
    if NExtrapLow != 0
        LowX = X[1] .* exp.(DLnXLow .* Array(-NExtrapLow:-1))
        X = vcat(LowX, X)
    end
    if NExtrapHigh != 0
        HighX = last(X) .* exp.(DLnXHigh .* Array(1:NExtrapHigh))
        X = vcat(X,HighX)
    end
    return X
end

function _zeropad(X::Vector, NPad::Int)
    ZeroArray = zeros(NPad)
    return vcat(ZeroArray, X, ZeroArray)
end

function _evaluateYArray(plan::AbstractPlan, Ell::Vector)
    reverse!(plan.XArray)

    plan.YArray = zeros(length(Ell), length(plan.XArray))
    plan.FYArrayCorr = zeros(size(plan.YArray))

    @inbounds for myl in 1:length(Ell)
        plan.YArray[myl,:] = (Ell[myl] + 1) ./ plan.XArray
        plan.FYArrayCorr[myl,:] =
        plan.YArray[myl,:] .^ (-plan.ν) .* sqrt(π) ./4
    end

    plan.FYArray = zeros(size(plan.YArray))

    reverse!(plan.XArray)
end

function _evaluateGLandHM(plan::AbstractPlan, Ell::Vector)
    ZArray = plan.ν .+ im .* plan.ηM
    plan.GLArray = zeros(length(Ell), length(ZArray))

    plan.HMArrayCorr = zeros(ComplexF64, length(Ell), Int(plan.N/2+1))

    @inbounds for myl in 1:length(Ell)
        plan.HMArrayCorr[myl,:] =
        (plan.XArray[1] .* plan.YArray[myl,1] ) .^ (-im .*plan.ηM)
        plan.GLArray[myl,:] = _gl(Ell[myl], ZArray, plan.n)
    end
    plan.HMArray = zeros(ComplexF64, size(plan.GLArray))

end

function prepareFFTLog!(plan::AbstractPlan, Ell::Vector)
    plan.XArray = _logextrap(plan.XArray, plan.NExtrapLow + plan.NPad,
	plan.NExtrapHigh + plan.NPad)
    
    _evaluateYArray(plan, Ell)
    
    plan.M = Array(0:length(plan.XArray)/2)
    _evalηm!(plan)
    
    _evaluateGLandHM(plan, Ell)

    plan.PlanFFT = plan_rfft(plan.XArray)
    plan.PlanIFFT = plan_irfft(randn(Complex{Float64}, length(Ell),
    Int((length(plan.XArray)/2) +1)),
    plan.OriginalLenght+plan.NExtrapLow+plan.NExtrapHigh+2*plan.NPad, 2)
end

function prepareHankel!(hankplan::HankelPlan, Ell::Vector)
    prepareFFTLog!(hankplan, Ell .-0.5)
end

function getY(plan::AbstractPlan)
    return plan.YArray[:,plan.NExtrapLow+plan.NPad+1:plan.NExtrapLow+
	plan.NPad+plan.OriginalLenght]
end

function evaluateFFTLog(plan::AbstractPlan, FXArray)
    FYArray = zeros(size(getY(plan)))
    
    evaluateFFTLog!(FYArray, plan, FXArray)
    
    return FYArray
end

function evaluateFFTLog!(FYArray, plan::AbstractPlan, FXArray)
    FXArray = _logextrap(FXArray, plan.NExtrapLow,
	plan.NExtrapHigh)
    FXArray = _zeropad(FXArray, plan.NPad)
    _evalcm!(plan, FXArray)

    @inbounds for myl in 1:length(plan.YArray[:,1])
        plan.HMArray[myl,:] = plan.CM .* @view plan.GLArray[myl,:]
        plan.HMArray[myl,:] .*= @view plan.HMArrayCorr[myl, :]
    end

    plan.FYArray[:,:] .= plan.PlanIFFT * conj!(plan.HMArray)
    plan.FYArray[:,:] .*= @view plan.FYArrayCorr[:,:]

    
    FYArray[:,:] .= @view plan.FYArray[:,plan.NExtrapLow+plan.NPad+
	1:plan.NExtrapLow+plan.NPad+plan.OriginalLenght]
end


function evaluateHankel(hankplan::HankelPlan, FXArray)
    FY = evaluateFFTLog(hankplan, FXArray .*(
        hankplan.XArray[hankplan.NExtrapLow+hankplan.NPad+1:hankplan.NExtrapLow+
	hankplan.NPad+hankplan.OriginalLenght]).^(5/2) )
    FY .*= sqrt.(2*getY(hankplan)/π)
    return FY
end

function evaluateHankel!(FYArray, hankplan::HankelPlan, FXArray)
    evaluateFFTLog!(FYArray, hankplan, FXArray .*(
        hankplan.XArray[hankplan.NExtrapLow+hankplan.NPad+1:hankplan.NExtrapLow+
	hankplan.NPad+hankplan.OriginalLenght]).^(5/2) )
    FYArray .*= sqrt.(2*getY(hankplan)/π)
    return FYArray
end

function mul!(Y, Q::FFTLogPlan, A)
    evaluateFFTLog!(Y, Q, A)
end

function mul!(Y, Q::HankelPlan, A)
    Y[:,:] .= evaluateHankel!(Y, Q, A)
end

end # module
