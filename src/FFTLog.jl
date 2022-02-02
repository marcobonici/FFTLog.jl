module FFTLog

using FFTW
using Base: @kwdef
using SpecialFunctions: gamma
export mul!

abstract type AbstractPlan end

"""
    _cwindow(N::Vector{Float64}, NCut::Int64)

This function returns the smoothing window function as defined
in Eq. (C.1) of [McEwen et al. (2016)](https://arxiv.org/abs/1603.04826).
"""
function _cwindow(N::AbstractArray{T}, NCut::I) where {T,I}
    NRight = last(N) - NCut
    NR = filter(x->x>=NRight, N)
    ThetaRight = (last(N).-NR) ./ (last(N) - NRight - 1)
    W = ones(length(N))
    W[findall(x->x>=NRight, N)] = ThetaRight .- 1 ./ (2*π) .* sin.(2 .* π .*
	ThetaRight)
    return W
end

@kwdef mutable struct FFTLogPlan <: AbstractPlan
    XArray::Vector{Float64}
    YArray::Matrix{Float64} = zeros(10,10)
    FYArray::Matrix{Float64} = zeros(10,10)
    HMArray::Matrix{ComplexF64} = zeros(10,10)
    HMArrayCorr::Matrix{ComplexF64} = zeros(10,10)
    DLnX::Float64 = log(XArray[2]/XArray[1])
    FYArrayCorr::Matrix{Float64} = zeros(10,10)
    OriginalLenght::Int64 = length(XArray)
    GLArray::Matrix{ComplexF64} = zeros(100,100)
    ν::Float64 = 1.01
    NExtrapLow::Int64 = 0
    NExtrapHigh::Int64 = 0
    CWindowWidth::Float64 = 0.25
    NPad::Int64 = 0
    N::Int64 = OriginalLenght+NExtrapHigh+NExtrapLow+2*NPad
    M::Vector{Float64} = zeros(N)
    CM::Vector{ComplexF64} = zeros(N)
    ηM::Vector{Float64} = zeros(N)
    PlanFFT::FFTW.rFFTWPlan = plan_rfft(randn(1024))
    PlanIFFT = plan_irfft(randn(Complex{Float64}, 2,
    Int((OriginalLenght+NExtrapLow+NExtrapHigh+2*NPad)/2) +1),
    OriginalLenght+NExtrapLow+NExtrapHigh+2*NPad, 2)
end

@kwdef mutable struct HankelPlan <: AbstractPlan
    XArray::Vector{Float64}
    YArray::Matrix{Float64} = zeros(10,10)
    FYArray::Matrix{Float64} = zeros(10,10)
    HMArray::Matrix{ComplexF64} = zeros(10,10)
    HMArrayCorr::Matrix{ComplexF64} = zeros(10,10)
    DLnX::Float64 = log(XArray[2]/XArray[1])
    FYArrayCorr::Matrix{Float64} = zeros(10,10)
    OriginalLenght::Int64 = length(XArray)
    GLArray::Matrix{ComplexF64} = zeros(100,100)
    ν::Float64 = 1.01
    NExtrapLow::Int64 = 0
    NExtrapHigh::Int64 = 0
    CWindowWidth::Float64 = 0.25
    NPad::Int64 = 0
    N::Int64 = OriginalLenght+NExtrapHigh+NExtrapLow+2*NPad
    M::Vector{Float64} = zeros(N)
    CM::Vector{ComplexF64} = zeros(N)
    ηM::Vector{Float64} = zeros(N)
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
    plan.ηM = 2 .* π ./ (plan.N .* plan.DLnX) .* plan.M
    return plan.ηM
end

function _gl(Ell::Float64, ZArray::Vector{ComplexF64})
    GL = 2 .^ ZArray .* gamma.((Ell .+ ZArray)/2) ./ gamma.((3 .+ Ell .- ZArray)/2)
    return GL
end

function _logextrap(X::Vector{Float64}, NExtrapLow::Int64, NExtrapHigh::Int64)
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

function _zeropad(X::Vector{Float64}, NPad::Int)
    ZeroArray = zeros(NPad)
    return vcat(ZeroArray, X, ZeroArray)
end

function _evaluateYArray(plan::AbstractPlan, Ell::Vector{T}) where T
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
        plan.GLArray[myl,:] = _gl(Ell[myl], ZArray)
    end
    plan.HMArray = zeros(ComplexF64, size(plan.GLArray))

end

function prepareFFTLog!(plan::AbstractPlan, Ell::Vector{T}) where T
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

function prepareHankel!(hankplan::HankelPlan, Ell::Vector{T}) where T
    prepareFFTLog!(hankplan, Ell .-0.5)
end

function getY(plan::AbstractPlan)
    return plan.YArray[:,plan.NExtrapLow+plan.NPad+1:plan.NExtrapLow+
	plan.NPad+plan.OriginalLenght]
end

"""
function evaluateFFTLog(plan::AbstractPlan, FXArray) where T
    FXArray = _logextrap(FXArray, plan.NExtrapLow,
	plan.NExtrapHigh)
    FXArray = _zeropad(FXArray, plan.NPad)
    _evalcm!(plan, FXArray)
    
    #FYArray = zeros(size(plan.YArray))
    HMArray = zeros(ComplexF64, size(plan.GLArray))
    @inbounds for myl in 1:length(plan.YArray[:,1])
        HMArray[myl,:] = plan.CM .* @view plan.GLArray[myl,:]
        HMArray[myl,:] .*= @view plan.HMArrayCorr[myl, :]
    end

    plan.FYArray[:,:] .= plan.PlanIFFT * conj!(HMArray)
    plan.FYArray[:,:] .*= @view plan.FYArrayCorr[:,:]

    
    return @view plan.FYArray[:,plan.NExtrapLow+plan.NPad+
	1:plan.NExtrapLow+plan.NPad+plan.OriginalLenght]
end
"""

function evaluateFFTLog(plan::AbstractPlan, FXArray) where T
    FYArray = zeros(size(getY(plan)))
    
    evaluateFFTLog!(FYArray, plan, FXArray)
    
    return FYArray
end

function evaluateFFTLog!(FYArray, plan::AbstractPlan, FXArray) where T
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
