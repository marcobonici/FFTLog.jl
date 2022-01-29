module FFTLog

using FFTW
using Base: @kwdef
using SpecialFunctions: gamma

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

@kwdef mutable struct FFTLogPlan
    XArray::Vector{Float64}
    YArray::Matrix{Float64} = zeros(10,10)
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
    PlanFFT::FFTW.rFFTWPlan = FFTW.plan_rfft(randn(1024))
    PlanIFFT = plan_irfft(randn(Complex{Float64},
    Int((OriginalLenght+NExtrapLow+NExtrapHigh+2*NPad)/2) +1),
    OriginalLenght+NExtrapLow+NExtrapHigh+2*NPad)
end

function _evalcm!(FFTLog::FFTLogPlan, FXArray)
    FFTLog.CM = FFTLog.PlanFFT * (FXArray .* FFTLog.XArray .^ (-FFTLog.ν))
    FFTLog.CM .*= _cwindow(FFTLog.M, floor(Int, FFTLog.CWindowWidth*FFTLog.N/2))
end

function _evalηm!(FFTLog::FFTLogPlan)
    FFTLog.ηM = 2 .* π ./ (FFTLog.N .* FFTLog.DLnX) .* FFTLog.M
    return FFTLog.ηM
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

function prepareFFTLog!(FFTLog::FFTLogPlan, Ell::Vector{T}) where T
    FFTLog.XArray = _logextrap(FFTLog.XArray, FFTLog.NExtrapLow + FFTLog.NPad,
	FFTLog.NExtrapHigh + FFTLog.NPad)

    FFTLog.YArray = zeros(length(Ell), length(FFTLog.XArray))
    FFTLog.FYArrayCorr = zeros(size(FFTLog.YArray))
    
    reverse!(FFTLog.XArray)

    @inbounds for myl in 1:length(Ell)
        FFTLog.YArray[myl,:] = (Ell[myl] + 1) ./ FFTLog.XArray
        FFTLog.FYArrayCorr[myl,:] =
        FFTLog.YArray[myl,:] .^ (-FFTLog.ν) .* sqrt(π) ./4
    end
    reverse!(FFTLog.XArray)
    
    FFTLog.M = Array(0:length(FFTLog.XArray)/2)
    _evalηm!(FFTLog)

    FFTLog.HMArrayCorr = zeros(ComplexF64, length(Ell), Int(FFTLog.N/2+1))

    @inbounds for myl in 1:length(Ell)
        FFTLog.HMArrayCorr[myl,:] =
        (FFTLog.XArray[1] .* FFTLog.YArray[myl,1] ) .^ (-im .*FFTLog.ηM)
    end

    ZArray = FFTLog.ν .+ im .* FFTLog.ηM
    FFTLog.GLArray = zeros(length(Ell), length(ZArray))
    @inbounds for myl in 1:length(Ell)
        FFTLog.GLArray[myl,:] = _gl(Ell[myl], ZArray)
    end
    FFTLog.PlanFFT = plan_rfft(FFTLog.XArray)
end

function prepareHankel!(FFTLog::FFTLogPlan, Ell::Vector{T}) where T
    prepareFFTLog!(FFTLog, Ell .-0.5)
end


function evaluateFFTLog(FFTLog::FFTLogPlan, FXArray) where T
    FXArray = _logextrap(FXArray, FFTLog.NExtrapLow,
	FFTLog.NExtrapHigh)
    FXArray = _zeropad(FXArray, FFTLog.NPad)
    _evalcm!(FFTLog, FXArray)

    FXArray = @view FXArray[FFTLog.NExtrapLow+FFTLog.NPad+
    1:FFTLog.NExtrapLow+FFTLog.NPad+FFTLog.OriginalLenght]
    
    #FYArray = zeros(length(FFTLog.YArray[:,1]), length(FFTLog.XArray))
    FYArray = zeros(size(FFTLog.YArray))

    @inbounds for myl in 1:length(FFTLog.YArray[:,1])
        HMArray = FFTLog.CM .* @view FFTLog.GLArray[myl,:]
        HMArray .*= @view FFTLog.HMArrayCorr[myl, :]
        FYArray[myl,:] = FFTLog.PlanIFFT * conj!(HMArray)
        FYArray[myl,:] .*= @view FFTLog.FYArrayCorr[myl,:]
    end

    FYArray = @view FYArray[:,FFTLog.NExtrapLow+FFTLog.NPad+
	1:FFTLog.NExtrapLow+FFTLog.NPad+FFTLog.OriginalLenght]
    return FFTLog.YArray[:,FFTLog.NExtrapLow+FFTLog.NPad+1:FFTLog.NExtrapLow+
	FFTLog.NPad+FFTLog.OriginalLenght], FYArray
end

function evaluateHankel(FFTLog::FFTLogPlan, FXArray)
    Y , FY = evaluateFFTLog(FFTLog, FXArray .*sqrt.(
        FFTLog.XArray[FFTLog.NExtrapLow+FFTLog.NPad+1:FFTLog.NExtrapLow+
	FFTLog.NPad+FFTLog.OriginalLenght]))
    FY .*= sqrt.(2*Y/π)
    return Y, FY
end

#maybe we can remove the following functions, and unite them using a macro
"""
function EvaluateFFTLogDJ(FFTLog::FFTLogPlan, Ell::Vector{T}) where T
    FFTLog.XArray = _logextrap(FFTLog.XArray, FFTLog.NExtrapLow,
	FFTLog.NExtrapHigh)
    FFTLog.FXArray = _logextrap(FFTLog.FXArray, FFTLog.NExtrapLow,
	FFTLog.NExtrapHigh)
    _zeropad!(FFTLog)
    _evalcm!(FFTLog)
    _evalηm!(FFTLog)
    X0 = FFTLog.XArray[1]
    ZAr = FFTLog.ν .+ im .* FFTLog.ηM
    TwoZAr = 2 .^ZAr
    YArray = zeros(length(Ell), length(FFTLog.XArray))
    HMArray = zeros(ComplexF64, length(Ell), length(FFTLog.CM))
    FYArray = zeros(length(Ell), length(FFTLog.XArray))
    @inbounds for myl in 1:length(Ell)
        YArray[myl,:] = (Ell[myl] + 1) ./ reverse(FFTLog.XArray)
        HMArray[myl,:]  = FFTLog.CM .* (FFTLog.XArray[1] .* YArray[myl,1] ) .^
		(-im .*FFTLog.ηM) .* _gl1(Ell[myl], ZAr)
        FYArray[myl,:] = FFTLog.PlanIFFT * conj(HMArray[myl,:])
    end
    
    return YArray[:,FFTLog.NExtrapLow+FFTLog.NPad+1:FFTLog.NExtrapLow+
	FFTLog.NPad+FFTLog.OriginalLenght], FYArray[:,FFTLog.NExtrapLow+FFTLog.NPad+
	1:FFTLog.NExtrapLow+FFTLog.NPad+FFTLog.OriginalLenght]
end

function EvaluateFFTLogDDJ(FFTLog::FFTLogPlan, Ell::Vector{T}) where T
    FFTLog.XArray = _logextrap(FFTLog.XArray, FFTLog.NExtrapLow,
	FFTLog.NExtrapHigh)
    FFTLog.FXArray = _logextrap(FFTLog.FXArray, FFTLog.NExtrapLow,
	FFTLog.NExtrapHigh)
    _zeropad!(FFTLog)
    _evalcm!(FFTLog)
    _evalηm!(FFTLog)
    X0 = FFTLog.XArray[1]
    ZAr = FFTLog.ν .+ im .* FFTLog.ηM
    YArray = zeros(length(Ell), length(FFTLog.XArray))
    HMArray = zeros(ComplexF64, length(Ell), length(FFTLog.CM))
    FYArray = zeros(length(Ell), length(FFTLog.XArray))
    @inbounds for myl in 1:length(Ell)
        YArray[myl,:] = (Ell[myl] + 1) ./ reverse(FFTLog.XArray)
        HMArray[myl,:]  = FFTLog.CM .* (FFTLog.XArray[1] .* YArray[myl,1] ) .^
		(-im .*FFTLog.ηM) .* _gl2(Ell[myl], ZAr)
        FYArray[myl,:] = FFTW.irfft(conj(HMArray[myl,:]),
		length(FFTLog.XArray)) .* YArray[myl,:] .^ (-FFTLog.ν) .* sqrt(π) ./4
    end
    return YArray[:,FFTLog.NExtrapLow+FFTLog.NPad+1:FFTLog.NExtrapLow+
	FFTLog.NPad+FFTLog.OriginalLenght], FYArray[:,FFTLog.NExtrapLow+FFTLog.NPad+
	1:FFTLog.NExtrapLow+FFTLog.NPad+FFTLog.OriginalLenght]
end

#This is the original implementation, as taken from FAST-PT. Maybe this is
#not necessary anymore and we can remove it

function _g_m_vals(Mu::Float64, Q::Array{Complex{Float64},1})
    if(Mu+1+real(Q[1]) == 0)
        println("Γ(0) encountered. Please change another ν value! Try ν=1.1 .")
    end
    ImagQ = imag(Q)
    GM = zeros(ComplexF64, length(Q))
    cut = 200
    AsymQ = filter(x->abs(imag(x)) + abs(Mu) >cut, Q)
    AsymPlus = (Mu .+ 1 .+ AsymQ) ./2
    AsymMinus = (Mu .+ 1 .- AsymQ) ./2

    QGood = filter(x->abs.(imag(x)) .+ abs(Mu) .<=cut && x != Mu + 1 ,Q)
    AlphaPlus  = (Mu .+1 .+ QGood) ./2
    AlphaMinus = (Mu .+1 .- QGood) ./2
    GM[findall(x->abs.(imag(x)) .+ abs(Mu) .<=cut && x != Mu + 1 , Q)] .=
	gamma.(AlphaPlus) ./ gamma.(AlphaMinus)
    GM[findall(x->abs.(imag(x))+abs(Mu) .> cut && x != Mu + 1 , Q)] = exp.(
	(AsymPlus .- 0.5) .* log.(AsymPlus) .- (AsymMinus .- 0.5) .*
	log.(AsymMinus) .-
	AsymQ .+ 1/12 .* (1 ./AsymPlus .- 1 ./ AsymMinus) .+ 1/360 .* (1 ./
	AsymMinus .^3 .- 1 ./AsymPlus .^3) +1/1260 .* (1 ./AsymPlus .^ 5  .- 1 ./
	AsymMinus .^ 5) )
    return GM
end

function _gl(Ell::Float64, ZArray::Vector{ComplexF64})
    GL = 2 .^ZArray .*  _g_m_vals(Ell+0.5, ZArray .- 1.5)
    return GL
end


function _gl1(Ell::Float64, ZArray::Vector{ComplexF64})
    GL1 = -2 .^(ZArray .- 1) * (z_array -1) .* _g_m_vals(Ell+0.5, ZArray .- 2.5)
    return GL1
end

function _gl2(Ell::Float64, ZArray::Vector{ComplexF64})
    GL2 = 2 .^ (ZArray .- 2) .* (ZArray .- 2) .* (ZArray .- 1) .*
	_g_m_vals(Ell+0.5, ZArray .- 3.5)
    return GL2
end
"""
end # module
