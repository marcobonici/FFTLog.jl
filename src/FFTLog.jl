module FFTLog

using FFTW
using Base: @kwdef
using SpecialFunctions

"""
    CWindow(N::Vector{Float64}, NCut::Int64)

This function evaluates the smoothing window function as defined
in Eq. (C.1) of [McEwen et (2016)](https://arxiv.org/abs/1603.04826).
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
    DLnX::Float64 = log(XArray[2]/XArray[1])
    FXArray::Vector{Float64}
    OriginalLenght::Int64 = length(XArray)
    ν::Float64 = 1.01
    NExtrapLow::Int64 = 0
    NExtrapHigh::Int64 = 0
    CWindowWidth::Float64 = 0.25
    NPad::Int64 = 0
    N::Int64 = OriginalLenght+NExtrapHigh+NExtrapLow+2*NPad
    M::Vector{Float64} = zeros(N)
    CM::Vector{ComplexF64} = zeros(N)
    ηM::Vector{Float64} = zeros(N)
end

function _evalcm!(FFTLog::FFTLogPlan)
    FFTLog.CM = FFTW.rfft(FFTLog.FXArray .* FFTLog.XArray .^ (-FFTLog.ν))
    FFTLog.M = Array(0:length(FFTLog.XArray)/2)
    FFTLog.CM = FFTLog.CM .* _cwindow(FFTLog.M, floor(Int,
	FFTLog.CWindowWidth*FFTLog.N/2))
end

function _evalηm!(FFTLog::FFTLogPlan)
    FFTLog.ηM = 2 .* π ./ (FFTLog.N .* FFTLog.DLnX) .* FFTLog.M
    return FFTLog.ηM
end

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
	SpecialFunctions.gamma.(AlphaPlus) ./ SpecialFunctions.gamma.(AlphaMinus)
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

function _gl(Ell::Float64, ZArray::Vector{ComplexF64}, TwoZArray::Vector{ComplexF64})
    GL = TwoZArray .* SpecialFunctions.gamma.((Ell .+ ZArray)/2) ./ SpecialFunctions.gamma.((3 .+ Ell .- ZArray)/2)
    return GL
end

function _gl1(Ell::Float64, ZArray::Vector{ComplexF64})
    GL1 = -2 .^(ZArray .- 1) * (z_array -1) .* _g_m_vals(Ell+0.5, ZArray .- 2.5)
    return GL1
end

function _gl2(Ell::Float64, ZArray::Vector{ComplexF64})
    GL2 = 2 .^ (ZArray .- 2) .* (ZArray .- 2) .* (ZArray .- 1) .*
	_g_m_vals(Ell+0.5, ZArray .- 3.5)
    return _gl
end

function _logextrap(X::Vector{Float64}, NExtrapLow::Int64, NExtrapHigh::Int64)
    DLnXLow = log(X[2]/X[1])
    DLnXHigh= log(reverse(X)[1]/reverse(X)[2])
    if NExtrapLow != 0
        LowX = X[1] .* exp.(DLnXLow .* Array(-NExtrapLow:-1))
        X = vcat(LowX, X)
    end
    if NExtrapHigh != 0
        HighX = last(X) .* exp.(DLnXLow .* Array(1:NExtrapHigh))
        X = vcat(X,HighX)
    end
    return X
end

function _zeropad!(FFTLog::FFTLogPlan)
    ZeroArray = zeros(FFTLog.NPad)
    FFTLog.XArray = _logextrap(FFTLog.XArray, FFTLog.NPad, FFTLog.NPad)
    FFTLog.FXArray = vcat(ZeroArray, FFTLog.FXArray, ZeroArray)
end

function _checknumberelements!(FFTLog::FFTLogPlan)
    if iseven(FFTLog.N+1)
        deleteat!(FFTLog.XArray, length(FFTLog.XArray))
        deleteat!(FFTLog.FXArray, length(FFTLog.FXArray))
        FFTLog.N -= 1
        if (FFTLog.NExtrapHigh != 0)
            FFTLog.NExtrapHigh -= 1
        end
    end
end

function evaluateFFTLog(FFTLog::FFTLogPlan, Ell::Vector{T}) where T
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
    TwoZArray = 2 .^ ZAr
    @inbounds for myl in 1:length(Ell)
        YArray[myl,:] = (Ell[myl] + 1) ./ reverse(FFTLog.XArray)
        HMArray[myl,:]  = FFTLog.CM .* (FFTLog.XArray[1] .* YArray[myl,1] ) .^
		(-im .*FFTLog.ηM) .* _gl(Ell[myl], ZAr, TwoZArray)
        FYArray[myl,:] = FFTW.irfft(conj(HMArray[myl,:]),
		length(FFTLog.XArray)) .* YArray[myl,:] .^ (-FFTLog.ν) .* sqrt(π) ./4
    end
    return YArray[:,FFTLog.NExtrapLow+FFTLog.NPad+1:FFTLog.NExtrapLow+
	FFTLog.NPad+FFTLog.OriginalLenght], FYArray[:,FFTLog.NExtrapLow+FFTLog.NPad+
	1:FFTLog.NExtrapLow+FFTLog.NPad+FFTLog.OriginalLenght]
end

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
        FYArray[myl,:] = FFTW.irfft(conj(HMArray[myl,:]),
		length(FFTLog.XArray)) .* YArray[myl,:] .^ (-FFTLog.ν) .* sqrt(π) ./4
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

end # module
