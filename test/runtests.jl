using FFTLog
using Test

function f(k, a=1)
    return exp(-k^2.0 *a^2 / 2)
end

function F(r, a=1)
    return exp(- r^2.0 / (2. *a^2))
end

function LogSpaced(min::T, max::T, n::I) where {T,I}
    logmin = log10(min)
    logmax = log10(max)
    logarray = Array(LinRange(logmin, logmax, n))
    return exp10.(logarray)
end

@testset "Analytical Hankel test" begin
    FFTLog.set_num_threads(Threads.nthreads())
    k = LogSpaced(10^(-5), 10., 2^10)
    fk = f.(k)
    FFTTest = FFTLog.HankelPlan(x = k, n_extrap_low = 1500, ν=1.01, n_extrap_high = 1500,
    n_pad = 500)
    Ell = Array([0.])
    prepare_Hankel!(FFTTest, Ell)
    Y = get_y(FFTTest)
    FY = zeros(size(Y));
    FY = evaluate_Hankel(FFTTest, fk)
    FY_mul = zeros(size(FY))
    Fr_analytical = F.(Y[1,:])
    @test isapprox(Fr_analytical, FY[1,:], rtol=1e-10)
    @test isapprox(FY, mul!(FY_mul, FFTTest, fk), rtol=1e-10)
end

@testset "mul! operator test" begin
    k = LogSpaced(10^(-5), 10., 2^10)
    fk = f.(k)
    FFTTest = FFTLog.FFTLogPlan(x = k, n_extrap_low = 1500, ν=1.01, n_extrap_high = 1500,
    n_pad = 500)
    Ell = Array([0.])
    prepare_FFTLog!(FFTTest, Ell)
    Y = FFTLog.get_y(FFTTest)
    FY = zeros(size(Y));
    fk_k2 = fk .* (k.^2)
    FY = evaluate_FFTLog(FFTTest, fk_k2)
    FY_mul = zeros(size(FY))
    mul!(FY_mul, FFTTest, fk_k2)
    @test isapprox(FY_mul, FY, rtol=1e-10)
end
