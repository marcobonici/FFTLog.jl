using FFTLog
using Test
using DelimitedFiles
using NPZ

run(`wget https://zenodo.org/record/6021744/files/test_FFTLog.tar.xz\?download=1`)
run(`mv test_FFTLog.tar.xz\?download=1 test_FFTLog.tar.xz`)
run(`tar xvf test_FFTLog.tar.xz`)

input_path = pwd()*"/test_FFTLog/"


read_file = readdlm(input_path*"Pk_test")
k = read_file[:,1]
Pk = read_file[:,2]

@testset "BeyondLimber checks" begin
    FFTTest = FFTLog.FFTLogPlan(x = k, ν=1.01, n_extrap_low = 1500, n_extrap_high = 1500,
    n_pad = 2000)
    Ell = Array([1., 2.])
    FFTLog.prepare_FFTLog!(FFTTest, Ell)
    Y = FFTLog.get_y(FFTTest)
    FY = FFTLog.evaluate_FFTLog(FFTTest, Pk)
    Fr = npzread(input_path*"check_by_Fr.py.npy")
    r = npzread(input_path*"check_by_r.py.npy")

    @test isapprox(Fr, FY[1,:], rtol=1e-5)
    @test isapprox(r, Y[1,:], rtol=1e-5)

    FFTTest = FFTLog.FFTLogPlan(x = k, ν=1.01, n_extrap_low = 1500, n_extrap_high = 1500,
    n_pad = 2000, n = 1)
    Ell = Array([1., 2.])
    FFTLog.prepare_FFTLog!(FFTTest, Ell)
    Y = FFTLog.get_y(FFTTest)
    FY = FFTLog.evaluate_FFTLog(FFTTest, Pk)
    Fr1 = npzread(input_path*"check_by_Fr1.py.npy")
    r1 = npzread(input_path*"check_by_r1.py.npy")

    @test isapprox(Fr1, FY[1,:], rtol=1e-8)
    @test isapprox(r1, Y[1,:], rtol=1e-8)

    FFTTest = FFTLog.FFTLogPlan(x = k, ν=1.01, n_extrap_low = 1500, n_extrap_high = 1500,
    n_pad = 2000, n = 2)
    Ell = Array([1., 2.])
    FFTLog.prepare_FFTLog!(FFTTest, Ell)
    Y = FFTLog.get_y(FFTTest)
    FY = FFTLog.evaluate_FFTLog(FFTTest, Pk)
    Fr2 = npzread(input_path*"check_by_Fr2.py.npy")
    r2 = npzread(input_path*"check_by_r2.py.npy")

    @test isapprox(Fr2, FY[1,:], rtol=1e-8)
    @test isapprox(r2, Y[1,:], rtol=1e-8)

    FFTTest = FFTLog.HankelPlan(x = k, n_extrap_low = 1500, ν=1.01, n_extrap_high = 1500,
    n_pad = 0)
    Ell = Array([0., 2.])
    FFTLog.prepare_Hankel!(FFTTest, Ell)
    Y = FFTLog.get_y(FFTTest)
    FY = FFTLog.evaluate_Hankel(FFTTest, Pk.*(k.^(-2)))
    Fr = npzread(input_path*"check_by_hank_Fr.py.npy")
    r = npzread(input_path*"check_by_hank_r.py.npy")
    @test isapprox(Fr, FY[1,:], rtol=1e-8)
    @test isapprox(r, Y[1,:], rtol=1e-8)
end



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
    @test isapprox(FY, FFTTest*fk, rtol=1e-10)
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
    @test isapprox(FY_mul, FFTTest*fk_k2, rtol=1e-10)
end
