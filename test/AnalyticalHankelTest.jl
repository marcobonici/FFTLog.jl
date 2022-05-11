
@testset "Analytical Hankel test" begin
    FFTLog.set_num_threads(Threads.nthreads())
    k = LogSpaced(10^(-5), 10.0, 2^10)
    fk = f.(k)
    FFTTest = FFTLog.HankelPlan(x=k, n_extrap_low=1500, ν=1.01, n_extrap_high=1500,
        n_pad=500)
    Ell = Array([0.0])
    prepare_Hankel!(FFTTest, Ell)
    Y = get_y(FFTTest)
    FY = zeros(size(Y))
    FY = evaluate_Hankel(FFTTest, fk)
    FY_mul = zeros(size(FY))
    Fr_analytical = F.(Y[1, :])
    @test isapprox(Fr_analytical, FY[1, :], rtol=1e-10)
    @test isapprox(FY, mul!(FY_mul, FFTTest, fk), rtol=1e-10)
    @test isapprox(FY, FFTTest * fk, rtol=1e-10)
end
