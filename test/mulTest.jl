

@testset "mul! operator test" begin
     k = LogSpaced(10^(-5), 10.0, 2^10)
     fk = f.(k)
     FFTTest = FFTLog.SingleBesselPlan(x=k, n_extrap_low=1500, ν=1.01, n_extrap_high=1500,
          n_pad=500)
     Ell = Array([0.0])
     prepare_FFTLog!(FFTTest, Ell)
     Y = FFTLog.get_y(FFTTest)
     FY = zeros(size(Y))
     fk_k2 = fk .* (k .^ 2)
     FY = evaluate_FFTLog(FFTTest, fk_k2)
     FY_mul = zeros(size(FY))
     mul!(FY_mul, FFTTest, fk_k2)
     @test isapprox(FY_mul, FY, rtol=1e-10)
     @test isapprox(FY_mul, FFTTest * fk_k2, rtol=1e-10)
end
