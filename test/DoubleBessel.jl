
@testset "test Gauss Hypergeometric 2F1" begin
     RTOL, ATOL = 1e-4, 0.0
     @test isapprox(FFTLog.my_hyp2f1(1, 2, 3, 4),
          -0.5 - 0.125 * log(3) - im * 0.125 * π; rtol=RTOL, atol=ATOL)
     @test isapprox(FFTLog.my_hyp2f1(1.65 - 1.05 * im, 2.15 - 1.05 * im, 1.5, 0.25),
          1.1959 - 1.16374 * im; rtol=RTOL, atol=ATOL)

end

@testset "test _eval_gll" begin
     RTOL, ATOL = 1e-4, 0.0
     @test isapprox(FFTLog._eval_gll(0, 0, 0.5, [0.56 - im * 10.81585942936043])[1,1,1,1],
          0.350758 + 0.327482*im; rtol=RTOL, atol=ATOL)
end
