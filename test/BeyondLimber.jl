

@testset "BeyondLimber checks" begin
     FFTTest = FFTLog.SingleBesselPlan(x=k, ν=1.01, n_extrap_low=1500, n_extrap_high=1500,
          n_pad=2000)
     Ell = Array([1.0, 2.0])
     FFTLog.prepare_FFTLog!(FFTTest, Ell)
     Y = FFTLog.get_y(FFTTest)
     FY = FFTLog.evaluate_FFTLog(FFTTest, Pk)
     Fr = npzread(input_path * "check_by_Fr.py.npy")
     r = npzread(input_path * "check_by_r.py.npy")

     @test isapprox(Fr, FY[1, :], rtol=1e-5)
     @test isapprox(r, Y[1, :], rtol=1e-5)

     FFTTest = FFTLog.SingleBesselPlan(x=k, ν=1.01, n_extrap_low=1500, n_extrap_high=1500,
          n_pad=2000, n=1)
     Ell = Array([1.0, 2.0])
     FFTLog.prepare_FFTLog!(FFTTest, Ell)
     Y = FFTLog.get_y(FFTTest)
     FY = FFTLog.evaluate_FFTLog(FFTTest, Pk)
     Fr1 = npzread(input_path * "check_by_Fr1.py.npy")
     r1 = npzread(input_path * "check_by_r1.py.npy")

     @test isapprox(Fr1, FY[1, :], rtol=1e-8)
     @test isapprox(r1, Y[1, :], rtol=1e-8)

     FFTTest = FFTLog.SingleBesselPlan(x=k, ν=1.01, n_extrap_low=1500, n_extrap_high=1500,
          n_pad=2000, n=2)
     Ell = Array([1.0, 2.0])
     FFTLog.prepare_FFTLog!(FFTTest, Ell)
     Y = FFTLog.get_y(FFTTest)
     FY = FFTLog.evaluate_FFTLog(FFTTest, Pk)
     Fr2 = npzread(input_path * "check_by_Fr2.py.npy")
     r2 = npzread(input_path * "check_by_r2.py.npy")

     @test isapprox(Fr2, FY[1, :], rtol=1e-8)
     @test isapprox(r2, Y[1, :], rtol=1e-8)

     FFTTest = FFTLog.HankelPlan(x=k, n_extrap_low=1500, ν=1.01, n_extrap_high=1500,
          n_pad=0)
     Ell = Array([0.0, 2.0])
     FFTLog.prepare_Hankel!(FFTTest, Ell)
     Y = FFTLog.get_y(FFTTest)
     FY = FFTLog.evaluate_Hankel(FFTTest, Pk .* (k .^ (-2)))
     Fr = npzread(input_path * "check_by_hank_Fr.py.npy")
     r = npzread(input_path * "check_by_hank_r.py.npy")
     @test isapprox(Fr, FY[1, :], rtol=1e-8)
     @test isapprox(r, Y[1, :], rtol=1e-8)
end