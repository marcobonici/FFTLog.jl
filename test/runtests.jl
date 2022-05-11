using FFTLog
using Test
using DelimitedFiles
using NPZ

#=
run(`wget https://zenodo.org/record/6021744/files/test_FFTLog.tar.xz\?download=1`)
run(`mv test_FFTLog.tar.xz\?download=1 test_FFTLog.tar.xz`)
run(`tar xvf test_FFTLog.tar.xz`)
=#


input_path = pwd() * "/test_FFTLog/"


read_file = readdlm(input_path * "Pk_test")
k = read_file[:, 1]
Pk = read_file[:, 2]

function f(k, a=1)
    return exp(-k^2.0 * a^2 / 2)
end

function F(r, a=1)
    return exp(-r^2.0 / (2.0 * a^2))
end


function LogSpaced(min::T, max::T, n::I) where {T,I}
    logmin = log10(min)
    logmax = log10(max)
    logarray = Array(LinRange(logmin, logmax, n))
    return exp10.(logarray)
end

include("./BeyondLimber.jl")
include("./AnalyticalHankelTest.jl")
include("./mulTest.jl")

include("./DoubleBessel.jl")

