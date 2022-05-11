module FFTLog

using FFTW
using Base: @kwdef
using SpecialFunctions: gamma
using Nemo: AcbField, ArbField, hypergeometric_2f1 #, hyp2f1 
import Base: *

export prepare_FFTLog!, evaluate_FFTLog, evaluate_FFTLog!
export prepare_Hankel!, evaluate_Hankel, evaluate_Hankel!
export mul!
export get_y




include("./common.jl")
include("./SingleBessel.jl")
include("./DoubleBessel.jl")


##########################################################################################92



function mul!(Y, Q::Union{SingleBesselPlan,DoubleBesselPlan}, A)
    evaluate_FFTLog!(Y, Q, A)
end

function mul!(Y, Q::HankelPlan, A)
    Y[:, :] .= evaluate_Hankel!(Y, Q, A)
end

function *(Q::Union{SingleBesselPlan,DoubleBesselPlan}, A)
    evaluate_FFTLog(Q, A)
end

function *(Q::HankelPlan, A)
    evaluate_Hankel(Q, A)
end

end # module
