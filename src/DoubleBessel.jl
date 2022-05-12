

@kwdef mutable struct DoubleBesselPlan{T,C} <: AbstractPlan
    x::Vector{T}
    a::Array{T, 4} = zeros(10, 10, 10, 10)
    ϕat::Array{T, 4} = zeros(10, 10, 10, 10)
    t::Vector{T} = zeros(10)
    hm::Array{C, 4} = zeros(ComplexF64, 10, 10, 10, 10)
    hm_corr::Array{C, 4} = zeros(ComplexF64, 10, 10, 10, 10)
    d_ln_x::T = log(x[2] / x[1])
    ϕat_corr::Array{T, 4} = zeros(10, 10, 10, 10)
    original_length::Int = length(x)
    gll::Array{C, 4} = zeros(ComplexF64, 10, 10, 10, 10)
    ν::T = 1.01
    n_extrap_low::Int = 0
    n_extrap_high::Int = 0
    c_window_width::T = 0.25
    n_pad::Int = 0
    n::Int = 0
    N::Int = original_length + n_extrap_low + n_extrap_high + 2 * n_pad
    m::Vector{T} = zeros(N)
    cm::Vector{C} = zeros(ComplexF64, N)
    ηm::Vector{T} = zeros(N)
    plan_rfft::FFTW.rFFTWPlan = plan_rfft(randn(1024))
    plan_irfft = begin
        NNN = original_length + n_extrap_low + n_extrap_high + 2 * n_pad
        plan_irfft(
            randn(Complex{Float64}, 2, 2, 2, Int(NNN / 2) + 1),
            NNN,
            4
        )
        end
end


function my_hyp2f1(a, b, c, x) 
    convert(ComplexF64, 
        #hypergeometric_2f1(
        hyp2f1(
            AcbField(128)(real(a), imag(a)),
            AcbField(128)(real(b), imag(b)),
            AcbField(128)(real(c), imag(c)),
            AcbField(128)(real(x), imag(x))
        )
    )
end


function _eval_gll(l1, l2, t, zz::Vector)
    return [2^(z - 1) * √π * gamma((l1 + l2 + z) / 2) /
            (gamma((z - 1 + l2 - l1) / 2) * gamma(l2 + 3 / 2)) *
            t^l2 * my_hyp2f1(
                0.5 * (z - 1 + l2 - l1),
                0.5 * (l1 + l2 + z),
                l2 + 1.5,
                t^2
            )
            for z in zz]

    #=[((-1)^n) * 2 ^ (z - n) * gamma((ell + z - n) / 2) /
         gamma((3 + ell + n - z) / 2)
        for z in zz]
        =#
end


function _eval_a!(plan::DoubleBesselPlan, l1::Vector, l2::Vector, t::Vector)

    plan.a = zeros(length(l1), length(l2), length(t), length(plan.x))
    plan.ϕat_corr = zeros(size(plan.a))
    plan.ϕat = zeros(size(plan.a))

    reverse!(plan.x)

    @inbounds for mal1 in 1:length(l1), mal2 in 1:length(l2), mt in 1:length(t)
        plan.a[mal1, mal2, mt, :] = 1.0 ./ plan.x
        #println("plan.a[mal1, mal2, mt, :] = $(plan.a[mal1, mal2, mt, :])")
        plan.ϕat_corr[mal1, mal2, mt, :] =
            plan.a[mal1, mal2, mt, :] .^ (-plan.ν) .* sqrt(π) ./ 4
        #println("plan_ϕat_corr[mal1, mal2, mt, :] = $(plan.ϕat_corr[mal1, mal2, mt, :])")
    
    end

    reverse!(plan.x)
end



function _eval_gll_hm!(plan::DoubleBesselPlan, l1::Vector, l2::Vector, t::Vector)
    z = plan.ν .+ im .* plan.ηm
    plan.gll = zeros(length(l1), length(l2), length(t), length(z))
    plan.hm = zeros(ComplexF64, size(plan.gll))

    plan.hm_corr = zeros(ComplexF64, length(l1), length(l2), length(t), Int(plan.N / 2 + 1))

    @inbounds for mal1 in 1:length(l1), mal2 in 1:length(l2), mt in 1:length(t)
        plan.hm_corr[mal1, mal2, mt, :] =
            (plan.x[1] .* plan.a[mal1, mal2, mt, 1]) .^ (-im .* plan.ηm)
        #println("l1, l2, t = ", l1[mal1]," , ", l2[mal2], ", ", t[mt])
        plan.gll[mal1, mal2, mt, :] = _eval_gll(l1[mal1], l2[mal2], t[mt], z)
        #println("gll = $(plan.gll[1,1,1,:])")
    end
end


function prepare_FFTLog!(plan::DoubleBesselPlan, l1::Vector, l2::Vector, t::Vector)
    plan.x = _logextrap(plan.x, plan.n_extrap_low + plan.n_pad,
        plan.n_extrap_high + plan.n_pad)

    _eval_a!(plan, l1, l2, t)

    plan.m = Array(0:length(plan.x)/2)
    _eval_ηm!(plan)

    _eval_gll_hm!(plan, l1, l2, t)

    plan.plan_rfft = plan_rfft(plan.x)
    NNN = plan.original_length + plan.n_extrap_low + plan.n_extrap_high + 2 * plan.n_pad
    plan.plan_irfft = plan_irfft(
        randn(Complex{Float64}, length(l1), length(l2), length(t), Int((length(plan.x) / 2) + 1)),
        NNN, 4
        )
end



function get_y(plan::DoubleBesselPlan)
    n1 = plan.n_extrap_low + plan.n_pad + 1
    n2 = plan.n_extrap_low + plan.n_pad + plan.original_length
    return plan.a[:, :, :, n1:n2]
end


function evaluate_FFTLog!(ϕat, plan::DoubleBesselPlan, fx)
    fx = _logextrap(fx, plan.n_extrap_low, plan.n_extrap_high)
    fx = _zeropad(fx, plan.n_pad)
    _eval_cm!(plan, fx)

    #println("\n plan.cm = $(plan.cm) \n")
    #println("\n plan.gll = $(plan.gll[1,1,1,:]) \n")
    #println("\n plan.hm_corr = $(plan.hm_corr[1,1,1,:]) \n")

    @inbounds for mal1 in 1:size(plan.a)[1], mal2 in 1:size(plan.a)[2], mt in 1:size(plan.a)[3]
        plan.hm[mal1, mal2, mt, :] = plan.cm .* @view plan.gll[mal1, mal2, mt, :]
        plan.hm[mal1, mal2, mt, :] .*= @view plan.hm_corr[mal1, mal2, mt, :]
    end

    #println("\n plan.hm = $(plan.hm[1,1,1,:]) \n")

    plan.ϕat[:, :, :, :] .= plan.plan_irfft * conj!(plan.hm)
    println("\n plan.phiat = $(plan.ϕat[1,1,1,:]); \n")
    #println("\n plan.hm = $(plan.hm[1,1,1,:]) \n")
    plan.ϕat[:, :, :, :] .*= @view plan.ϕat_corr[:, :, :, :]

    n1 = plan.n_extrap_low + plan.n_pad + 1
    n2 = plan.n_extrap_low + plan.n_pad + plan.original_length
    ϕat[:, :, :, :] .= @view plan.ϕat[:, :, :, n1:n2]
end
