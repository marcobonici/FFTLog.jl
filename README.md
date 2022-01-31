# FFTLog.jl
[![Build status (Github Actions)](https://github.com/marcobonici/FFTLog.jl/workflows/CI/badge.svg)](https://github.com/marcobonici/FFTLog.jl/actions) [![codecov](https://codecov.io/gh/marcobonici/FFTLog.jl/branch/main/graph/badge.svg?token=RCMDNON0JD)](https://codecov.io/gh/marcobonici/FFTLog.jl)

Package to compute integrals involving Bessels functions such as


<img src="https://latex.codecogs.com/svg.image?F(y)=\int_{0}^{\infty}&space;\frac{d&space;x}{x}&space;f(x)&space;j_{\ell}(x&space;y)" title="F(y)=\int_{0}^{\infty} \frac{d x}{x} f(x) j_{\ell}(x y)" />

The integrals are performed using the FFTLog algorithm ([Hamilton 2000](https://arxiv.org/abs/astro-ph/9905191)).
