# FFTLog.jl
[![Build status (Github Actions)](https://github.com/marcobonici/FFTLog.jl/workflows/CI/badge.svg)](https://github.com/marcobonici/FFTLog.jl/actions) [![codecov](https://codecov.io/gh/marcobonici/FFTLog.jl/branch/main/graph/badge.svg?token=RCMDNON0JD)](https://codecov.io/gh/marcobonici/FFTLog.jl)

Package to compute integrals involving Bessels functions such as


<img src="https://latex.codecogs.com/svg.image?F(y)=\int_{0}^{\infty}&space;\frac{d&space;x}{x}&space;f(x)&space;j_{\ell}(x&space;y)" title="F(y)=\int_{0}^{\infty} \frac{d x}{x} f(x) j_{\ell}(x y)" />

or Hankel transforms

<img src="https://latex.codecogs.com/svg.image?F(y)=\int_{0}^{\infty}&space;\mathrm{d}x&space;f(x)&space;J_{n}(x&space;y)&space;" title="F(y)=\int_{0}^{\infty} \mathrm{d}x f(x) J_{n}(x y) " />

The integrals are performed using the FFTLog algorithm ([Hamilton 2000](https://arxiv.org/abs/astro-ph/9905191)).

## Usage

Given an array *logarithmically spaced* `r` and a function `f` evaluated over this array, we
want to evaluate the Hankel transform for the multipole values contained in the array `Ell`.
For instance, let us consider the following Hankel pair

<img src="https://latex.codecogs.com/svg.image?F_{0}(r)=\int_{0}^{\infty}&space;e^{-\frac{1}{2}&space;a^{2}&space;k^{2}}&space;J_{0}(k&space;r)&space;k&space;\mathrm{~d}&space;k=e^{-\frac{r^{2}}{2&space;a^{2}}}" title="F_{0}(r)=\int_{0}^{\infty} e^{-\frac{1}{2} a^{2} k^{2}} J_{0}(k r) k \mathrm{~d} k=e^{-\frac{r^{2}}{2 a^{2}}}" />

Since we know the analytical transform, we can perform a check

1. Instantiate an object `HankelPlan`
```julia
HankelTest = FFTLog.HankelPlan(XArray = k)
```
2. Perform some precomputations
```julia
FFTLog.PrepareHankel!(HankelTest, Ell)
```
3. Compute the Hankel transform
```julia
Fy = FFTLog.evaluateHankel(HankelTest, Pk)
```
4. If needed, the array `y` (the counterpart of the array `r`) can be obtained with
```julia
y = FFTlog.getY(HankelTest)
```
Now, let us compare the numerical and the analytical transforms

![analytical_check](https://user-images.githubusercontent.com/58727599/151894066-f10a5be0-e259-4762-aa48-a5799fda0458.png)

We can also plot the relative difference

![analytical_residuals](https://user-images.githubusercontent.com/58727599/151894064-c620532d-36ce-416b-a592-7612cb95f396.png)

Quite good, isn't it?

## Roadmap

Step | Status| Comment
:------------ | :-------------| :-------------
Integrals with a Bessel function | :white_check_mark: | Implemented, need some polishing
Hankel Transform | :white_check_mark: | Implemented, need some polishing 
Integrals with a Bessel derivative | :construction: | WIP
Automatic Differentiation| :construction: | WIP
GPU compatibility| :construction: | WIP
Integrals with multiple Bessel function| :construction: | WIP

## Authors

- [Marco Bonici](https://www.github.com/marcobonici), PostoDoctoral Researcher at INAF-IASF Milano