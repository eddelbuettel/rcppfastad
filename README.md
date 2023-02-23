  
## RcppFastAD: Rcpp Bindings for the FastAD C++ Header-Only Library

[![CI](https://github.com/eddelbuettel/rcppfastad/workflows/ci/badge.svg)](https://github.com/eddelbuettel/rcppfastad/actions?query=workflow%3Aci)
[![License](https://eddelbuettel.github.io/badges/GPL2+.svg)](https://www.gnu.org/licenses/gpl-2.0.html)
<!-- [![CRAN](https://www.r-pkg.org/badges/version/RcppFastAD)](https://cran.r-project.org/package=RcppFastAD) -->
<!-- [![Dependencies](https://tinyverse.netlify.com/badge/RcppFastAD)](https://cran.r-project.org/package=RcppFastAD) -->
<!-- [![Downloads](https://cranlogs.r-pkg.org/badges/RcppFastAD?color=brightgreen)](https://www.r-pkg.org/pkg/RcppAD) -->
[![Last Commit](https://img.shields.io/github/last-commit/eddelbuettel/rcppfastad)](https://github.com/eddelbuettel/rcppfastad)

### Motivation

[FastAD](https://github.com/JamesYang007/FastAD) is a header-only C++ template library for automatic differentiation
supporting both forward and reverse mode.  It utilizes the latest features in C++17 and expression templates for
efficient computation. See the [FastAD](https://github.com/JamesYang007/FastAD) repo for more.

This package brings this header-only library to R so that other R user can access it simply by
adding `LinkingTo: RcppFastAD`.

### Example

Three examples, taken from FastAS, are included. We can look at the `black_scholes()` one here:

```r
> library(RcppFastAD)
> black_scholes()
       value     delta    vega      rho   theta
call 56.5136  0.773818 9.05493  2.03321 275.730
put  51.4109 -0.226182 9.05493 -6.17753 274.481
> 
```

This evaluates a put and a call struck at 100 with spot at 105, and some default values (all taken from [an example
included with Boost](https://www.boost.org/doc/libs/master/libs/math/doc/html/math_toolkit/autodiff.html#math_toolkit.autodiff.example-black_scholes)).
The values can be set in the call too.  Returned all the value along the first partial derivatives relative to spot,
volatility, short rate and time to maturity---which are all calculated using automatic differentiation.

(FastAD has a focus on speed leading to some design choices that make taking _second_ derivatives harder. So no 'gamma' here.)

### Status

The package is complete and contains a mature version of FastAD.

### Contributing

Any problems, bug reports, or features requests for the package can be submitted and handled most
conveniently as [Github issues](https://github.com/eddelbuettel/rcppfastAD/issues) in the
repository.

Before submitting pull requests, it is frequently preferable to first discuss need and scope in such
an issue ticket.  See the file
[Contributing.md](https://github.com/RcppCore/Rcpp/blob/master/Contributing.md) (in the
[Rcpp](https://github.com/RcppCore/Rcpp) repo) for a brief discussion.

### AuthorS

For the R package, [Dirk Eddelbuettel](https://github.com/eddelbuettel).

For everything pertaining to FastAD: [James Yang](https://github.com/JamesYang007).

