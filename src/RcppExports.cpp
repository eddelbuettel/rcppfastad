// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// black_scholes_example
void black_scholes_example();
RcppExport SEXP _RcppFastAD_black_scholes_example() {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    black_scholes_example();
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RcppFastAD_black_scholes_example", (DL_FUNC) &_RcppFastAD_black_scholes_example, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_RcppFastAD(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
