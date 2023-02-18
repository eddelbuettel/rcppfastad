// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// black_scholes
Rcpp::NumericMatrix black_scholes(double spot, double strike, double vol, double r, double tau);
RcppExport SEXP _RcppFastAD_black_scholes(SEXP spotSEXP, SEXP strikeSEXP, SEXP volSEXP, SEXP rSEXP, SEXP tauSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type spot(spotSEXP);
    Rcpp::traits::input_parameter< double >::type strike(strikeSEXP);
    Rcpp::traits::input_parameter< double >::type vol(volSEXP);
    Rcpp::traits::input_parameter< double >::type r(rSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    rcpp_result_gen = Rcpp::wrap(black_scholes(spot, strike, vol, r, tau));
    return rcpp_result_gen;
END_RCPP
}
// quadratic_expression
Rcpp::List quadratic_expression(Eigen::Map<Eigen::VectorXd> X, Eigen::Map<Eigen::MatrixXd> Sigma);
RcppExport SEXP _RcppFastAD_quadratic_expression(SEXP XSEXP, SEXP SigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::VectorXd> >::type X(XSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::MatrixXd> >::type Sigma(SigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(quadratic_expression(X, Sigma));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_RcppFastAD_black_scholes", (DL_FUNC) &_RcppFastAD_black_scholes, 5},
    {"_RcppFastAD_quadratic_expression", (DL_FUNC) &_RcppFastAD_quadratic_expression, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_RcppFastAD(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
