
#include <RcppEigen.h>
#include <fastad>

enum class option_type {
    call, put
};

// Standard Normal CDF
template <class T>
inline auto Phi(const T& x)
{
    return 0.5 * (ad::erf(x / std::sqrt(2.)) + 1.);
}

// Generates expression that computes Black-Scholes option price
template <option_type cp, class Price, class Vol, class Rho, class Tau, class Cache>
auto black_scholes_option_price(const Price& S,
                                double K,
                                const Vol& sigma,
                                const Rho& r,
                                const Tau& tau,
                                Cache& cache)
{
    cache.resize(4);
    auto common_expr = (cache[0] = K * ad::exp(-r * tau), 	// PV
                        cache[1] = ad::log(S / K),
                        cache[2] = (cache[1] + ((r + sigma * sigma / 2.) * tau)) / (sigma * ad::sqrt(tau)),
                        cache[3] = cache[2] - (sigma * ad::sqrt(tau))
                        );
    if constexpr (cp == option_type::call) {
        return (common_expr, Phi(cache[2]) * S - Phi(cache[3]) * cache[0]);
    } else {
        return (common_expr, Phi(-cache[3]) * cache[0] - Phi(-cache[2]) * S);
    }
}

//' Black-Scholes valuation and first derivatives via Automatic Differentiation
//'
//' This example illustrate how to use automatic differentiation to
//' calculate the delte of a Black-Scholes call and put. It is based
//' on the same example in the FastAD sources.
//' @param spot A double with the spot price, default is 105 as in Boost example
//' @param strike A double with the strike price, default is 100 as in Boost example
//' @param vol A double with the (annualized) volatility (in percent), default is 5
//' (for 500 per cent) as in Boost example
//' @param r A double with the short-term risk-free rate, default is 0.0125 as in Boost example
//' @param tau A double with the time to expiration (in fractional years), default is 30/365
//' as in Boost example
//' @return A matrix with rows for the call and put variant, and columns
//' for option value, delta and vega
//' @examples
//' black_scholes()
// [[Rcpp::export]]
Rcpp::NumericMatrix black_scholes(double spot = 105,   		// current spot
                                  double strike = 100, 		// strike
                                  double vol = 5,               // annual vol (500% !!)
                                  double r = 1.25 / 100,        // short-term rate
                                  double tau = 30.0 / 365) {	// time remaining
    ad::Var<double> Delta(spot);
    ad::Var<double> Vol(vol);
    ad::Var<double> Rho(r);
    ad::Var<double> Theta(tau);
    std::vector<ad::Var<double>> cache;

    Rcpp::NumericMatrix M(2,5);
    Rcpp::rownames(M) = Rcpp::CharacterVector::create("call", "put");
    Rcpp::colnames(M) = Rcpp::CharacterVector::create("value", "delta", "vega", "rho", "theta");

    auto call_expr = ad::bind(black_scholes_option_price<option_type::call>(Delta, strike, Vol, Rho, Theta, cache));
    double call_price = ad::autodiff(call_expr);
    M.row(0) = Rcpp::NumericVector::create(call_price, Delta.get_adj(), Vol.get_adj(), Rho.get_adj(), Theta.get_adj());

    // reset adjoints before differentiating again
    Delta.reset_adj();
    Vol.reset_adj();
    Rho.reset_adj();
    Theta.reset_adj();
    for (auto& c : cache) c.reset_adj();

    auto put_expr = ad::bind(black_scholes_option_price<option_type::put>(Delta, strike, Vol, Rho, Theta, cache));
    double put_price = ad::autodiff(put_expr);
    M.row(1) = Rcpp::NumericVector::create(put_price, Delta.get_adj(), Vol.get_adj(), Rho.get_adj(), Theta.get_adj());

    return(M);
}
