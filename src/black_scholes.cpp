
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
template <option_type cp, class Price, class Vol, class Cache>
auto black_scholes_option_price(const Price& S,
                                double K,
                                const Vol& sigma,
                                double tau,
                                double r,
                                Cache& cache)
{
    cache.resize(3);
    double PV = K * std::exp(-r * tau);
    auto common_expr = (
            cache[0] = ad::log(S / K),
            cache[1] = (cache[0] + ((r + sigma * sigma / 2.) * tau)) / 
                            (sigma * std::sqrt(tau)),
            cache[2] = cache[1] - (sigma * std::sqrt(tau))
    );
    if constexpr (cp == option_type::call) {
        return (common_expr,
                Phi(cache[1]) * S - Phi(cache[2]) * PV);
    } else {
        return (common_expr,
                Phi(-cache[2]) * PV - Phi(-cache[1]) * S);
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
    ad::Var<double> Spot(spot);
    ad::Var<double> Vol(vol);
    std::vector<ad::Var<double>> cache;

    Rcpp::NumericMatrix M(2,3);
    Rcpp::rownames(M) = Rcpp::CharacterVector::create("call", "put");
    Rcpp::colnames(M) = Rcpp::CharacterVector::create("value", "delta", "vega");

    auto call_expr = (black_scholes_option_price<option_type::call>(Spot, strike, Vol, tau, r, cache));
    auto size_pack_c = call_expr.bind_cache_size();
    std::vector<double> del_buf_c(size_pack_c(0));
    std::vector<double> veg_buf_c(size_pack_c(1));
    call_expr.bind_cache({del_buf_c.data(), veg_buf_c.data()});
    double call_price = ad::autodiff(call_expr);
    M.row(0) = Rcpp::NumericVector::create(call_price, Spot.get_adj(), Vol.get_adj());

    // reset adjoints before differentiating again
    Spot.reset_adj();
    Vol.reset_adj();
    for (auto& c : cache) c.reset_adj();

    auto put_expr = (black_scholes_option_price<option_type::put>(Spot, strike, Vol, tau, r, cache));
    auto size_pack_p = put_expr.bind_cache_size();
    std::vector<double> del_buf_p(size_pack_p(0));
    std::vector<double> veg_buf_p(size_pack_p(1));
    put_expr.bind_cache({del_buf_p.data(), veg_buf_p.data()});
    double put_price = ad::autodiff(put_expr);
    M.row(1) = Rcpp::NumericVector::create(put_price, Spot.get_adj(), Vol.get_adj());

    return(M);
}
