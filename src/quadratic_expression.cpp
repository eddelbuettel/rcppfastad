
#include <RcppEigen.h>
#include <fastad>

//' Compute the value and derivate of a quadratic expression X' * Sigma * X
//'
//' @param X A 2 element vector
//' @param Sigma A 2 x 2 matrix
//' @return A list with two elements for the expression evaluated for X and Sigma as well as
//' @examples
//' X <- c(0.5, 0.6)
//' S <- matrix(c(2, 3, 3, 6), 2, 2)
//' quadratic_expression(X, S)
// [[Rcpp::export]]
Rcpp::List quadratic_expression(Eigen::Map<Eigen::VectorXd> X,
                                Eigen::Map<Eigen::MatrixXd> Sigma) {

    Eigen::VectorXd x_adj(2);
    x_adj.setZero(); // Set adjoints to zeros.

    // Initialize variable.
    ad::VarView<double, ad::vec> x(X.data(), x_adj.data(), 2);

    // Initialize (constant) Sigma matrix as a view.
    auto CSigma = ad::constant_view(Sigma.data(), Sigma.rows(), Sigma.cols());

    // Quadratic expression: x^T*Sigma*x
    auto expr = ad::bind(ad::dot(ad::dot(ad::transpose(x), CSigma), x));
    // Seed
    Eigen::Array<double, 1, 1> seed(1);
    auto f = ad::autodiff(expr, seed);

    Rcpp::List res = Rcpp::List::create(Rcpp::Named("value") = f(0,0),
                                        Rcpp::Named("gradient") = x.get_adj());

    return res;
}
