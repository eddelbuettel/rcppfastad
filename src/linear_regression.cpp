#include <RcppEigen.h>
#include <fastad>

//' Evaluate a squared-loss linear regression at a given parameter value
//'
//' Not that this function does not actually fit the model. Rather it evaluates
//' the squared sum of residuals and \sQuote{gradient} of parameters.
//'
//' @param theta_hat Vector of parameters
//' @param y Vector with dependent variable
//' @param X Matrix with independent explanatory variables
//' @return A list object with the \sQuote{loss} and \sQuote{gradient}
//' @examples
//' beta_guess <- c(1,2)
//' y <- c(32,64,96,128,160)
//' X <- matrix(c(1,10,2,20,3,30,4,40,5,50), ncol=2, byrow=TRUE)
//' linear_regression(beta_guess, y, X)
// [[Rcpp::export]]
Rcpp::List linear_regression(Eigen::Map<Eigen::VectorXd> theta_hat,
                             Eigen::Map<Eigen::VectorXd> y,
                             Eigen::Map<Eigen::MatrixXd> X) {

    const auto p = theta_hat.size();
    Eigen::VectorXd theta_adj(p);
    theta_adj.setZero(); // Set adjoints to zeros.

    // Initialize variable.
    ad::VarView<double, ad::vec> theta(theta_hat.data(), theta_adj.data(), p);

    // Create expr. Use a row buffer to store data. Then we only need to manipulate data when
    // looping.
    auto expr = ad::bind(ad::norm(y - ad::dot(X, theta)));

    // Differentiate and get loss
    auto loss = ad::autodiff(expr);

    Rcpp::List res = Rcpp::List::create(Rcpp::Named("loss") = loss,
                                        Rcpp::Named("gradient") = theta.get_adj());

    theta_adj.setZero(); // Reset differential to zero after one full pass.

    return res;
}
