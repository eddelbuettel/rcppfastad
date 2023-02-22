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

    Eigen::MatrixXd theta_adj(2, 1);
    theta_adj.setZero(); // Set adjoints to zeros.

    // Initialize variable.
    ad::VarView<double, ad::mat> theta(theta_hat.data(), theta_adj.data(), 2, 1);

    // Create expr. Use a row buffer to store data. Then we only need to manipulate data when
    // looping.
    Eigen::MatrixXd x_row_buffer = X.row(0);
    auto xi = ad::constant_view(x_row_buffer.data(), 1, X.cols());
    Eigen::MatrixXd y_row_buffer = y.row(0);
    auto yi = ad::constant_view(y_row_buffer.data(), 1, y.cols());
    auto expr = ad::bind(ad::pow<2>(yi - ad::dot(xi, theta)));

    // Seed
    Eigen::MatrixXd seed(1, 1);
    seed.setOnes(); // Usually seed is 1. DONT'T FORGET!

    // Loop over each row to calulate loss.
    double loss = 0;
    for (int i = 0; i < X.rows(); ++i) {
        x_row_buffer = X.row(i);
        y_row_buffer = y.row(i);

        auto f = ad::autodiff(expr, seed.array());
        loss += f.coeff(0);
    }

    Rcpp::List res = Rcpp::List::create(Rcpp::Named("loss") = loss,
                                        Rcpp::Named("gradient") = theta.get_adj());

    theta_adj.setZero(); // Reset differential to zero after one full pass.

    return res;
}
