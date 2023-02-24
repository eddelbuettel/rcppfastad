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
Rcpp::List linear_regression(Eigen::Map<Eigen::MatrixXd> X,
                             Eigen::Map<Eigen::VectorXd> y,
                             Eigen::Map<Eigen::VectorXd> theta_hat,
                             double initial_lr=0.1,
                             size_t max_iter=100,
                             double tol=1e-7) {
    // Perform gradient descent (Barzilai-Borwein Method):
    // https://en.wikipedia.org/wiki/Gradient_descent#Description
    
    const auto p = theta_hat.size();
    Eigen::VectorXd theta_hat_prev(p);
    Eigen::VectorXd theta_hat_curr = theta_hat;
    Eigen::VectorXd theta_adj_prev(p);
    Eigen::VectorXd theta_adj_curr(p);

    // Set adjoints to zeros.
    theta_adj_curr.setZero(); 

    // Initialize variable.
    ad::VarView<double, ad::vec> theta(theta_hat_curr.data(), theta_adj_curr.data(), p);

    // Create expr. 
    auto expr = ad::bind(ad::norm(y - ad::dot(X, theta)));
    
    // Get loss value and differentiate.
    double prev_loss = std::numeric_limits<double>::infinity();
    double loss = ad::autodiff(expr);
    
    const auto update_invariant = [&](auto lr) {
        // MUST write into prev 
        theta_hat_prev = theta_hat_curr;
        theta_adj_prev = theta_adj_curr;
        
        // Reset adjoint
        theta_adj_curr.setZero();
        
        // Descend
        theta_hat_curr = theta_hat_prev - lr * theta_adj_prev;
    };
    
    // Perform 1-step before looping
    update_invariant(initial_lr);
    
    size_t iter = 0;
    while ((iter < max_iter) && (std::abs(loss-prev_loss) >= tol)) {
        prev_loss = loss;
        loss = ad::autodiff(expr); 
        
        // compute smart learning rate
        const auto dtheta = theta_hat_curr - theta_hat_prev;
        const auto ddf = theta_adj_curr - theta_adj_prev;
        const auto gamma = std::abs(dtheta.dot(ddf)) / ddf.squaredNorm();

        update_invariant(gamma);

        ++iter;
    }

    Rcpp::List res = Rcpp::List::create(Rcpp::Named("loss") = loss,
                                        Rcpp::Named("theta") = theta_hat_prev,
                                        Rcpp::Named("gradient") = theta_adj_prev,
                                        Rcpp::Named("iter") = iter);

    return res;
}

