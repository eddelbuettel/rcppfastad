
library(RcppFastAD)

## reference
data(trees, package="datasets")   # also used in help(lm)
X <- as.matrix(cbind(const=1, log(trees[, c("Girth", "Height")])))
y <- log(trees$Volume)
ref <- coef(lm(y ~ X - 1))  # for comparison

## compare to least squared minimization via FastAD and gradient search
fit <- linear_regression(X, y, rep(0, 3), tol=1e-6)
expect_equal(fit[["theta"]], unname(ref), tol=1e-2)

fit <- linear_regression(X, y, rep(0, 3), tol=1e-9)
expect_equal(fit[["theta"]], unname(ref), tol=1e-4)



