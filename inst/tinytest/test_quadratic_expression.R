
library(RcppFastAD)

X <- c(0.5, 0.6)
S <- matrix(c(2, 3, 3, 6), 2, 2)
res <- quadratic_expression(X, S)
expect_equal(res[["value"]][1,1], 4.46)
expect_equal(res[["gradient"]][,1], c(5.6,10.2))

X <- c(0.0, 0.0)
res <- quadratic_expression(X, S)
expect_equal(res[["value"]][1,1], 0.0)
expect_equal(res[["gradient"]][,1], c(0.0, 0.0))
