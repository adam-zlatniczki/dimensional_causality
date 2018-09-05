library(dimensionalcausality)

set.seed(0)
x <- runif(10000)
y <- runif(10000)
k_range <- seq(10, 40, 2)

ret <- infer_causality(x, y, 4, 1, k_range)
print(ret$probs)