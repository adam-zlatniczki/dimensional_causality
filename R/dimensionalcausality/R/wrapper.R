infer_causality <- function(x, y, emb_dim, tau, k_range, eps=0.05, c=3.0, bins=20.0, downsample_rate=1){

  n <- length(x)
  len_range <- length(k_range)

  probs <- c(0, 0, 0, 0, 0)

  ret <- .C("infer_causality_R",
     probs=probs,
     as.numeric(x),
     as.numeric(y),
     as.integer(n),
     as.integer(emb_dim),
     as.integer(tau),
     as.integer(k_range),
     as.integer(len_range),
     as.numeric(eps),
     as.numeric(c),
     as.numeric(bins),
     as.integer(downsample_rate)
  )

  return(ret$probs)
}
