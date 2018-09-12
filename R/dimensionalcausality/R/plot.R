plot_probabilities <- function(final_probabilities){
  final_probabilities <- round(final_probabilities, 2)
  final_probabilities[5] <- 1 - sum(final_probabilities[1:4])
  df <- data.frame(case=c("X -> Y", "X <-> Y", "X <- Y", "X cc Y", "X | Y"), probability=final_probabilities)
  df$case <- factor(df$case, levels = df$case)
  
  bp <- ggplot(data=df, aes(x=case, y=probability, fill=case)) + 
        geom_bar(stat="identity") +
        geom_text(aes(label=final_probabilities), vjust=-0.3, size=3.5) +
        scale_fill_manual(values=c("red", "purple", "blue", "#f075ae", "gold")) +
        theme(legend.position = "none")

  return(bp)
}


plot_k_range_dimensions <- function(k_range, exported_dims, exported_stdevs) {
  df <- data.frame(k=k_range, dims=exported_dims, dims_lower=exported_dims-exported_stdevs, dims_upper=exported_dims+exported_stdevs)

  p <-  ggplot(df, aes(x=k, y=dims_upper.4)) + 
        geom_line(aes(y=dims.1, color="1"), lwd=1) + 
        geom_ribbon(aes(ymin=dims_lower.1, ymax=dims_upper.1), fill="red", alpha=0.5) +
        
        geom_line(aes(y=dims.2, color="2"), lwd=1) + 
        geom_ribbon(aes(ymin=dims_lower.2, ymax=dims_upper.2), fill="blue", alpha=0.5) +
        
        geom_line(aes(y=dims.3, color="3"), lwd=1) + 
        geom_ribbon(aes(ymin=dims_lower.3, ymax=dims_upper.3), fill="black", alpha=0.5) +
        
        geom_line(aes(y=dims.4, color="4"), lwd=1) + 
        geom_ribbon(aes(ymin=dims_lower.4, ymax=dims_upper.4), fill="yellow", alpha=0.5) +
        
        scale_colour_manual(values=c('1'='red', '2'='blue', '3'='black', '4'='yellow'), labels = c("X", "Y", "J", "Z")) +
        
        ylab("Dimension")
  
  return(p)
}