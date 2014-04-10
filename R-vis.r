library(ggplot2)

#mydata=read.csv("/home/rschadmin/Data/CCB/svr_stats/DataFrame.csv")

#PEARSON

p1 <- ggplot(mydata,aes(x=ROI,y=(accuracy_2.pearson+accuracy_1.pearson)*0.5)) +
      geom_point(aes(colour=subid)) +
      theme(legend.position="none") +
      ylim(0,1) +
      labs(title="Accuracy for Each ROI", y="accuracy (r value)")
ggsave(filename="/home/rschadmin/Pictures/stats/ROI_accR.png",plot=p1)

p2 <- ggplot(mydata,aes(x=ROI,y=(accuracy_2.pearson+accuracy_1.pearson)*0.5)) +
      geom_boxplot() +
      ylim(0,1) +
      labs(title="Accuracy for Each ROI", y="accuracy (r value)")
ggsave(filename="/home/rschadmin/Pictures/stats/ROI_accR_box.png",plot=p2)

p3 <- ggplot(mydata,aes(x=ROI,y=reproducibility.pearson)) +
      geom_point(aes(colour=subid)) +
      theme(legend.position="none") +
      ylim(0,1) +
      labs(title="Reproducibility for Each ROI", y="reproducibility (r value)")
ggsave(filename="/home/rschadmin/Pictures/stats/ROI_repR.png",plot=p3)

p4 <- ggplot(mydata,aes(x=ROI,y=reproducibility.pearson)) +
      geom_boxplot() +
      ylim(0,1) +
      labs(title="Reproducibility for Each ROI", y="reproducibility (r value)")
ggsave(filename="/home/rschadmin/Pictures/stats/ROI_repR_box.png",plot=p4)

p5 <- ggplot(mydata,aes(x=reproducibility.pearson,y=(accuracy_2.pearson+accuracy_1.pearson)*0.5)) +
      geom_point(aes(colour=subid)) +
      theme(legend.position="none") +
      labs(title="Accuracy v. Reproducibility", x="reproducibility(r value)", y="averaged accuracy (r value)")
ggsave(filename="/home/rschadmin/Pictures/stats/repR_accR.png",plot=p5)

#CONCORDANCE

p6 <- ggplot(mydata,aes(x=ROI,y=(accuracy_2.concordance+accuracy_1.concordance)*0.5)) +
      geom_point(aes(colour=subid)) +
      theme(legend.position="none") +
      ylim(0,1) +
      labs(title="Accuracy for each ROI", y="accuracy (correlation value)")
ggsave(filename="/home/rschadmin/Pictures/stats/ROI_accC.png",plot=p6)

p7 <- ggplot(mydata,aes(x=ROI,y=(accuracy_2.concordance+accuracy_1.concordance)*0.5)) +
      geom_boxplot() +
      ylim(0,1) +
      labs(title="Accuracy for each ROI", y="accuracy (correlation value)")
ggsave(filename="/home/rschadmin/Pictures/stats/ROI_accC_box.png",plot=p7)

p8 <- ggplot(mydata,aes(x=ROI,y=reproducibility.concordance)) +
      geom_point(aes(colour=subid)) + 
      theme(legend.position="none") +
      ylim(0,1) +
      labs(title="Reproducibility for each ROI", y="reproducibility (correlation value)")
ggsave(filename="/home/rschadmin/Pictures/stats/ROI_repC.png",plot=p8)

p9 <- ggplot(mydata,aes(x=ROI,y=reproducibility.concordance)) +
      geom_boxplot() +
      ylim(0,1) +
      labs(title="Reproducibility for each ROI", y="reproducibility (correlation value)")
ggsave(filename="/home/rschadmin/Pictures/stats/ROI_repC_box.png",plot=p9)

p10 <-ggplot(mydata,aes(x=reproducibility.concordance,y=(accuracy_2.concordance+accuracy_1.concordance)*.05)) +
      geom_point(aes(colour=subid)) +
      theme(legend.position="none") +
      labs(title="Accuracy v. Reproducibility", x="reproducibility(correlation value)", y="averaged accuracy (correlation value)")
ggsave(filename="/home/rschadmin/Pictures/stats/repC_accC.png",plot=p10)

t <- "Report Pages for Replicating:
        Prediction Intrinsic Brain Activity (Craddock et al. 2013)"

pdf(file = "/home/rschadmin/Pictures/stats/all_stats.pdf", paper="letter", bg = "white")
mp <- multiplot(p1, p2, p6, p7, cols=2)
mp2 <- multiplot(p3, p4, p8, p9, cols=2)
mp3 <- multiplot(p5,p10, cols=1)
dev.off()

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  require(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
