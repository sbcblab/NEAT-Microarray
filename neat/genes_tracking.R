#!/usr/bin/env Rscript
#Bruno Iochins Grisci
#June 2018

library(gplots);

args = commandArgs(trailingOnly=TRUE); #should receive as arguments inputfile.gct and inputfile.cls

# Read values from tab-delimited autos.dat
genes_data <- read.table(args[1], header=T, sep=",") 

# Expand right side of clipping rect to make room for the legend
par(xpd=T, mar=par()$mar+c(0,0,0,4))

# Graph autos (transposing the matrix) using heat colors,  
# put 10% of the space between each bar, and make labels  
# smaller with horizontal y-axis labels
barplot(t(genes_data), main="Genes", ylab="Count", 
   col=heat.colors(4), space=0.1, cex.axis=0.8, las=1, cex=0.8) 
   
# Place the legend at (6,30) using heat colors
legend(6, 30, names(genes_data), cex=0.8, fill=heat.colors(4));
   
# Restore default clipping rect
par(mar=c(5, 4, 4, 2) + 0.1)
