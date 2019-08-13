#!/usr/bin/env Rscript
#Bruno Iochins Grisci
#March 2018

library(gplots);

args = commandArgs(trailingOnly=TRUE); #should receive as arguments inputfile.gct and inputfile.cls

class <- as.double(read.table(args[2], header=FALSE, skip=2)); #gets the third line from the .cls file with the class markers for each sample in the .gct file
while (!is.element(0, class)) {class <- class - 1}; #if the first class marker is larger than 0, subtracts 1 from all markers so that the first class is always marked as 0
class_names <- strsplit(readLines(args[2])[2], " ")[[1]]; #gets the second line from the .cls file that has the ordered name of each class
class_names <- class_names[2:length(class_names)]; #removes first element because it should be a comment marker '#'

data  <- read.table(args[1], header=TRUE, skip=2); #reads the data from the .gct file
temp <- as.double(t(data)[3:nrow(t(data)),]);
d <- matrix(temp,ncol(t(data)),byrow=TRUE);
rownames(d) <- as.character(t(data)[1,]);
colnames(d) <- as.character(rownames(t(data))[3:nrow(t(data))]);

dir <- args[1]
dir <- gsub(basename(dir),'', dir)
setwd(dir);

#here we define the colors for the top bar in the heatmap identifying the class of each sample
legend_colors = c("#FF0000","#0000FF","#FFFF00","#008000","#000000","#FFC0CB","#808080","#FFA500");
color.map <- function(class) { 
    if (class == 0) 
        legend_colors[1] 
    else ( 
        if (class == 1)
            legend_colors[2]
        else (
            if (class == 2)
                legend_colors[3]
            else (
                if (class == 3)
                    legend_colors[4]
                 else (
                    if (class == 4) 
                        legend_colors[5]
                    else (
                        if (class == 5)  
                            legend_colors[6]
                        else (
                            if (class == 6)
                                legend_colors[7]
                            else (
                                legend_colors[8] ) ) ) ) ) ) ) };                     
classcolors <- unlist(lapply(class, color.map));
main_title = tools::file_path_sans_ext(basename(args[1])); #the heatmap title is the name of the .gct file

#draw heatmap sorted and with dendrograms
heatmap.2(d, distfun=function(x) as.dist((1-cor(t(x)))), col=redgreen(75), scale="row", trace="none", ColSideColors=classcolors, density.info="none", cexRow=0.6, margins=c(8,12), main=main_title);
legend("topright", 
    legend = class_names, 
    fill   = legend_colors[1:length(class_names)], border=FALSE, bty="n", y.intersp = 0.7, cex=0.7);
    
#draw heatmap with the samples sorted as in the data file
heatmap.2(d, distfun=function(x) as.dist((1-cor(t(x)))), Colv=FALSE, dendrogram='row', col=redgreen(75), scale="row", trace="none", ColSideColors=classcolors, density.info="none", cexRow=0.6, margins=c(8,12), main=main_title);
legend("topright", 
    legend = class_names, 
    fill   = legend_colors[1:length(class_names)], border=FALSE, bty="n", y.intersp = 0.7, cex=0.7)
