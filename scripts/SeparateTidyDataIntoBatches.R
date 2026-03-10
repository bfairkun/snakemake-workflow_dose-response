#!/usr/bin/env Rscript

######################################################################
# @author      : Your Name (your@email.com)
# @file        : SeparateTidyDoseCP3Experiment_Data_For_BayesianModelFitting.R
# @created     : 2025-07-03 12:53
#
# @description : Sepearate data into n separate files for batches of genes to pallelize fitting model gene by gene
######################################################################

# Use hard coded arguments in interactive R session, else use command line args
if(interactive()){
    args <- scan(text=
                 "ExperimentGeneralizedDoseResponseModelling/Expression/Data/Exp11_TidyDataForModelling.tsv.gz scratch/Batch_ 10", what='character')
} else{
    args <- commandArgs(trailingOnly=TRUE)
}

library(tidyverse)

f_in <- args[1]
f_out_prefix <- args[2]
n_batches <- as.integer(args[3])


dat <- read_tsv(f_in) %>%
    mutate(IsControl = dose==0) %>%
    arrange(featureID, desc(IsControl), treatment, dose)

# first let's get rid of the dummy redundant DMSO controls and also recode the 100nM samples (dummy recoded for DMSO) back to 0nM



batches <- dat %>%
  distinct(featureID) %>%
  mutate(batch = ntile(row_number(), n_batches)) %>%
  mutate(batch = batch - 1)  # make batches 0-indexed

dat %>%
  left_join(batches, by = "featureID") %>%
  group_by(batch) %>%
  dplyr::select(-batch, -IsControl) %>%
  group_walk(~ write_tsv(.x, paste0(f_out_prefix, .y$batch, ".tsv.gz")))
