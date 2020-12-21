library("protr")
library(tidyverse)
library("readr")

#path="data/DrugBankH/allfastas/"
#file="data/DrugBankH/DrugBankH_ID2FASTA_more30.tsv"
#listfiles = list.files(path)
#x = readFASTA(system.file(path+filename, package = "protr"))[[1]]
file="/Users/matthieu/ownCloud/Code/CFTR_PROJECT/data/drugbank_v5.1.5/S0h/preprocessed/S0h_uniprot2fasta.tsv"

# https://cran.r-project.org/web/packages/protr/vignettes/protr.html#10_summary

prots = read_tsv(file, col_names = FALSE)

features = lapply(1:dim(prots)[1],function(prot_i){
   
   print(prots$X1[prot_i])
   
   c(extractAAC(prots$X2[prot_i]), # Amino acid composition
      extractDC(prots$X2[prot_i]), # Diptetide composition
      extractMoreauBroto(prots$X2[prot_i], nlag = min(c(30,nchar(prots$X2[prot_i])-1))), # Normalized Moreau-Broto Autocorrelation Descriptor
      extractMoran(prots$X2[prot_i], nlag = min(c(30,nchar(prots$X2[prot_i])-1))), # Moran autocorrelation
      extractGeary(prots$X2[prot_i], nlag = min(c(30,nchar(prots$X2[prot_i])-1))), # Geary autocorrelaton
      # Composition, Transition and Distribution descriptors
      extractCTDC(prots$X2[prot_i]), # Composition
      extractCTDT(prots$X2[prot_i]), # Transition
      extractCTDD(prots$X2[prot_i]), # Distribution
      extractCTriad(prots$X2[prot_i]), # Conjoint Triad Descirptors
      extractSOCN(prots$X2[prot_i], nlag = min(c(30,nchar(prots$X2[prot_i])-1))), # Sequence-order-coupling_numbers
      extractQSO(prots$X2[prot_i], nlag = min(c(30,nchar(prots$X2[prot_i])-1))), # Quasi-Sequence-Order descriptors
      extractPAAC(prots$X2[prot_i], lambda = min(c(30,nchar(prots$X2[prot_i])-1))), # Pseudo-Amino Acid Composition
      extractAPAAC(prots$X2[prot_i], lambda = min(c(30,nchar(prots$X2[prot_i])-1)))) # Amphiphilic Pseudo-Amino Acid Composition (APAAC)
   } ) %>% do.call(rbind, .)
   
mutate(as_tibble(features), protein = prots$X1) %>%
    select(protein, everything()) %>%
   write_tsv('/Users/matthieu/ownCloud/Code/CFTR_PROJECT/data/drugbank_v5.1.5/S0h/features/S0h_prot_standardfeatures.tsv') 
   
   
   
   
#file="data/DrugBankH/DrugBankH_ID2FASTA_less30_2.tsv"
#prots = read_tsv(file, col_names = FALSE)
#features = lapply(prots$X2, function(x){
#    nlags = nchar(x) - 1
#    print(nlags)
#    c(extractAAC(x),
#           extractDC(x),
#           extractMoreauBroto(x),
#           extractMoran(x),
#           extractGeary(x),
#           extractCTDC(x),
#           extractCTDT(x),
#           extractCTDD(x),
#           extractCTriad(x),
#           extractSOCN(x, nlag=nlags),
#           extractQSO(x, nlag=nlags),
#           extractPAAC(x, lambda=nlags),
#           extractAPAAC(x, lambda=nlags))
#   } ) %>% do.call(rbind, .)
#   
#mutate(as.tibble(features), protein = prots$X1) %>%
#    select(protein, everything()) %>%
#   write_tsv('data/DrugBankH/DrugBanH_standardfeatures_less_2.tsv') 
#   

l = data.frame(lapply(extractAAC(prots$X2[1]), function(x) t(data.frame(x))))
data.frame(l)  %>% write_tsv('data/DrugBankH/DrugBanH_standardfeatures_test.tsv')

#l = data.frame(lapply(extractSOCN(x, nlag=nlags), function(x) t(data.frame(x))))
#data.frame(l)  %>% write_tsv('data/DrugBankH/DrugBanH_standardfeatures_test.tsv')

#l = data.frame(lapply(extractPAAC(x, lambda=nlags), function(x) t(data.frame(x))))
#data.frame(l)  %>% write_tsv('data/DrugBankH/DrugBanH_standardfeatures_test.tsv')
#   
