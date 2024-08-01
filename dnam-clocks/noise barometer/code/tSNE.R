library("tidyverse")
library("Rtsne")
library("readxl") 
library("dplyr") 
library(reticulate) 
require("reticulate") 
py_install("pandas")
source_python("/Users/xiaoyuemei/Desktop/lab_file/pickle_file.py") 
DNAme <- read_pickle_file("/Users/xiaoyuemei/scaled_hannum.pkl")
Ages <- read_excel("/Users/xiaoyuemei/Desktop/SynologyDrive/MXY/Summer21/Conboy Lab/DS3_ages_vertical.xlsx")
DNAme <- DNAme %>%
  mutate(ID = row_number())

DNAme <-
  t(DNAme)
Ages <- Ages %>% 
  mutate(ID = row_number())
new_DNAme <- merge(DNAme, Ages, by="ID") 
tSNE_fit <- DNAme_Pre %>% 
  select(where(is.numeric)) %>% 
  column_to_rownames("ID") %>%
  scale() %>% 
  Rtsne(perplexity = 40) 
tSNE_df <- tSNE_fit$Y %>% 
  as.data.frame() %>% 
  rename(tSNE1="V1",tSNE2="V2") %>%
  mutate(ID=row_number()) 
tSNE_df <- tSNE_df %>% 
  inner_join(DNAme_Pre, by="ID") 
tSNE_df %>%
  head()
tSNE_df %>% 
  ggplot(aes(x = tSNE1,y = tSNE2, color = ID))+ geom_point()+ theme(legend.position="bottom") 
ggsave("DNAmeVSAge.png")
