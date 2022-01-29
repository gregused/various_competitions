library(tabnet)
library(tidyverse)
library(tidymodels)
library(here)
library(doParallel)
library(parallel)

# data import
train_raw <- read_csv(here("tabular-playground-series-sep-2021/train.csv"))
train_raw


test_raw <- read_csv(here("tabular-playground-series-sep-2021/test.csv"))


train_df <- train_raw %>%
  select(-id) %>%
  mutate(claim = factor(if_else(claim == 1, "yes", "no"))) %>%
  select(claim, everything()) %>%
  mutate(claim = fct_relevel(claim, "yes"))


rec <- recipe(claim ~ ., data = train_df) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())


# makes my CPU run 100%, kinda scary...
fit <- tabnet_fit(rec, train_df, epochs = 30)


fit %>%
  augment(test_raw, type = "preds") %>% 
  select(id, claim = .pred_yes) %>%
  write_csv(here('sub_sept_tabnet_base.csv'))

