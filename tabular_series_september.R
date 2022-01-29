library(tidyverse)
library(parsnip)
library(here)
library(rsample)
library(yardstick)
library(recipes)
library(workflows)
library(dials)
library(tune)
library(janitor)
library(remotes)
library(finetune)
library(parallel)
library(doParallel)

# for catboost/lightgbm for parsnip
#remotes::install_github("curso-r/treesnip", dependencies = FALSE) 
library(treesnip)
# baseline Xgboost --------------------------------------------------------


train_raw <- read_csv(here("tabular-playground-series-sep-2021/train.csv"))
train_raw

test_raw <- read_csv(here("tabular-playground-series-sep-2021/test.csv"))
  
  
colSums(is.na(test_raw))

skimr::skim(train_raw)
colSums(is.na(train_raw))

train_df <- train_raw %>%
  select(-id) %>%
  mutate(claim = factor(if_else(claim == 1, "yes", "no"))) %>%
  select(claim, everything())

glimpse(train_df)

names(train_df)

# pretty balanced classes
train_df %>%
  count(claim)


rec <- recipe(claim ~ ., data = train_df) %>%
  step_impute_median(all_numeric_predictors())

# needs a run
xgb_spec <- boost_tree(
  mode = "classification",
  trees = 1800,
  tree_depth = 9,
  learn_rate = 0.01,
  #mtry = 24
) %>%
  set_engine("xgboost")

# keeps crashing Rsession...
#lgb_spec <- boost_tree(mode = "classification",
#                      trees = 2000,
#                       learn_rate = 0.04,
#) %>%
#  set_engine("lightgbm")

xgb_wf <- workflow(rec, xgb_spec)

#lgb_wf <- workflow(rec, lgb_spec)

# create parallel processing (not the best, but i don't have Mac :( )
cores <- parallel::detectCores(logical = FALSE)
cores

cl <- makePSOCKcluster(cores - 1)
registerDoParallel(cl)

xgb_fit <- xgb_wf %>%
  fit(train_df)

xgb_fit

# session crashes
lgb_fit <- lgb_wf %>%
  fit(train_df)

# plot feature importance
library(vip)

xgb_fit %>%
  extract_fit_engine() %>%
  vip(geom = "col", num_features = 40) +
  theme_minimal()


xgb_fit %>%
  augment(test_raw, type = "preds") %>% 
  select(id, claim = .pred_yes) %>%
  write_csv(here('sub_sept_baseline4.csv'))


# save RDS

saveRDS(xgb_fit, 'xgb_mod.rds')

