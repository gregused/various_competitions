library(tidyverse)
library(tidymodels)
library(here)
library(parallel)
library(doParallel)
library(vip)
library(tictoc)

# import data -------------------------------------------------------------

train_raw <- read_csv(here("tabular-playground-series-oct-2021/train.csv"))
dim(train_raw)

glimpse(train_raw)


# Do minor EDA ------------------------------------------------------------


train_df <- train_raw %>%
  mutate(target = factor(target)) %>%
  select(-id)

glimpse(train_df)

# viz first 60 numeric variables and their distributions
train_df %>%
  select(where(is.numeric)) %>%
  select(1:60) %>%
  pivot_longer(everything()) %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 40) +
  facet_wrap(~name, scales = "free")

colSums(is.na(train_df))

# xgboost :) ---------------------------------------------------------------

# set model formula and preprocessing steps.
rec <- recipe(target ~ ., data = train_df)

# set model parameters
xgb_specs <- boost_tree(mode = "classification",
                        trees = 10100,
                        sample_size = .65,
                        tree_depth = 6,
                        learn_rate = 0.01,
                        mtry = 10) %>%
  set_engine("xgboost")



# do this when tuning
cores <- parallel::detectCores(logical = FALSE)
cores

cl <- makePSOCKcluster(cores - 1)
registerDoParallel(cl)


xgb_wf <- workflow(rec, xgb_specs)


folds <- train_df %>%
  vfold_cv(3)


# makes my memory go BOOM! (Run at your own risk $$$)
tic()
xgb_tuned <- xgb_wf %>%
  tune_grid(folds, 
            grid = crossing(trees = c(10000, 15000),
                            tree_depth = c(9, 18),
                            learn_rate = c(0.01, 0.04),
                            mtry = c(12, 19)),
            metrics = metric_set(roc_auc),
            control = control_grid(verbose = FALSE))
toc()

stopCluster(cl)



# fit the model about 5:42 hours
tic()
xgb_fit <- xgb_wf %>% 
  fit(train_df)
toc()

# remove to free memory space
rm(train_raw, train_df)
gc()

# feature importance  ((this tells me i can mb drop variables?????))
xgb_fit %>%
  extract_fit_engine() %>%
  vip(geom = "col", num_features = 30) +
  theme_minimal()


xgb_fit %>%
  extract_fit_engine() %>%
  xgboost::xgb.importance(model = .) %>%
  xgboost::xgb.plot.importance(main = "XGBoost Feature Importance", top_n = 30)

test_raw <- read_csv(here("tabular-playground-series-oct-2021/test.csv"))


xgb_fit %>%
  augment(test_raw, type = "preds") %>%
  select(id, target = .pred_1) %>%
  write_csv(here('tabular-playground-series-oct-2021/xgb_sub2_oct.csv'))

