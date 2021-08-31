library(tidymodels)
library(tidyverse)
library(here)
library(doParallel)
library(parallel)


# import data -------------------------------------------------------------

train_raw <- read_csv(here("august", "train.csv", "train.csv")) %>%
  select(-id)

test_raw <- read_csv(here("august", "test.csv", "test.csv"))

train_raw %>%
  glimpse()

train_raw %>%
  ggplot(aes(loss)) +
  geom_histogram()



# begin modelings ---------------------------------------------------------

# recipe 
rec <- recipe(loss ~ ., data = train_raw) #%>%
  step_normalize(all_predictors())

# model specs 
xgb_spec <- boost_tree(
  mode = "regression",
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  mtry = tune()
) %>%
  set_engine("xgboost")

xgb_wf <- workflow(rec, xgb_spec)


folds <- train_raw %>%
  vfold_cv(strata = loss, v = 3)

# create parallel processing (not the best, but i don't have Mac :( )
cores <- parallel::detectCores(logical = FALSE)
cores

cl <- makePSOCKcluster(cores - 1)
registerDoParallel(cl)



# Tune/Train model --------------------------------------------------------

# train model
xgb_tune <- tune_grid(
  xgb_wf,
  folds,
  grid = crossing(
    trees = c(1750, 1850),
    tree_depth = c(9, 10),
    learn_rate = c(0.08, 0.01),
    mtry = c(18)
  ),
  metrics = metric_set(rmse),
  control = control_grid(verbose = FALSE)
)

# plot params
autoplot(xgb_tune)

# list all models
xgb_tune %>%
  collect_metrics() %>%
  arrange(mean)

# show best model
select_best(xgb_tune)



# fit last model on training data
xgb_best <- xgb_wf %>%
  finalize_workflow(select_best(xgb_tune)) %>%
  fit(train_raw)

xgb_best

# plot feature importance
library(vip)

xgb_best %>%
  extract_fit_engine() %>%
  vip(geom = "col", num_features = 20) +
  theme_minimal()


# save predictions to file
xgb_best %>%
  augment(test_raw) %>%
  select(id, loss = .pred) %>%
  write_csv("sub_aug_final.csv")


# Fit last model ----------------------------------------------------------


# edit for final params
xgb_spec <- boost_tree(
  mode = "regression",
  trees = 1900,
  tree_depth = 10,
  learn_rate = 0.01,
  mtry = 18
) %>%
  set_engine("xgboost")

xgb_wf <- workflow(rec, xgb_spec)


fit_train <- xgb_wf %>%
  fit(train_raw)


fit_train %>%
  augment(test_raw) %>%
  select(id, loss = .pred) %>%
  write_csv("sub_aug_final4.csv")


