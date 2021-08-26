library(tidymodels)
library(tidyverse)
library(here)

train_raw <- read_csv(here("train.csv")) %>%
  select(-id)

test_raw <- read_csv(here("test.csv"))

train_raw %>%
  glimpse()

train_raw %>%
  ggplot(aes(loss)) +
  geom_histogram()



rec <- recipe(loss ~ ., data = train_raw) %>%
  step_normalize(all_predictors())

xgb_spec <- boost_tree(
  mode = "regression",
  mtry = tune(),
  trees = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost")

xgb_wf <- workflow(rec, xgb_spec)


folds <- train_raw %>%
  vfold_cv(strata = loss, v = 3)

xgb_tune <- tune_grid(
  xgb_wf,
  folds,
  grid = crossing(
    trees = c(1200, 1600, 1800),
    tree_depth = c(4, 8, 15),
    learn_rate = c(0.0000749, 0.001, 0.0717),
    mtry = c(20, 28, 34)
  ),
  metrics = metric_set(rmse),
  control = control_grid(verbose = TRUE)
)


xgb_tune %>%
  collect_metrics() %>%
  arrange(mean)

select_best(xgb_tune)


xgb_best <- xgb_wf %>%
  finalize_workflow(select_best(xgb_tune)) %>%
  fit(train)


xgb_best %>%
  augment(test_raw) %>%
  select(id, loss = .pred) %>%
  write_csv("sub3.csv")




