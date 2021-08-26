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

xgb_tune <- tune_grid(xgb_wf,
                      folds,
                      grid = crossing(
                        trees = seq(900, 1800, 100),
                        learn_rate = c(0.001, 0.01, 0.02),
                        mtry = seq(60, 100, 10)
                      ),
                      metrics = metric_set(rmse),
                      control = control_grid(verbose = TRUE))


xgb_tune %>%
  collect_metrics() %>%
  arrange(mean)


xgb_best <- xgb_wf %>%
  finalize_workflow(select_best(xgb_tune)) %>%
  fit(train)


xgb_best %>%
  augment(test_raw) %>% select(.pred)
  mutate(loss = exp(.pred)) %>%
  select(id, loss) %>% tail()
  write_csv("sub.csv")

