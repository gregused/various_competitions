library(tidyverse)
library(here)
library(tidymodels)
library(parallel)
library(doParallel)
library(tictoc)
library(finetune)

train <- read_csv(here('january_playground/train.csv'))
glimpse(train)



df <- train %>%
  mutate(month = lubridate::month(date),
         weekday = lubridate::wday(date),
         weekday = case_when(weekday == 1 ~ "mon",
                             weekday == 2 ~ "tues",
                             weekday == 3 ~ "wed",
                             weekday == 4 ~ "thu",
                             weekday == 5 ~ "fri",
                             weekday == 6 ~ "sat",
                             weekday == 7 ~ "sun")) %>%
  select(-c(row_id, date))

df %>%
  count(country) %>%
  ggplot(aes(country, n)) +
  geom_col()

rec <- recipe(num_sold ~ ., data = df) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

xgb_spec <- boost_tree(
  mode = "regression",
  mtry = tune(),
  trees = tune(),
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune()
) %>%
  set_engine("xgboost")

xgb_wf <- workflow(rec, xgb_spec)

folds <- vfold_cv(df, v = 5)



# grid 
xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  trees(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), df),
  learn_rate(),
  size = 30
)

xg_tuned_anova <- 
  tune_race_anova(
    xgb_wf,
    folds,
    grid = xgb_grid,
    metrics = metric_set(rmse),
    control = control_race(verbose = TRUE)
  )



plot_race(xg_tuned_anova)

show_best(xg_tuned_anova)



xg_tuned_anova %>%
  select_best(metric = "rmse")

xgb_last <- xgb_wf %>%
  finalize_workflow(select_best(xg_tuned_anova, metric = "rmse"))


xgb_last <- xgb_last %>%
  fit(df)

xgb_last %>%
  extract_fit_engine() %>%
  vip::vip(geom = "col", num_features = 30) +
  theme_minimal()

xgb_last %>%
  augment(test_df) %>%
  select(row_id, num_sold = .pred) %>%
  write_csv(here("january_playground/sub3_latin_grid.csv"))



# regular grid
cores <- parallel::detectCores(logical = FALSE)

cl <- makePSOCKcluster(cores - 1)
registerDoParallel(cl)


tic()
xgb_tuned <- xgb_wf %>% 
  tune_grid(folds, 
            grid = 10,
            metrics = metric_set(rmse, mae),
            control = control_grid(verbose = FALSE))
toc()

stopCluster(cl)

xgb_tuned %>%
  select_best(metric = "rmse")

xgb_last <- xgb_wf %>%
  finalize_workflow(select_best(xgb_tuned, metric = "rmse"))


tic()
xgb_fit <- xgb_last %>%
  fit(df)
toc()

xgb_fit %>%
  extract_fit_engine() %>%
  vip::vip(geom = "col", num_features = 30) +
  theme_minimal()

test_df <- read_csv(here("january_playground/test.csv"))

test_df <- test_df %>%
  mutate(month = lubridate::month(date),
         weekday = lubridate::wday(date),
         weekday = case_when(weekday == 1 ~ "mon",
                             weekday == 2 ~ "tues",
                             weekday == 3 ~ "wed",
                             weekday == 4 ~ "thu",
                             weekday == 5 ~ "fri",
                             weekday == 6 ~ "sat",
                             weekday == 7 ~ "sun")) %>%
  select(-c(date))


xgb_fit %>%
  augment(test_df) %>%
  select(row_id, num_sold = .pred) %>%
  write_csv(here("january_playground/sub2.csv"))

