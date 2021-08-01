library(tidyverse)
library(here)
library(lubridate)
library(tidymodels)
library(janitor)
library(corrplot)
library(doParallel)
library(parallel)
library(finetune)


# load data ---------------------------------------------------------------

test_raw <- read_csv(here("test.csv"))

train_raw <- read_csv(here("train.csv"))

# 
dim(train_raw)
str(train_raw)



# EDA ---------------------------------------------------------------------

# plot histograms of numerics
train_raw %>%
  select(where(is.numeric)) %>%
  pivot_longer(everything(), names_to = "name", values_to = 'value') %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 40) +
  facet_wrap(~name, scales = "free") +
  theme_minimal()

# plot timeseries
train_raw %>%
  pivot_longer(cols = -date_time) %>%
  ggplot(aes(date_time, value)) +
  geom_point(alpha = 0.5) + 
  geom_smooth() + 
  facet_wrap(~name, scales = "free") +
  theme_minimal()

# heatmap of correlations
train_raw %>%
  select(where(is.numeric)) %>%
  cor() %>%
  heatmap()

train_raw %>%
  select(where(is.numeric)) %>%
  cor() %>%
  corrplot()


# we can see that DV's are right skewed and might need transformation. 
# also there are non-linear relationships

# create weekend, working hours, weekdays, saturday indicators
train_df <- train_raw %>%
  mutate(date = ymd_hms(date_time),
         hour = hour(date),
         weekdays = factor(wday(date), label = TRUE),
         working_hours = ifelse(hour %in% c(8:21), 1, 0),
         is_weekend = ifelse(wday(date) >= 5, 1, 0),
         satday = ifelse(wday(date) == 6, 1, 0),
         log_benz = log(target_benzene),
         log_co = log(target_carbon_monoxide),
         log_no = log(target_nitrogen_oxides),
         working_hours = factor(ifelse(working_hours == 1, "yes", "no")),
         is_weekend = factor(ifelse(is_weekend == 1, "yes", "no")),
         satday = factor(ifelse(satday == 1, "yes", "no"))
  ) %>%
  select(-hour, -date_time) #%>%
as_tsibble(index = date)

glimpse(train_df)




# make sure all variables are indentical. 
test_df <- test_raw %>%
  mutate(date = ymd_hms(date_time),
         hour = hour(date),
         weekdays = factor(wday(date), label = TRUE),
         working_hours = ifelse(hour %in% c(8:21), 1, 0),
         is_weekend = ifelse(wday(date) >= 5, 1, 0),
         satday = ifelse(wday(date) == 6, 1, 0),
         working_hours = factor(ifelse(working_hours == 1, "yes", "no")),
         is_weekend = factor(ifelse(is_weekend == 1, "yes", "no")),
         satday = factor(ifelse(satday == 1, "yes", "no"))
  ) %>%
  select(-hour, -date_time) #%>%
as_tsibble(index = date)



# check distributions now
train_df %>%
  as_tibble() %>%
  select(starts_with("log")) %>%
  pivot_longer(everything(), names_to = "name", values_to = 'value') %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 40) +
  facet_wrap(~name, scales = "free") +
  theme_minimal()


train_df %>%
  as_tibble() %>%
  select(starts_with("log"), date) %>%
  pivot_longer(cols = -date) %>%
  ggplot(aes(date, value)) +
  geom_point(alpha = 0.5) + 
  geom_smooth() + 
  facet_wrap(~name, scales = "free") +
  theme_minimal()



# XGBOOST for Benzene Model -----------------------------------------------

set.seed(2021)

# create recipe 
rec <- recipe(
  log_benz ~ deg_C + relative_humidity + absolute_humidity + is_weekend +
    sensor_1 + sensor_2 + sensor_3 + sensor_4 + sensor_5 + 
    satday + working_hours + weekdays, data = train_df,
) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors())

# folds 
folds <- train_df %>%
  vfold_cv(v = 10, strata = log_benz)

# view data after processing
rec %>%
  prep() %>% 
  bake(train_df) %>% 
  View()



# set metric
mset <- metric_set(rmse)

# creat model specs
xgb_spec <- 
  parsnip::boost_tree(
    trees = 1000,
    tree_depth = tune(),
    #mtry = finalize(mtry, train_df),
    loss_reduction = tune(),
    learn_rate = tune()
  ) %>%
  set_mode(mode = "regression") %>%
  set_engine("xgboost")



# workflow
xgb_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(xgb_spec)


xgb_grid

# parameters for bayesian tuning
xgb_set <- dials::parameters(xgb_spec)


# create parallel processing
cores <- parallel::detectCores(logical = FALSE)
cores

cl <- makePSOCKcluster(cores - 1)
registerDoParallel(cl)


# use finetune to select best model
tune_xgb <- tune_race_anova(
  xgb_wf,
  folds,
  grid = 10,
  metrics = metric_set(rmse),
  control = control_race(verbose_elim = TRUE)
)




# plot models 
plot_race(tune_xgb)

# metrics
tune_xgb %>%
  collect_metrics() %>%
  arrange(mean)

# select best model
xgb_best <- xgb_wf %>%
  finalize_workflow(select_best(tune_xgb, metric = "rmse"))


# fit final training set
xgb_best_fit <- xgb_best %>%
  fit(train_df)


# plot variable importance
xgb_best_fit %>%
  extract_fit_parsnip() %>%
  vip::vip(geom = "col")



xg_1 <- xgb_best_fit %>%
  augment(test_df) %>%
  mutate(.pred = exp(.pred)) %>%
  select(date_time = date, target_benzene = .pred)






# XGBoost for carbon monoxide ---------------------------------------------

rec2 <- recipe(
  log_co ~ deg_C + relative_humidity + absolute_humidity + is_weekend +
    sensor_1 + sensor_2 + sensor_3 + sensor_4 + sensor_5 + 
    satday + working_hours + weekdays, data = train_df,
) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors())





xgb_wf2 <- workflow() %>%
  add_recipe(rec2) %>%
  add_model(xgb_spec)


tune_xgb2 <- tune_race_anova(
  xgb_wf2,
  folds,
  grid = 10,
  metrics = metric_set(rmse),
  control = control_race(verbose_elim = TRUE),
)


# plot models
plot_race(tune_xgb2)

#metrics
tune_xgb2 %>%
  collect_metrics() %>%
  arrange(mean)

# select best model
xgb_best2 <- xgb_wf2 %>%
  finalize_workflow(select_best(tune_xgb2))


# fit final training set
xgb_best_fit2 <- xgb_best2 %>%
  fit(train_df)


# plot variable importances
xgb_best_fit2 %>%
  extract_fit_parsnip() %>%
  vip::vip(geom = "col")


# write to df
xg_2 <- xgb_best_fit2 %>%
  augment(test_df) %>%
  mutate(.pred = exp(.pred)) %>%
  select(date_time = date, target_carbon_monoxide = .pred)



# XGBoost for nitrogen oxides ---------------------------------------------

# create recipe 
rec3 <- recipe(
  log_no ~ deg_C + relative_humidity + absolute_humidity + is_weekend +
    sensor_1 + sensor_2 + sensor_3 + sensor_4 + sensor_5 + 
    satday + working_hours + weekdays, data = train_df,
) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors())


# create workflow
xg_wf3 <- workflow() %>%
  add_recipe(rec3) %>%
  add_model(xgb_spec)


# tune model
tune_xgb3 <- tune_race_anova(
  xg_wf3,
  folds,
  grid = 10,
  metric = metric_set(rmse),
  control = control_race(verbose_elim = TRUE)
)


# plot metrics
plot_race(tune_xgb3)


# show metrics
tune_xgb3 %>%
  collect_metrics() %>%
  arrange(mean)

# select best model
xgb_best3 <- xg_wf3 %>%
  finalize_workflow(select_best(tune_xgb3, metric = "rmse"))


# fit final training set
xgb_best_fit3 <- xgb_best3 %>%
  fit(train_df)

#plot variable importances
xgb_best_fit3 %>%
  extract_fit_parsnip() %>%
  vip::vip(geom = "col")



#save to df
xg_3 <- xgb_best_fit3 %>%
  augment(test_df) %>%
  mutate(.pred = exp(.pred)) %>%
  select(date_time = date, target_nitrogen_oxides = .pred)


# submission
xg_1 %>%
  bind_cols(xg_2 %>% select(-date_time)) %>%
  bind_cols(xg_3 %>% select(-date_time)) %>%
  mutate(date_time = lubridate::ymd_hms(date_time) %>% 
           as.character()) %>%
  select(date_time, target_carbon_monoxide, target_benzene, target_nitrogen_oxides) %>%
  write_csv(here("sub.csv"))




















