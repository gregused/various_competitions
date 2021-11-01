library(tidymodels)
library(tidyverse)
library(h2o)
library(here)

# load train df
train_raw <- read_csv(here("tabular-playground-series-nov-2021/train.csv"))
glimpse(train_raw)

# modify data
train_df <- train_raw %>%
  select(-id) %>%
  mutate(target = case_when(target == 1 ~ "yes",
                            TRUE ~ "no") %>% as.factor())

glimpse(train_df)

remove(train_raw)
gc()

# load test df
test_raw <- read_csv(here("tabular-playground-series-nov-2021/test.csv"))

# no NAs
#map(train_df, ~sum(is.na(.))) %>% unlist

# basic recipe or no recipe lol
my_rec <- recipe(target ~ ., data = train_df)

# prepare recipe
baked_df <- my_rec %>%
  prep() %>%
  bake(train_df)

# convert to h2o 
h2o.init()

train_h2o_tbl <- as.h2o(baked_df)

# select variables X Y
y <- "target"
x <- setdiff(names(train_h2o_tbl), y)

# train autoML
auto_mod <- h2o.automl(
  y = y,
  x = x,
  training_frame = train_h2o_tbl,
  project_name = "kaggle_november_series",
  max_runtime_secs = 3600,
  seed = 0212
)


auto_mod@leaderboard

# prep test dataset
test_h2o <- my_rec %>%
  prep() %>%
  bake(test_raw) %>%
  as.h2o()

# select best model
top_model <- auto_mod@leader

# select top model and attach id 
top_model_basic_preds <- h2o.predict(top_model, newdata = test_h2o) %>%
  as_tibble() %>%
  bind_cols(test_raw) %>% 
  select(id, target = yes) 

# write sub to file
top_model_basic_preds %>%
  mutate(target = round(target, 4)) %>%
  write_csv(here("tabular-playground-series-nov-2021/h2o_preds.csv"))

