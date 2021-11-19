# load libs
library(tidyverse)
library(tidymodels)

theme_set(theme_minimal())

# data import
hotel_raw <- read_csv("../input/hotel-booking-demand/hotel_bookings.csv")

glimpse(hotel_raw)


# explore data
hotel_raw %>%
  ggplot(aes(is_canceled)) +
  geom_histogram()

hotel_raw %>%
  count(is_canceled)

hotel_raw %>%
  count(reservation_status)

hotel_raw %>%
  count(hotel)

hotel_raw %>%
  ggplot(aes(lead_time)) +
  geom_histogram()

hotel_raw %>%
  select(where(is.numeric)) %>%
  pivot_longer(everything()) %>%
  ggplot(aes(value)) +
  geom_histogram() +
  facet_wrap(~name, scales = "free")



hotel_raw %>%
  count(required_car_parking_spaces)

colSums(is.na(hotel_raw))

dim(hotel_raw)



# manipulate data
hotel_df <- hotel_raw %>%
  mutate(babies = case_when(babies > 2 ~ "plus 3 babies",
                            babies == 0 ~ "0 baby",
                            babies == 1 ~ "1 baby",
                            babies == 2 ~ "2 baby",
                            TRUE ~ as.character(babies)),
         adults = case_when(adults > 4 ~ "plus 4 adults",
                            adults == 3 ~ "3 adults",
                            adults == 4 ~ "4 adults",
                            adults == 2 ~ "2 adults",
                            adults == 1 ~ "1 adult",
                            adults == 0 ~ "0 adults",
                            TRUE ~ as.character(adults)),
         arrival_date_year = as.character(arrival_date_year),
         children = ifelse(children > 3, "more than 3", as.character(children)),
         children = ifelse(is.na(children), "0", children),
         is_canceled = as.character(if_else(is_canceled == 1, "yes", "no")),
         required_car_parking_spaces = case_when(required_car_parking_spaces > 1 ~ "plus 1 parking",
                                                 required_car_parking_spaces < 1 ~ "0 parking",
                                                 TRUE ~ "1 parking"),
         stays_in_weekend_nights = fct_lump(as.character(stays_in_weekend_nights), 8),
         stays_in_week_nights = if_else(stays_in_week_nights > 10, "more than 10", as.character(stays_in_week_nights)),
         previous_bookings_not_canceled = fct_lump(as.character(previous_bookings_not_canceled), 12),
         market_segment = case_when(market_segment == "Undefined" ~ "Online TA",
                                    TRUE ~ market_segment), 
         previous_cancellations = ifelse(previous_cancellations > 2, "more than two", "two and less"),
         booking_changes = ifelse(booking_changes > 5, "above 5", as.character(booking_changes)),
         days_in_waiting_list = ifelse(days_in_waiting_list > 5, "above 5", as.character(days_in_waiting_list)),
         country = fct_lump(country, 10),
         is_repeated_guest = ifelse(is_repeated_guest == 1, "yes", "no"),
         agent = fct_lump(agent, 10),
         company = fct_lump(company, 10),
         reserved_room_type = fct_lump(reserved_room_type, 5),
         total_of_special_requests = as.character(total_of_special_requests),
         assigned_room_type = fct_lump(assigned_room_type, 5)) %>%
  select(-reservation_status_date, -arrival_date_year, -arrival_date_month, -arrival_date_week_number,
         -meal, -country, -distribution_channel, -agent, -company, -required_car_parking_spaces, -reservation_status,
         -reservation_status_date)

glimpse(hotel_df)



# set up for modeling
init_split <- initial_split(hotel_df, prop = 0.75, strata = is_canceled)
init_split

train_df <- training(init_split)

test_df <- testing(init_split)

folds <- train_df %>%
  vfold_cv(3)

# XGB :) 
xgb_spec <- boost_tree(mode = "classification",
                       trees = tune(),
                       learn_rate = tune(),
                       mtry = tune()
) %>%
  set_engine("xgboost")

# recipe
rec <- recipe(is_canceled ~ ., data = train_df) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)


# workflow
wf_xgb <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(rec)


## TUNE MODEL
tune_xgb <- tune_grid(wf_xgb,
                      folds,
                      grid = 5,
                      metrics = metric_set(mn_log_loss),
                      control = control_grid(verbose = TRUE))

# plot parameter results
tune_xgb %>%
  autoplot()

# best model
tune_xgb %>%
  select_best()

# fit model to training set
best_mod_fit <- wf_xgb %>%
  finalize_workflow(select_best(tune_xgb)) %>%
  fit(train_df)

# show predictions on testing set
best_mod_fit %>%
  augment(test_df %>% select(-is_canceled), type.predict = "prob") %>% head()


# plot ROC curve 
best_mod_fit %>% 
  predict(new_data = test_df, type = "prob") %>%
  bind_cols(test_df %>% select(is_canceled)) %>% 
  roc_curve(., factor(is_canceled, levels = c("yes", "no")), .pred_yes) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_abline(linetype = 2, slope = 1, intercept = 0) +
  geom_line(size = 1.2, color = "midnightblue")




