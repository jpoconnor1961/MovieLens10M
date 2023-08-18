# Determination of Regularized Biases in MovieLens Model with edx_train and edx_test Data Sets ----

# Order of model components developed below are Overall Mean, Regularized Movie Bias, Regularized Temporal Movie Bias, 
# Regularized User Bias, and Regularized Genre Bias.  

library(tidyverse)
library(data.table)
library(lubridate)

# RMSE LOSS FUNCTION ----
# Loss Function that computes the RMSE for ratings and their corresponding predictors:
RMSE <- function(True_Ratings, Predicted_Ratings){
  sqrt(mean((True_Ratings - Predicted_Ratings)^2))
}

# edx_train OVERALL MEAN RATING ----
mu_hat_edx_train <- mean(edx_train$rating)

# CROSS VALIDATION OF REGULARIZATION PARAMETER LAMBDA FOR edx_train GENRE BIAS + USER BIAS + MOVIE BIAS MODEL----
# The regularization parameter Lambda will be calculated using the baseline rating biases of movies, users, and genres 
# in a 3-way simultaneous calculation of the Lambda parameter for all three biases.

# dplyr slice_sample() function is used to generate random bootstrap samples at 100% sample size of the full edx_train data set,
# and thus the full edx_train$rating is used to test predicted ratings.
# 
# The following 30-fold cross validation takes about 6 hours to run on a standard laptop.
# MOVIE USER GENRE (MUG) CROSS VALIDATED REGULARIZATION (CVR), a.k.a. FULL MODEL REGULARIZATION (FMR):

set.seed(1, sample.kind = "Rounding")
Bootstraps <- seq(1:30)
MinRMSE_Lambdas_GenreUserMovieBias <- sapply(Bootstraps, function(B) {
  edx_bootstrap <- slice_sample(edx_train, prop = 1.00, replace = TRUE)
  mu <- mean(edx_bootstrap$rating)
  Lambdas <- seq(0, 15, 0.20)
  rmses <- sapply(Lambdas, function(L){
    MovieBiasReg <- edx_bootstrap %>% 
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n() + L))
    UserBiasReg <- edx_bootstrap %>% 
      left_join(MovieBiasReg, by = "movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n() + L))
    GenreBiasReg <- edx_bootstrap %>% 
      left_join(MovieBiasReg, by = "movieId") %>%
      left_join(UserBiasReg, by = "userId") %>%
      group_by(genreId) %>%
      summarize(b_g = sum(rating - b_i - b_u - mu)/(n() + L))
    predicted_ratings <- edx_train %>% 
      left_join(MovieBiasReg, by = "movieId") %>%
      left_join(UserBiasReg, by = "userId") %>%
      left_join(GenreBiasReg, by = "genreId") %>%
      mutate(pred = mu + b_i + b_u + b_g) %>%
      .$pred
    predicted_ratings[is.na(predicted_ratings)] <- 0
    return(RMSE(edx_train$rating, predicted_ratings))
  })
  tibble(Min_RMSE = min(rmses), Lambda = Lambdas[which.min(rmses)])
})

t(MinRMSE_Lambdas_GenreUserMovieBias)

#      Min_RMSE  Lambda
# [1,] 0.8603006 4     
# [2,] 0.8603001 4     
# [3,] 0.8602919 4     
# [4,] 0.86028   4     
# [5,] 0.860276  4     
# [6,] 0.8603011 4     
# [7,] 0.8603055 4     
# [8,] 0.8602216 3.8   
# [9,] 0.8602262 4     
#[10,] 0.8603109 4     
#[11,] 0.8602422 4     
#[12,] 0.8603241 4     
#[13,] 0.8603309 4     
#[14,] 0.8602778 4     
#[15,] 0.860277  4     
#[16,] 0.8602977 4     
#[17,] 0.8603199 4     
#[18,] 0.8602588 4     
#[19,] 0.8602443 4     
#[20,] 0.8602833 4     
#[21,] 0.8603087 3.8   
#[22,] 0.8602829 4     
#[23,] 0.8602302 4     
#[24,] 0.8602576 4     
#[25,] 0.8603201 4     
#[26,] 0.8603    4     
#[27,] 0.8602995 3.8   
#[28,] 0.8603189 3.8   
#[29,] 0.8602922 4     
#[30,] 0.8603263 4 

MinRMSE_Lambdas_GenreUserMovieBias <- as.data.frame(t(MinRMSE_Lambdas_GenreUserMovieBias))

Lambda_FMR_GenreUserMovieBias_edx_train <- mean(as.numeric(as.vector(MinRMSE_Lambdas_GenreUserMovieBias$Lambda)))
# Lambda_FMR_GenreUserMovieBias_edx_train
# [1] 3.973333

# Full Model Regularized (FMR) Movie Bias Calculation ----
# edx_train FMR movie bias, b_i_fmr, is estimated by applying Lambda_FMR_GenreUserMovieBias_edx_train in the calculation 
# of the average of all the ratings of a movie after subtracting out mu_hat_edx_train from each movie rating:
movie_bias_FMR_edx_train <- edx_train %>% group_by(movieId) %>% 
                                          summarize(b_i_fmr = sum(rating - mu_hat_edx_train)/
                                                              (n() + Lambda_FMR_GenreUserMovieBias_edx_train))
# movie_bias_FMR_edx_train
# A tibble: 10,677 x 2
#   movieId b_i_fmr
#     <dbl>   <dbl>
# 1       1  0.415 
# 2       2 -0.306 
# 3       3 -0.368 
# 4       4 -0.646 
# 5       5 -0.448 
# 6       6  0.306 
# 7       7 -0.155 
# 8       8 -0.372 
# 9       9 -0.506 
#10      10 -0.0860
# ... with 10,667 more rows

# edx_train MOVIE TEMPORAL BIAS MODEL ---- 
# Refer to report section 2.3 for optimization of temporal parameter to the two-year time bins. 

# Assign each movie rating observation in edx_test to a two_year time bin:
edx_test_date <- edx_test %>% mutate(date = as_datetime(timestamp), bin_2_year = round_date(date, unit = "2 years")) %>% 
                              select(-timestamp, -title, -genres)

# Assign each movie rating observation in edx_train to a two_year time bin:
edx_train_date <- edx_train %>% mutate(date = as_datetime(timestamp), bin_2_year = round_date(date, unit = "2 years")) %>% 
                                select(-timestamp, -title, -genres)

# Calculation of Mean Slope of Squared Error Cost Function from Partial Derivative with respect to Lambda_t 
# for Prediction Model from Total Movie Bias.  (Also see Section 2.3 and Appendix B of the report for more background and context.)

# Determine the average rating of each movie in edx_train:
edx_train_movie_rating_avg <- edx_train %>% group_by(movieId) %>% summarise(movie_rating_avg = mean(rating))

# edx_train_movie_rating_avg
# A tibble: 10,677 x 2
#   movieId movie_rating_avg
#     <dbl>            <dbl>
# 1       1             3.93
# 2       2             3.21
# 3       3             3.14
# 4       4             2.86
# 5       5             3.06
# 6       6             3.82
# 7       7             3.36
# 8       8             3.14
# 9       9             3.01
#10      10             3.43
# ... with 10,667 more rows

# Partial Derivative of Squared Error of r_hat_i Prediction from Total Movie Bias Model with respect to Lambda_t:
#         dSEi/dLt = -2*(r_hat_i - r_i)*sum(r_it)/(n_it + L_t)^2               (From Equation B.1 in Appendix B)

                                                     # 3-Way Lambda_FMR_GenreUserMovieBias_edx_train = 3.973333
                                                     # Cross Validated/Full Model Regularization Parameter (CVR/FMR)  
Lambda_m <- Lambda_FMR_GenreUserMovieBias_edx_train  # Lambda_m applied for regularization of Static Movie Bias.
Lambdas_t <- seq(-0.50, 0.50, 0.01)  # Range and precision of Lambda_t tested to pinpoint value where mean slope is
                                     # no more than ±0.00001 from zero slope result of partial derivative of
                                     # squared error of cost function, for prediction model from total movie bias.
# mu_hat_edx_train <- 3.51242785050505                       # Global Mean Rating of All Movies in Training Dataset 

# Algorithm for calculation of Mean Slope of Squared Error of Total Movie Bias Prediction Model w.r.t. Lambda_t:
dSEidLt_train <- sapply(Lambdas_t, function(L_t){            # Lambda_t parameter is tuned on entire Training dataset
  reg_movie_bias <- edx_train_date %>%                       # that also includes bin_2_year date data.
    group_by(movieId) %>% 
    summarize(b_m_reg = sum(rating - mu_hat_edx_train)/(n() + Lambda_m))  # regularized static movie bias
  reg_2yr_avg_rating <- edx_train_date %>% 
    group_by(movieId, bin_2_year) %>% 
    summarise(reg_2yr_avg_rating = sum(rating)/(n() + L_t))               # regularized part of temporal movie bias
  r_hat_i <- edx_train_date %>% 
    left_join(reg_movie_bias, by = 'movieId') %>%
    left_join(reg_2yr_avg_rating, by = c('movieId', 'bin_2_year')) %>%
    left_join(edx_train_movie_rating_avg, by = 'movieId') %>%
    mutate(r_hat_i = mu_hat_edx_train + b_m_reg + reg_2yr_avg_rating - movie_rating_avg) %>% 
    select(userId, movieId, bin_2_year, r_hat_i)                          # total movie bias model prediction r_hat_i
  regSqrd_2yr_avg_rating <- edx_train_date %>% 
    group_by(movieId, bin_2_year) %>% 
    summarise(regSqrd_2yr_avg_rating = sum(rating)/(n() + L_t)^2)         # sum(r_it)/(n_it + L_t)^2 term of dSEi/dLt
  SEiLt_df <- edx_train_date %>%                                                      
    left_join(r_hat_i, by = c('userId', 'movieId', 'bin_2_year')) %>%
    left_join(regSqrd_2yr_avg_rating, by = c('movieId', 'bin_2_year')) %>%
    mutate(SEiLt = -2 * (r_hat_i - rating) * regSqrd_2yr_avg_rating)      # Individual Slope of Squared Error
  SEiLt_avg <- SEiLt_df %>% mutate(SEiLt_avg = sum(SEiLt)/n()) %>% pull(SEiLt_avg) %>% first()  
  return(SEiLt_avg)                                                       # Mean Slope of Squared Error
})

# Data frame of  Mean Slope of Squared Error results from range of Lambda_t values tested:
dSEidLt_train_df <- data.frame(Lambda_t = seq(-0.50, 0.50, 0.01), MeanSlope_SquareError = dSEidLt_train)

# Minimum of Absolute Value of MeanSlope_SquareError = Optimized Value of Lambda_t Regularization Parameter:
Lt <- dSEidLt_train_df$Lambda_t[which.min(abs(dSEidLt_train_df$MeanSlope_SquareError))]
Lt
#[1] 0.05

# PARTIAL DISPLAY OF DATA FROM dSEidLt_train_df DATAFRAME, FOR PLOT FIGURE 4B OF REPORT:
dSEidLt_train_df[51:60,]

#   Lambda_t MeanSlope_SquareError
#51     0.00         -7.746359e-04
#52     0.01         -6.056021e-04
#53     0.02         -4.425634e-04
#54     0.03         -2.852265e-04
#55     0.04         -1.333156e-04
#56     0.05          1.342904e-05 <- Minimum Slope rounds to
#57     0.06          1.552521e-04       No More Than 0.00001
#58     0.07          2.923840e-04            from Zero Slope
#59     0.08          4.250420e-04
#60     0.09          5.534312e-04

# determine the Lt = 0.05 regularized average rating of each movie for each 2 year time period that the movie has ratings in edx_train
edx_train_Lt.05reg_2yr_avg_rating <- edx_train_date %>% group_by(movieId, bin_2_year) %>% 
                                                        summarise(reg_2yr_rating = sum(rating)/(n() + Lt))

# determine the Lt 0.05 regularized time bias of each movie's average rating within each two year period that it was rated, 
# as the difference of its Lt.05reg_2yr_avg_rating from the movie's overall average rating across all time periods in edx_train:
movie_time_bias_Lt.05reg_edx_train <- edx_train_Lt.05reg_2yr_avg_rating %>% left_join(edx_train_movie_rating_avg, by = 'movieId') %>% 
                                                                            mutate(b_i_Lt = reg_2yr_rating - movie_rating_avg) %>% 
                                                                            select(movieId, bin_2_year, b_i_Lt)
# movie_time_bias_Lt.05reg_edx_train
# A tibble: 43,939 × 3
# Groups:   movieId [10,677]
#   movieId bin_2_year           b_i_Lt
#     <dbl> <dttm>                <dbl>
# 1       1 1996-01-01 00:00:00  0.198 
# 2       1 1998-01-01 00:00:00 -0.0476
# 3       1 2000-01-01 00:00:00  0.139 
# 4       1 2002-01-01 00:00:00  0.140 
# 5       1 2004-01-01 00:00:00 -0.0237
# 6       1 2006-01-01 00:00:00 -0.186 
# 7       1 2008-01-01 00:00:00 -0.235 
# 8       1 2010-01-01 00:00:00 -0.0643
# 9       2 1996-01-01 00:00:00  0.354 
#10       2 1998-01-01 00:00:00  0.135 
# ... with 43,929 more rows

# Full Model Regularized (FMR) User Bias Calculation ----
# edx_train FMR user bias, b_u_fmr, is estimated by applying Lambda_FMR_GenreUserMovieBias_edx_train in the calculation 
# of the average of all the movie ratings by a user after subtracting out mu_hat_edx_train, b_i_fmr, and b_i_Lt
# for each user and movie combination:
user_bias_FMR_edx_train <- edx_train_date %>% 
                           left_join(movie_bias_FMR_edx_train, by='movieId') %>% 
                           left_join(movie_time_bias_Lt.05reg_edx_train, by = c('movieId', 'bin_2_year')) %>% 
                           group_by(userId) %>% 
                           summarize(b_u_fmr = sum(rating - mu_hat_edx_train - b_i_fmr - b_i_Lt)/
                                               (n() + Lambda_FMR_GenreUserMovieBias_edx_train))
# user_bias_FMR_edx_train
# A tibble: 69,878 × 2
#   userId b_u_fmr
#    <int>   <dbl>
# 1      1  1.24  
# 2      2 -0.263 
# 3      3  0.334 
# 4      4  0.550 
# 5      5  0.170 
# 6      6  0.225 
# 7      7  0.0119
# 8      8  0.283 
# 9      9  0.213 
#10     10  0.0238
# ... with 69,868 more rows

## Full Model Regularized (FMR) Genre Bias Calculation ----
# edx_train FMR genre bias, b_g_fmr, is estimated by applying Lambda_FMR_GenreUserMovieBias_edx_train in the calculation 
# of the average of all the movie ratings in a genre after subtracting out mu_hat_edx_train, b_u_fmr, b_i_fmr and 
# b_i_Lt for each user and movie combination in a given genre:
genre_bias_FMR_edx_train <- edx_train_date %>% 
                            left_join(user_bias_FMR_edx_train, by='userId') %>% 
                            left_join(movie_bias_FMR_edx_train, by='movieId') %>% 
                            left_join(movie_time_bias_Lt.05reg_edx_train, by = c('movieId', 'bin_2_year')) %>%
                            group_by(genreId) %>% 
                            summarize(b_g_fmr = sum(rating - mu_hat_edx_train - b_u_fmr - b_i_fmr - b_i_Lt)/
                                                (n() + Lambda_FMR_GenreUserMovieBias_edx_train))
# genre_bias_FMR_edx_train
# A tibble: 797 × 2
#   genreId  b_g_fmr
#     <int>    <dbl>
# 1       1 -0.00673
# 2       2 -0.0107 
# 3       3 -0.0182 
# 4       4 -0.0237 
# 5       5 -0.0219 
# 6       6 -0.0346 
# 7       7 -0.0208 
# 8       8 -0.0296 
# 9       9 -0.0289 
#10      10 -0.00910
# ... with 787 more rows

# TEST FMR GENRE BIAS + FMR USER BIAS + FMR MOVIE BIAS + Lt 0.05 REGULARIZED MOVIE TEMPORAL BIAS MODEL ON THE EDX_TEST DATA SET ----
# FMR_GenreUserMovieTime_model <- edx_test_date %>% 
#                                 left_join(genre_bias_FMR_edx_train, by='genreId') %>% 
#                                 left_join(user_bias_FMR_edx_train, by='userId') %>% 
#                                 left_join(movie_bias_FMR_edx_train, by='movieId') %>% 
#                                 left_join(movie_time_bias_Lt.05reg_edx_train, by = c('movieId', 'bin_2_year')) %>% 
#                                 replace_na(list(b_g_fmr = 0, b_u_fmr = 0, b_i_fmr = 0, b_i_Lt = 0)) %>%
#                                 mutate(y_hat = mu_hat_edx_train + b_g_fmr + b_u_fmr + b_i_fmr + b_i_Lt) %>% 
#                                 pull(y_hat)

# RMSE of the FMR_GenreUserMovieTime_model against the edx_test data set:
# GenreUserMovieTime_FMR_rmse_edx_test <- RMSE(edx_test$rating, FMR_GenreUserMovieTime_model)
# GenreUserMovieTime_FMR_rmse_edx_test
# [1] 0.8617136

