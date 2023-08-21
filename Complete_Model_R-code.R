# COMPLETE MODEL R-CODE FOR MOVIE-LENS 10M RATING PREDICTIONS AND ACCURACY BY RMSE

library(tidyverse)
library(data.table)
library(lubridate)

# ACCURACY BY ROOT MEAN SQUARED ERROR (RMSE) LOSS FUNCTION ----
# Loss Function that computes the RMSE for ratings and their corresponding predictors:
RMSE <- function(True_Ratings, Predicted_Ratings){
  sqrt(mean((True_Ratings - Predicted_Ratings)^2))
}

# Determination of Biases and Latent Factors in MovieLens 10M Recommender Model, and then Final Holdout Test predictions and RMSE ----
# Order of model components developed below are all Bias components (see immediately below) and all Latent Factor components (starting on code line 491).
# Final Holdout Test predictions and RMSE starts on code line 798. (See Table of Contents selection bar at bottom of this R Script window for all other sections.)

# BIAS COMPONENTS MODEL DEVELOPMENT ----
# Order of model Bias components developed below are Overall Mean, Regularized Movie Bias, Regularized Temporal Movie Bias, 
# Regularized User Bias, Regularized Genre Bias, and Tikhonov Regularized User-Specific Movie-Genre Bias.  

# OVERALL MEAN RATING in edx_train ----
mu_hat_edx_train <- mean(edx_train$rating)

# CROSS VALIDATION OF REGULARIZATION PARAMETER LAMBDA FOR edx_train MOVIE BIAS + USER BIAS + GENRE BIAS ----
# The regularization parameter Lambda will be calculated using the baseline rating biases of movies, users, and genres 
# in a 3-way simultaneous calculation of the Lambda parameter for all three biases.

# dplyr slice_sample() function is used to generate random bootstrap samples at 100% sample size of the full edx_train data set,
# and thus the full edx_train$rating is used to test predicted ratings.

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

# MOVIE BIAS Full Model Regularized (FMR) ----
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

# MOVIE TEMPORAL BIAS Lt = 0.05 Regularized ---- 
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
# Lt
# [1] 0.05

# PARTIAL DISPLAY OF DATA FROM dSEidLt_train_df DATAFRAME, FOR PLOT FIGURE 4B OF REPORT:
# dSEidLt_train_df[51:60,]

#    Lambda_t MeanSlope_SquareError
# 51     0.00         -7.746359e-04
# 52     0.01         -6.056021e-04
# 53     0.02         -4.425634e-04
# 54     0.03         -2.852265e-04
# 55     0.04         -1.333156e-04
# 56     0.05          1.342904e-05 <- Minimum Slope rounds to
# 57     0.06          1.552521e-04       No More Than 0.00001
# 58     0.07          2.923840e-04            from Zero Slope
# 59     0.08          4.250420e-04
# 60     0.09          5.534312e-04

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

# USER BIAS Full Model Regularized (FMR) ----
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

# GENRE BIAS Full Model Regularized (FMR) ----
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
#                                 mutate(r_hat = mu_hat_edx_train + b_g_fmr + b_u_fmr + b_i_fmr + b_i_Lt) %>% 
#                                 pull(r_hat)

# RMSE of the FMR_GenreUserMovieTime_model against the edx_test data set:
# GenreUserMovieTime_FMR_rmse_edx_test <- RMSE(edx_test$rating, FMR_GenreUserMovieTime_model)
# GenreUserMovieTime_FMR_rmse_edx_test
# [1] 0.8617136

# USER-SPECIFIC MOVIE-GENRE BIASES: 2 PART CALCULATION; PART 1 = USER-SPECIFIC GENRE-ELEMENT BIASES; PART 2 = USER-SPECIFIC MOVIE-GENRE BIASES ----
# PART 1 ----
# PARALLEL USER ITERATIONS THROUGH TIKHONOV REGULARIZED LEFT INVERSE MATRIX (TRLIM) LEAST SQUARES FIT OF USER'S RATING RESIDUALS FOR USER'S MOVIE GENRE ELEMENTS:
library(tidyr)                                                    # user library for pivot_wider() function
library(dplyr)                                                    # user library for arrange() function
library(magrittr)                                                 # user library for pipe %>% operator
library(parallel)                                                 # system library for parallel processing functions
train_userIds <- unique(edx_train$userId)                         # length(unique(edx_train$userId)) = 69,878
Lambda_T <- 12.75                                                 # Tikhonov Lambda Regularization Parameter
core_cluster <- detectCores()/2                                   # detect number of CPU cores available and assign half of the cores to the cluster
wcl <- makeCluster(core_cluster)                                  # Initiate the working cluster   
clusterEvalQ(wcl, {                                               # Assign all non-System (user) libraries to each of the nodes in the working cluster
  library(tidyr)
  library(dplyr)                                                  
  library(magrittr) 
  })  
clusterExport(wcl, c("train_userIds", "Lambda_T", "edx_train", "edx_train_date", "movie_bias_FMR_edx_train", "movie_time_bias_Lt.05reg_edx_train", 
                     "user_bias_FMR_edx_train", "genre_bias_FMR_edx_train", "mu_hat_edx_train"))  # Assign all data to each of the nodes in the working cluster
# options(dplyr.summarise.inform = FALSE)                         # Option to suppress summarise() grouping messages.
timestamp() # Select/Highlight Code Lines 327 through 369 and click Run (takes about 16 hours to process all 69,878 users) ##------ Wed Jun 7  03:58:48  2023 ------##
UserIdGenreBiasesTRLIM12.75CVRT <- parSapply(wcl, train_userIds, function(user) {  # Parallel User Iteration through TRLIM Algorithm for 69,878 userIds
  A <- edx_train %>%                                              # Find all instances of genreIds for each userId in edx_train and separate each 
       select(userId, movieId, genreId, genres) %>%               # instance out into its individual genre elements, where the presence of a genre
       filter(userId == user) %>%                                 # element is indicated by a numeric "1" in the corresponding row of matrix A.
       separate_rows(genres, sep = "\\|") %>%                     # Separate genreId into individual genre elements with each genre on its own row.
       select(movieId, genreId, genres) %>%                       # Include movieId to separate any repeated genreIds into individual movieId rows.
       mutate(true = c(rep(1, times = length(genres)))) %>%       # Assigns a value of 1 to indicate the presence of a genre element in a genreId.
       pivot_wider(names_from = genres, values_from = true) %>%   # Pivot data table to wide matrix format with movieId/genreId rows & genre element columns.
       arrange(genreId) %>%                                       # Arrange by genreId to group rows of repeated genreIds together for matrix row reduction.
       as.matrix()
  A <- A[ , -1]                                                   # Delete the userId column from matrix A.
  A[is.na(A)] <- 0                                                # Replace all NAs in matrix A with numeric zeros "0".
  A <- A %>% as.data.frame() %>% group_by(genreId) %>% summarise_all(sum) %>% as.matrix() # Add rows with the same genreId together for row reduction.
  A <- A[ , -1]                                                   # Delete the genreId column from matrix A
  y <- edx_train_date %>% 
       left_join(movie_bias_FMR_edx_train, by ='movieId') %>%     # The user's genreId bias, b_ug_fmr, is estimated by the user's FMR movie rating 
       left_join(movie_time_bias_Lt.05reg_edx_train, by = c('movieId', 'bin_2_year')) %>%  # residuals in a genreId after subtracting out mu_hat_edx_train,  
       left_join(user_bias_FMR_edx_train, by ='userId') %>%       # b_i_fmr, b_i_Lt (0.05), b_u_fmr and b_g_fmr from each movie rating in the user's genreId 
       left_join(genre_bias_FMR_edx_train, by ='genreId') %>%     # of the edx_train data set.
       filter(userId == user) %>%                                 # Iterate algorithm through each userId listed in train_userIds.
       group_by(movieId, genreId) %>%                             # Grouping by movieId forces each instance of a genreId to be calculated as below.
       summarize(b_ug_fmr = (rating - mu_hat_edx_train - b_i_fmr - b_i_Lt - b_u_fmr - b_g_fmr)) %>%    # Note that this is NOT an average residual    
       arrange(genreId) %>%                                       # calculation, b_ug_fmr is calculated for each instance of a genreId for the user.
       as.matrix()                                                
  y <- y[ , 2:3]                                                  # Removed the movieId column from matrix y.
  y <- y %>% as.data.frame() %>% group_by(genreId) %>% summarise_all(sum) %>% as.matrix() # Add rows with the same genreId together, to match matrix A
  y <- matrix(y[ , 2])                                            # Removed the genreId from matrix y and restored its matrix format.  
  dimnames(y)[[2]] <- "b_ug_cvrt"                                 # Restore the column name of the residuals, which was lost in the previous step.
  I <- diag(dim(A)[2])                                            # Identity Matrix based on number of cols in matrix A, for compatibility with Left-Inverse of A.
  x_reg <- (solve(((t(A) %*% A) + (Lambda_T * I)))) %*% t(A) %*% y  # x_reg from Best Fit of y with the (Lambda_T * I) Regularized Left-Inverse of A.
  x_reg[is.na(x_reg)] <- 0
  user_genrebias <- x_reg %>% as.data.frame() %>% mutate(genres = rownames(x_reg))     # Prepare the final x_reg results as data frame user_genrebias.
  rownames(user_genrebias) <- c(seq(1:length(user_genrebias$genres)))
  user_genrebias <- user_genrebias %>%
                    mutate(userId = user) %>%                     # List userId iterated by algorithm from userIds listed in train_userIds.     
                    select(userId, genres, b_ug_cvrt)             # Format final x-reg genre-element biases for each user, b_ug_cvrt, for left join operations.
  user_genrebias$b_ug_cvrt[is.na(user_genrebias$b_ug_cvrt)] <- 0
  user_genrebias <- as.data.frame(t(as.matrix(user_genrebias)))
  return(user_genrebias)                                          # Append rows of user_genrebias to bottom end of UserIdGenreBiasesTRLIM12.75CVRT
})
stopCluster(wcl)
timestamp()                                                       ##------ Wed Jun 7  20:06:46  2023 ------##   16 hours  7 minuets  58 seconds  for 69,878 userIds

library(tidyverse)                                                # contains user library purrr for map_dfr() function
library(data.table)

run_index <- seq(1:69878)                      # matches the results in the matrix generated by running the algorithm on the selection of userIds.
# Format the matrix of run results into a data frame of results with a continuous and seamless listing of data by userId for left join operations.
UserIdGenreBiasesTRLIM12.75CVRT <- map_dfr(run_index, function(ri) {           # ri refers to the run_index of results generated by the algorithm.
  as.data.frame(t(as.matrix(UserIdGenreBiasesTRLIM12.75CVRT[[ri]])))                    
})

UserIdGenreBiasesTRLIM12.75CVRT$userId <- as.integer(UserIdGenreBiasesTRLIM12.75CVRT$userId)                      # reformat userId as an integer
UserIdGenreBiasesTRLIM12.75CVRT$genres <- as.factor(UserIdGenreBiasesTRLIM12.75CVRT$genres)                      # reformat the genres as factors
UserIdGenreBiasesTRLIM12.75CVRT$b_ug_cvrt <- as.numeric(UserIdGenreBiasesTRLIM12.75CVRT$b_ug_cvrt)              # format genre-element bias as numeric

UserIdGenreBiasesTRLIM12.75CVRT <- as_tibble(UserIdGenreBiasesTRLIM12.75CVRT)                                 # Reformat from a Table of Lists to a single Tibble

Weight <- 0.638                                                   # Tuned weight parameter applied to the hyperbolic tangent of b_ug_fmr (user's genre-element bias)
                                                                  # Add weighted hyperbolic tangent nonlinear transform (to between -1 and 1) with Weight = 0.638
UserIdGenreBiases.638wtanhTRLIM12.75CVRT <- UserIdGenreBiasesTRLIM12.75CVRT %>% 
                                            mutate(tanh.638_b_ug_cvrt = Weight * tanh(b_ug_cvrt)) %>% 
                                            select(userId, genres, tanh.638_b_ug_cvrt)                  
# UserIdGenreBiases.638wtanhTRLIM12.75CVRT
# A tibble: 1,085,796 × 3
#   userId genres    tanh.638_b_ug_cvrt
#    <int> <fct>                  <dbl>
# 1      1 Comedy              0.107   
# 2      1 Romance             0.000229
# 3      1 Action              0.0720  
# 4      1 Drama              -0.0200  
# 5      1 Sci-Fi              0.0136  
# 6      1 Thriller            0.0237  
# 7      1 Adventure           0.0246  
# 8      1 Children            0.0444  
# 9      1 Fantasy             0.0342  
#10      1 War                -0.000928
# ... with 1,085,786 more rows

# PART 2 ----
# Align each userId's genre-elements tanh.638_b_ug_cvrt data with the userId's individual genre elements in edx_train. Sum & format user's genreIds for left_join.
UserIdGenreBiases.638wtanhTRLIM12.75CVRT_edx_train <- edx_train %>% 
                                                      select(userId, movieId, genreId, genres) %>% 
                                                      separate_rows(genres, sep = "\\|") %>% 
                                                      left_join(UserIdGenreBiases.638wtanhTRLIM12.75CVRT, by = c('userId', 'genres')) %>%
                                                      group_by(userId, movieId, genreId) %>% 
                                                      summarise(b_ugId_cvrt = sum(tanh.638_b_ug_cvrt))

# List the aligned and calculated wtanhTRLIM genreId biases, b_ugId_cvrt, of all userIds in edx_train:
# UserIdGenreBiases.638wtanhTRLIM12.75CVRT_edx_train
# A tibble: 8,000,056 × 4
# Groups:   userId, movieId [8,000,056]
#   userId movieId genreId b_ugId_cvrt
#    <int>   <dbl>   <int>       <dbl>
# 1      1     122       1      0.108 
# 2      1     292       3      0.0893
# 3      1     316       4      0.110 
# 4      1     329       5      0.0902
# 5      1     355       6      0.186 
# 6      1     356       7      0.0867
# 7      1     362       8      0.0692
# 8      1     364       9      0.0452
# 9      1     370      10      0.179 
#10      1     377      11      0.0960
# ... with 8,000,046 more rows

# Confirm there are no missing values in the b_ugId_cvrt column of UserIdGenreBiases.638wtanhTRLIM12.75CVRT_edx_train
# sum(is.na(UserIdGenreBiases.638wtanhTRLIM12.75CVRT_edx_train$b_ugId_cvrt))
# [1] 0

# Align each userId's genre-elements tanh.638_b_ug_cvrt data with the userId's individual genre elements in edx_test. Sum & format user's genreIds for left_join.
UserIdGenreBiases.638wtanhTRLIM12.75CVRT_edx_test <- edx_test %>% 
                                                     select(userId, movieId, genreId, genres) %>% 
                                                     separate_rows(genres, sep = "\\|") %>% 
                                                     left_join(UserIdGenreBiases.638wtanhTRLIM12.75CVRT, by = c('userId', 'genres')) %>%
                                                     group_by(userId, movieId, genreId) %>% 
                                                     summarise(b_ugId_cvrt = sum(tanh.638_b_ug_cvrt)) %>%
                                                     replace_na(list(userId = 0, movieId = 0, genreId = 0, b_ugId_cvrt = 0))

# List the aligned and calculated wtanhTRLIM genreId biases, b_ugId_cvrt, of all userIds in edx_test:
# UserIdGenreBiases.638wtanhTRLIM12.75CVRT_edx_test
# A tibble: 999,999 × 4
# Groups:   userId, movieId [999,999]
#   userId movieId genreId b_ugId_cvrt
#    <int>   <dbl>   <int>       <dbl>
# 1      1     185       2      0.123 
# 2      1     520      14      0.107 
# 3      1     588      16      0.173 
# 4      2     376      21     -0.0649
# 5      2    1049      29      0.0587
# 6      3    1148      35     -0.0591
# 7      3    1552      24      0.0229
# 8      3    5527      28      0.126 
# 9      4     110      20      0.102 
#10      4     435      60     -0.0890
# ... with 999,989 more rows

# Confirm there are no missing values in the b_ugId_cvrt column of UserIdGenreBiases.638wtanhTRLIM12.75CVRT_edx_test
# sum(is.na(UserIdGenreBiases.638wtanhTRLIM12.75CVRT_edx_test$b_ugId_cvrt))
# [1] 0

# Calculate the predicted ratings from the model based on the w = 0.638 tanh Lambda_T = 12.75 TRLIM UserIdGenreBiases + 
# Full Model CV Regularized Movie-User-Genre Biases + LAMBDAt=0.05 Regularized Movie Temporal Bias:
# UserIdGenreBiases.638wtanhTRLIM12.75CVRT_model <- edx_test_date %>% 
#                                                   left_join(movie_time_bias_Lt.05reg_edx_train, by = c('movieId', 'bin_2_year')) %>% 
#                                                   replace_na(list(userId = 0, movieId = 0, rating = 0, genreId = 0, date = 0, bin_2_year = 0, b_i_Lt = 0)) %>%
#                                                   left_join(movie_bias_FMR_edx_train, by = 'movieId') %>%
#                                                   left_join(user_bias_FMR_edx_train, by = 'userId') %>% 
#                                                   left_join(genre_bias_FMR_edx_train, by = 'genreId') %>%
#                                                   left_join(UserIdGenreBiases.638wtanhTRLIM12.75CVRT_edx_test, by = c('userId', 'movieId', 'genreId')) %>%
#                                                   mutate(r_hat = mu_hat_edx_train + b_i_fmr + b_i_Lt + b_u_fmr + b_g_fmr + b_ugId_cvrt) %>% 
#                                                   pull(r_hat)

# RMSE of the UserIdGenreBiases.638wtanhTRLIM12.75CVRT_model against actual ratings in edx_test:
# UserIdGenreBiases.638wtanhTRLIM12.75CVRT_rmse_edx_test <- RMSE(edx_test$rating, UserIdGenreBiases.638wtanhTRLIM12.75CVRT_model)
# UserIdGenreBiases.638wtanhTRLIM12.75CVRT_rmse_edx_test
# [1] 0.8463952  

# LATENT FACTOR COMPONENTS MODEL DEVELOPMENT ----
# Order of development: Rating Residuals, Residuals Matrix, LRMC Imputed Residuals Matrix, SVD-PCA Factorization of LRMC Imputed Residuals Matrix, PC-106 Latent Factors
# See Section 2.6 of Project Report for background and context of SVD-PCA Latent Factor Collaborative Filtering Algorithm

# RATING RESIDUALS ----
# Calculate Residuals remaining in edx_train after subtracting mu_hat_edx_train - b_i_fmr - b_i_Lt - b_u_fmr - b_g_fmr - b_ugId_cvrt   
# from each respective movie rating:
residuals <- edx_train_date %>% 
             left_join(movie_bias_FMR_edx_train, by='movieId') %>%
             left_join(movie_time_bias_Lt.05reg_edx_train, by = c('movieId', 'bin_2_year')) %>%
             left_join(user_bias_FMR_edx_train, by='userId') %>% 
             left_join(genre_bias_FMR_edx_train, by='genreId') %>% 
             left_join(UserIdGenreBiases.638wtanhTRLIM12.75CVRT_edx_train, by = c('userId', 'movieId', 'genreId')) %>%
             mutate(residual = rating - mu_hat_edx_train - b_i_fmr - b_i_Lt - b_u_fmr - b_g_fmr - b_ugId_cvrt) %>% 
             pull(residual)
               
# Add the residuals to the edx_train data frame and select userId, movieId, and residual for a data frame that will be pivot_wider into a matrix for factorization:
residuals_df <- edx_train %>% mutate(residual = residuals) %>% select(userId, movieId, residual)

# residuals_df
#          userId movieId    residual
#       1:      1     122  0.70681746
#       2:      1     292  0.13120683
#       3:      1     316  0.34642868
#       4:      1     329  0.27057414
#       5:      1     355  0.84967701
#      ---                           
# 8000052:  59269   64903  0.24973223
# 8000053:  59342   61768 -0.03852123
# 8000054:  63134   54318 -0.58558810
# 8000055:  65308   64611  0.22483370
# 8000056:  67385    7537 -0.71586868

# RESIDUALS MATRIX ----
# Use pivot_wider and convert residuals_df to a matrix of residuals with userId rows and movieId columns:
residuals_matrix <- residuals_df %>% pivot_wider(names_from = movieId, values_from = residual) %>% as.matrix()

# residuals_matrix[1:10, 1:11]
#      userId       122        292       316       329      355        356      362         364        370        377
# [1,]      1 0.7068175  0.1312068 0.3464287 0.2705741 0.849677 -0.4190664 0.115587 -0.06079595  0.4322812 -0.1062309
# [2,]      2        NA         NA        NA        NA       NA         NA       NA          NA         NA         NA
# [3,]      3        NA         NA        NA        NA       NA         NA       NA          NA         NA         NA
# [4,]      4        NA -1.1969667 1.0173922 0.8217935       NA         NA       NA  0.25059994         NA -1.1518907
# [5,]      5        NA         NA        NA        NA       NA         NA       NA          NA         NA         NA
# [6,]      6        NA         NA        NA        NA       NA         NA       NA          NA         NA         NA
# [7,]      7        NA         NA        NA        NA       NA         NA       NA          NA         NA         NA
# [8,]      8        NA  0.0262706        NA        NA       NA         NA       NA -0.54480292 -1.6222095 -0.4571868
# [9,]      9        NA         NA        NA        NA       NA         NA       NA          NA         NA         NA
#[10,]     10        NA         NA        NA        NA       NA -1.1483094       NA          NA         NA         NA

# Create row names for residuals_matrix with the userId column:
rownames(residuals_matrix)<- residuals_matrix[ ,1]

# Delete redundant userId column from residuals_matrix:
residuals_matrix <- residuals_matrix[ ,-1]

# residuals_matrix[1:10, 1:10]
#          122        292       316       329      355        356      362         364        370        377
# 1  0.7068175  0.1312068 0.3464287 0.2705741 0.849677 -0.4190664 0.115587 -0.06079595  0.4322812 -0.1062309
# 2         NA         NA        NA        NA       NA         NA       NA          NA         NA         NA
# 3         NA         NA        NA        NA       NA         NA       NA          NA         NA         NA
# 4         NA -1.1969667 1.0173922 0.8217935       NA         NA       NA  0.25059994         NA -1.1518907
# 5         NA         NA        NA        NA       NA         NA       NA          NA         NA         NA
# 6         NA         NA        NA        NA       NA         NA       NA          NA         NA         NA
# 7         NA         NA        NA        NA       NA         NA       NA          NA         NA         NA
# 8         NA  0.0262706        NA        NA       NA         NA       NA -0.54480292 -1.6222095 -0.4571868
# 9         NA         NA        NA        NA       NA         NA       NA          NA         NA         NA
# 10        NA         NA        NA        NA       NA -1.1483094       NA          NA         NA         NA

# Final Dimensions of residuals_matrix:
# dim(residuals_matrix)
# [1] 69878 10677

# Proportion of NAs in residuals_matrix:
# sum(is.na(residuals_matrix)) / (dim(residuals_matrix)[1] * dim(residuals_matrix)[2])
# [1] 0.9892773   # Conclusion:  residuals_matrix is a very sparse matrix

# LRMC IMPUTED RESIDUALS MATRIX ----
# Imputation of NAs by Low-Rank Matrix Completion (LRMC)

# Load R package "cmfrec" (Collective Matrix Factorization for Recommender Systems)
library(cmfrec)

# Preserve original residuals_matrix (with NAs) by making copy of it as residuals_matrix_na
residuals_matrix_na <- residuals_matrix    

# Imputation of NAs by Low-Rank Matrix Completion (LRMC) uses the Collective Matrix Factorization for Recommender Systems package 
# cmfrec, which contains the CMF() function to factorize the original sparse data matrix X (i.e., R) into a LRMCmodel that consists 
# of two low-rank matrices of user factors (matrix A) and movie-item factors (matrix B).  The CMF() function assumes that X is a 
# sparse  matrix in which users represent rows, items represent columns, and the non-missing values denote explicit feedback movie 
# ratings from users on items.  LRMC imputation of the sparse data matrix X is completed by the imputeX() function in the cmfrec 
# package, which takes the LRMCmodel and the sparse matrix X as arguments.  Optimized CMF() parameters, for the LRMCmodel to minimize 
# RMSE on the edx_test dataset, are set at: k = 480, lambda = 0.059, niter = 30, center = TRUE, user_bias = TRUE, and item_bias = TRUE.

                    # LRMCmodel = (A, B) output of the CMF function 
LRMCmodel <- CMF(   # CMF function takes about 9 hours to process a 69878 rows x 10677 cols matrix when k = 480 and niter = 30
  X = residuals_matrix_na,   # residuals_matrix_na is an identical working copy of the original residuals_matrix (with NAs)
  k = 480,          # feature dimension of the low-rank factorization for user matrix A and item matrix B (where k_max = 10677)
  lambda = 0.059,   # L2-norm regularization parameter optimized at 0.059 for minimum edx_test RMSE (tuned for scale_lam = TRUE)    
  method = "als",   # alternating least-squares optimization to fit LRMCmodel (faster and more memory efficient than gradient)
  use_cg = FALSE,   # defaults to Cholesky algorithm to solve closed-form least squares, instead of less exact/stable gradient
  user_bias = TRUE, # adds user (row) intercepts to the LRMCmodel, which improves stability and RMSE of edx_test predictions 
  item_bias = TRUE, # adds item (column) intercepts to the LRMCmodel, which improves stability and RMSE of edx_test predictions
  center = TRUE,    # centers X data by subtracting mean values in the LRMCmodel, which improves stability and RMSE of edx_test
  scale_lam = TRUE, # increases lambda for each row in A & B according to number of non-missing entries in the X data for that row
  niter = 30,       # number of ALS iterations, better accuracy & lower RMSE with more iterations, balanced against time required
  finalize_chol = TRUE,  # performs final iteration with Cholesky solver, improves accuracy and is consistent with use_cg = FALSE
  NA_as_zero = FALSE,    # default setting = FALSE = do not take missing entries in X as zeros (to not interfere with imputation)
  nonneg = FALSE,  # default = FALSE = do not constrain LRMCmodel to be non-negative (residuals_matrix_na contains negative data)
  precompute_for_predictions = TRUE,   # default = TRUE (to prepare LRMCmodel for use in imputeX function)
  verbose = TRUE,   # monitor LRMCmodel progress through each of the 30 iterations of fitting low-rank matrices A and B
  handle_interrupt = TRUE,   # default = TRUE (to stop model with usable fitted model object if interrupt is necessary)
  seed = 1,   # seed for random number generation in the initial matrices of A & B, for reproducibility of imputed missing values
  nthreads = parallel::detectCores()/2  # number of parallel threads for parallel processing in the imputeX() function step below
)                                       # ALS method in the CMF() function does NOT use parallel processing, so nthreads ignored

# Impute the NAs in the Residuals Matrix with the LRMCmodel object using parallel processing (should only take about 15 minuets)
residuals_matrix_LRMCimputed <- imputeX(LRMCmodel, residuals_matrix_na, nthreads = LRMCmodel$info$nthreads)

# residuals_matrix_LRMCimputed[1:10, 1:10]  # 10 x 10 sample of the Imputed Residuals Matrix (compare with Residuals Matrix above)
#            122        292         316         329        355         356          362         364         370         377
# 1   0.70681746  0.1312068  0.34642868  0.27057414  0.8496770 -0.41906637  0.115586992 -0.06079595  0.43228125 -0.10623087
# 2  -0.11275461 -0.2859850 -0.39728850 -0.50004171 -0.1865108 -0.03442356 -0.117905348 -0.03537676 -0.15135524 -0.11329069
# 3   0.08737603  0.1800540  0.16998775  0.21741491  0.3397911  0.05116116  0.080488352  0.11376302  0.01929575  0.20284162
# 4  -0.02538663 -1.1969667  1.01739215  0.82179351  0.4597019  0.12572960  0.089626581  0.25059994 -0.11332941 -1.15189069
# 5  -0.30853230 -0.4348464 -0.60586481 -0.43799718 -0.3627595 -0.77151677 -0.100173464 -0.75742333  0.12018838 -0.61945370
# 6  -0.25407688 -0.1099476 -0.40221712 -0.30397864 -0.4470596  0.08707153 -0.032594767 -0.06973103 -0.32251466 -0.07546702
# 7  -0.17820901 -0.3433118 -0.42807820 -0.36384573 -0.3206117 -0.31544016 -0.191604615 -0.31578820  0.01758948 -0.11240441
# 8  -0.06695511  0.0262706  0.06276693 -0.07132330  0.1203327 -0.20883589 -0.056004616 -0.54480292 -1.62220947 -0.45718676
# 9  -0.50288924 -0.3414170 -0.07655912  0.02918981 -0.2683232 -0.48168915  0.060723537 -0.46408739 -0.04962225 -0.47611516
# 10 -0.05832001 -0.1053899  0.03464984  0.12593818 -0.0281441 -1.14830938 -0.001208181 -0.05544321  0.05254412 -0.10951058

# dim(residuals_matrix_LRMCimputed)   # Imputed Residuals Matrix is full-size with correct dimensions
# [1] 69878 10677

# sum(is.na(residuals_matrix_LRMCimputed))   # confirmed that no NAs remain in the Imputed Residuals Matrix
# [1] 0

# SVD-PCA FACTORIZATION OF LRMC IMPUTED RESIDUALS MATRIX ----   
# The prcomp factorization step takes about 6 hours to process:
svdpca_residuals_matrix_LRMCimputed <- prcomp(residuals_matrix_LRMCimputed, retx = TRUE, center = TRUE, scale = FALSE)

# Left Singular Matrix Factorization Results Scaled by Singular Values (10 x 10 sample)
# svdpca_residuals_matrix_LRMCimputed$x[1:10, 1:10]     
# userId     PC1         PC2         PC3        PC4        PC5          PC6        PC7        PC8        PC9      PC10
# 1    0.1557520  27.6375378  -3.9777994   5.660375 -0.7907444  0.662545406  0.1487092  0.2387962 -0.8625725 -1.351201
# 2    7.2961071 -10.2498144  -0.9069611   3.095912 -1.2661935  1.296814581 -0.2585148  0.6402068 -1.1842973  1.270696
# 3  -17.9646790   6.9912974  -4.7620942   2.266428  1.5492859 -5.131322439  0.6568630 -0.2884255  3.5048326 -2.802885
# 4   -8.4823953   0.6611347 -11.0015908  -2.205912  2.4222065 -3.598249264 -3.1153390 -3.1905203  0.8813063 -2.759444
# 5   23.1468028  -8.1526259   9.8087866  -3.539270 11.9371491  0.009587407 -7.2757422 -2.5399356  1.9421431 -1.019155
# 6   14.7103041 -11.7451553  -3.0922942  -6.189766 -8.3699837  1.896839344  0.5055226 -0.2117009  0.7277894 -4.586155
# 7   18.7777933 -19.5104135  -0.1694383   3.765573  9.4301991  2.774634893  0.7270994  4.7247704 -0.1398372 -3.393864
# 8   -6.2418060   5.8873125   6.9632110  -4.134962  3.0129710 -2.997591297  5.3726121  2.3527280  1.0237550  1.910803
# 9   16.9176129  -5.0984759   4.8448138 -10.715842 -0.7453049 -8.034034507 -2.2484934 -0.2660305  5.6206702 -1.725810
# 10   0.5959234   2.3260612 -10.1887284  -6.929450  5.7474632 -0.203276927 -0.1039334 -2.0162524  1.6709311  2.617967

# dim(svdpca_residuals_matrix_LRMCimputed$x)
# [1] 69878 10677

# Right Singular Matrix Factorization Results (10 x 10 sample)
# svdpca_residuals_matrix_LRMCimputed$rotation[1:10, 1:10]                  
# movieId       PC1          PC2          PC3           PC4           PC5          PC6           PC7          PC8          PC9         PC10
# 122 -0.0047609396  0.005878454  0.002736025  0.0177021073 -0.0034914472  0.009296556 -1.468093e-03  0.000889944 -0.006808686  0.004876590
# 292 -0.0125803434  0.005821761 -0.004319989  0.0069741986 -0.0095942228  0.002568180  1.877709e-03 -0.003667639 -0.003242854 -0.006411436
# 316 -0.0104468062  0.010997596 -0.000397542 -0.0063286907 -0.0095830512 -0.011622118  3.981542e-03  0.001886109 -0.001256331  0.015214102
# 329 -0.0066215550  0.009535891 -0.006963734 -0.0007428098 -0.0038134560 -0.012487149 -7.462593e-05  0.007503929  0.005033978 -0.021088669
# 355 -0.0112874321  0.012729643 -0.003833296  0.0058757651  0.0101829958  0.010220813  1.018531e-03  0.005383743  0.004991792 -0.012140622
# 356 -0.0152436287 -0.008007859 -0.004875772  0.0073352631 -0.0486102529  0.028301998 -2.637453e-03  0.013081528  0.001743616 -0.026030233
# 362  0.0030152564  0.006268940 -0.006663382  0.0056375614 -0.0059871011 -0.001847775 -9.382113e-03 -0.000011553 -0.004154932 -0.003899280
# 364 -0.0113582941 -0.003823022 -0.010699099  0.0139012001 -0.0144763038  0.006871465  1.516252e-02 -0.010326272 -0.020250462 -0.019171598
# 370 -0.0004961752  0.008558329  0.003824832  0.0203793899 -0.0007990453 -0.001450912 -3.958086e-03  0.010132043  0.017490301 -0.018641209
# 377 -0.0070468631 -0.002535332 -0.013594503  0.0275730759  0.0010408782  0.002998567  3.140116e-02 -0.011104738  0.012132058  0.002196028

# dim(svdpca_residuals_matrix_LRMCimputed$rotation)
# [1] 10677 10677

# PC-106 LATENT FACTORS ----
# Latent-Factor Sum Calculations Based on First 106 Principal Components:

# Load the "parallel" package for efficient processing: 
library(parallel) 

# Setup CPU Cores for Parallel Processing:
core_cluster <- detectCores()/2                               # assign half of the available CPU cores to a core_cluster object
wcl <- makeCluster(core_cluster)                              # make the core_cluster object a working cluster socket on the ‘localhost'

# Assign all data objects to each of the cores in the working cluster:
UserIndex <- 1:69878                                          # Full user index 
MovieIndex <- 1:10677                                         # Full movie index                         
P <- svdpca_residuals_matrix_LRMCimputed$x                    # Singular-Value Scaled SVD-PCA Left Singular Vectors of Principal Components. 
Qt <- t(svdpca_residuals_matrix_LRMCimputed$rotation)         # Transposed SVD-PCA Right Singular Vectors of Principal Components.
clusterExport(wcl, c("UserIndex", "MovieIndex", "P", "Qt"))   # Export vectors and matrices of data to the core nodes in the working cluster.

# Parallel S-Apply Function takes about 15 minuets to process the PC-106 Latent Factor Sums with 8 CPU cores in the working cluster:
LatentFactorSum_PC106LRMCimputed <- parSapply(wcl, UserIndex, function(ui){
  sapply(MovieIndex, function(mi){
    P[ui, 1:106] %*% Qt[1:106, mi]                            # Truncated User & Movie Vectors to 106 Principal Components for Latent Factor Sum Calculations
  })                                 
})
stopCluster(wcl)                                              # Stops the parallel working cluster to return the cores/CPUs to normal working mode.

# Check Dimensions of PC-106 Latent Factor Sums Results Matrix
# dim(LatentFactorSum_PC106LRMCimputed)
# [1] 10677 69878                                               # Dimensions show that the results matrix is transposed

# Ensure packages are loaded to support creating a transposed Data Table from the results matrix
library(tidyverse)
library(data.table)

# Transpose the results matrix to 69878 rows by 10677 columns, and convert into a data.table format
LatentFactorSum_PC106LRMCimputed_dt <- as.data.table(t(LatentFactorSum_PC106LRMCimputed))

# Verify dimensions of data.table are correct
# dim(LatentFactorSum_PC106LRMCimputed_dt)
# [1] 69878 10677

# Assign the movieId numbers as the column names in the data.table
colnames(LatentFactorSum_PC106LRMCimputed_dt) <- colnames(t(svdpca_residuals_matrix_LRMCimputed$rotation))

# Assign the userId numbers as a column of row names in the data.table
LatentFactorSum_PC106LRMCimputed_dt <- LatentFactorSum_PC106LRMCimputed_dt %>% 
                                        mutate(userId = rownames(svdpca_residuals_matrix_LRMCimputed$x)) %>% 
                                        select(userId, 1:10677)

# The data.table is now one column wider due to addition of the userId column 
# dim(LatentFactorSum_PC106LRMCimputed_dt)
# [1] 69878 10678

# PC-106 Latent Factor Sums Results (10 x 11 sample)
# LatentFactorSum_PC106LRMCimputed_dt[1:10, 1:11]
#     userId         122         292         316        329         355         356          362         364         370          377
#  1:      1  0.48260789  0.23653283  0.35533192  0.3474575  0.58984433 -0.52324319  0.207408079 -0.07289591  0.53959384 -0.152524916
#  2:      2 -0.04738856 -0.28591677 -0.37528951 -0.6026522 -0.19442097 -0.03402969 -0.113431827 -0.11578284 -0.04560144 -0.070438335
#  3:      3  0.08292988  0.28315704  0.20187958  0.2428388  0.39327865 -0.06497949  0.093609016  0.15971138 -0.01862918  0.189619334
#  4:      4 -0.03398417 -0.65592773  0.56302364  0.8647768  0.50490463  0.18232545  0.149386877  0.24833746 -0.10081180 -0.683909869
#  5:      5 -0.37917389 -0.49475448 -0.77352238 -0.3877704 -0.34956085 -0.88243929 -0.001331489 -0.93242060  0.06451146 -0.711925660
#  6:      6 -0.21185358 -0.07844738 -0.52287698 -0.2285524 -0.44075437  0.07266437 -0.018393208 -0.08481047 -0.32633912  0.094296358
#  7:      7 -0.19096753 -0.21777020 -0.29862736 -0.4300856 -0.35839663 -0.22685366 -0.249630663 -0.40111337  0.08134932 -0.169477253
#  8:      8 -0.06254857  0.03608508  0.14027859 -0.1567546  0.13909945 -0.34075098 -0.094190258 -0.07527481 -0.18828470 -0.009407292
#  9:      9 -0.40367416 -0.44472518 -0.08402934  0.1590944 -0.27016028 -0.63953273  0.222083846 -0.52562000  0.02007647 -0.539286742
# 10:     10 -0.08090274 -0.28263963  0.13260836  0.2197566 -0.04106331 -0.81239339 -0.086639934  0.01738328  0.08036595 -0.323648869

# Remove Gigabyte size matrices that should no longer be needed.  The next step is RAM intensive, so removing these from the RAM Environment is helpful.
rm(LatentFactorSum_PC106LRMCimputed, P, Qt)   

# Pivot the wide data.table into a long data.table with the PC-106 Latent Factor Sums sorted by userId and furthermore by movieId
LatentFactorSum_PC106LRMCimputed_dt_long <- pivot_longer(data = LatentFactorSum_PC106LRMCimputed_dt, 
                                                         cols = 2:10678, names_to = "movieId", values_to = "pc106")

# Reformat the userId and movieId columns as integer values instead of the default character values.  This will also reduce the table's RAM size by about 5GB.
LatentFactorSum_PC106LRMCimputed_dt_long <- LatentFactorSum_PC106LRMCimputed_dt_long %>% 
                                            mutate(userId = as.integer(userId), movieId = as.integer(movieId))

# TABLE OF THE PC-106 LATENT FACTORS ----
# LatentFactorSum_PC106LRMCimputed_dt_long
# A tibble: 746,087,406 × 3
#    userId movieId   pc106
#     <int>   <int>   <dbl>
#  1      1     122  0.483 
#  2      1     292  0.237 
#  3      1     316  0.355 
#  4      1     329  0.347 
#  5      1     355  0.590 
#  6      1     356 -0.523 
#  7      1     362  0.207 
#  8      1     364 -0.0729
#  9      1     370  0.540 
# 10      1     377 -0.153 
# ... with 746,087,396 more rows

# Verify the data.table contains no NA values for pc106
# sum(is.na(LatentFactorSum_PC106LRMCimputed_dt_long$pc106))
# [1] 0

# Remove Gigabyte size matrix that should no longer be needed.
rm(LatentFactorSum_PC106LRMCimputed_dt)

# Calculate the predicted ratings for edx_test based on the Complete Model per Equations 1.0 & 1.1 in Section 2.2 of the Project Report ----
# pca_residuals_PC106LRMCimputed_model adds the corresponding cross validated fully regularized  movie bias, user bias, genre bias, 
# the Lt 0.05 regularized movie time bin bias, the Tikhonov regularized user specific genre bias, and the sum of latent factors 
# from the 1st 106 principle components, to the mean rating of all movies:
# pca_residuals_PC106LRMCimputed_model <- edx_test_date %>% select(userId, movieId, genreId, bin_2_year) %>% 
#                                         left_join(movie_time_bias_Lt.05reg_edx_train, by = c('movieId', 'bin_2_year')) %>% 
#                                         replace_na(list(userId = 0, movieId = 0, genreId = 0, bin_2_year = 0, b_i_Lt = 0)) %>%
#                                         left_join(movie_bias_FMR_edx_train, by='movieId') %>%
#                                         left_join(user_bias_FMR_edx_train, by='userId') %>% 
#                                         left_join(genre_bias_FMR_edx_train, by='genreId') %>% 
#                                         left_join(UserIdGenreBiases.638wtanhTRLIM12.75CVRT_edx_test, by = c('userId', 'movieId', 'genreId')) %>%
#                                         left_join(LatentFactorSum_PC106LRMCimputed_dt_long, by = c('userId', 'movieId')) %>% 
#                                         mutate(r_hat = mu_hat_edx_train + b_i_fmr + b_i_Lt + b_u_fmr + b_g_fmr + b_ugId_cvrt + pc106) %>% 
#                                         pull(r_hat)

# Adjust (clip) predictions from pca_residuals_PC106LRMCimputed_model that are below 0.5 to be equal to min of 0.5 
# and adjust (clip) predictions that are above 5.0 to be equal to max of 5.0:
# adj_pca_residuals_PC106LRMCimputed_model <- pmax(0.5, pca_residuals_PC106LRMCimputed_model)  
# adj_pca_residuals_PC106LRMCimputed_model <- pmin(5.0, adj_pca_residuals_PC106LRMCimputed_model)  

# Confirm ratings distribution of predictions from adj_pca_residuals_PC106LRMCimputed_model
# summary(adj_pca_residuals_PC106LRMCimputed_model)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#  0.500   3.152   3.643   3.573   4.072   5.000 

# RMSE for the adjusted (clipped) pca_residuals_PC106LRMCimputed_model test dataset predictions against actual ratings in edx_test:
# adj_pca_residuals_PC106LRMCimputed_rmse_edx_test <- RMSE(edx_test$rating, adj_pca_residuals_PC106LRMCimputed_model_edx_test)
# adj_pca_residuals_PC106LRMCimputed_rmse_edx_test
# [1] 0.7924317   

# FINAL HOLDOUT TEST ----

library(tidyverse)
library(data.table)
library(lubridate)

# Add genreId numbers to the corresponding movie-genres in the final_holdout_test dataset
final_holdout_test <- data.table(left_join(final_holdout_test, edx_genreId, by = 'genres'))

# Assign each movie rating observation in the final_holdout_test dataset to a two_year time bin:
final_holdout_test <- final_holdout_test %>% mutate(date = as_datetime(timestamp), bin_2_year = round_date(date, unit = "2 years")) %>% 
                      select(-timestamp, -date)

# final_holdout_test
#         userId movieId rating bin_2_year                                             title                           genres genreId
#      1:      1     231      5 1996-01-01                              Dumb & Dumber (1994)                           Comedy      14
#      2:      1     480      5 1996-01-01                              Jurassic Park (1993) Action|Adventure|Sci-Fi|Thriller      31
#      3:      1     586      5 1996-01-01                                 Home Alone (1990)                  Children|Comedy      61
#      4:      2     151      3 1998-01-01                                    Rob Roy (1995)         Action|Drama|Romance|War      33
#      5:      2     858      2 1998-01-01                             Godfather, The (1972)                      Crime|Drama      66
#     ---                                                                                                                            
# 999995:  71566     235      5 1996-01-01                                    Ed Wood (1994)                     Comedy|Drama      43
# 999996:  71566     273      3 1996-01-01 Frankenstein (Mary Shelley's Frankenstein) (1994)                     Drama|Horror      55
# 999997:  71566     434      3 1996-01-01                                Cliffhanger (1993)        Action|Adventure|Thriller      24
# 999998:  71567     480      4 1998-01-01                              Jurassic Park (1993) Action|Adventure|Sci-Fi|Thriller      31
# 999999:  71567     898      4 1998-01-01                    Philadelphia Story, The (1940)                   Comedy|Romance       1

# Align each userId's genre-elements tanh.638_b_ug_cvrt data with the userId's individual genre elements in final_holdout_test. 
# Sum & format user's genreIds for left_join.
UserIdGenreBiases.638wtanhTRLIM12.75CVRT_holdout <- final_holdout_test %>% 
                                                    select(userId, movieId, genreId, genres) %>% 
                                                    separate_rows(genres, sep = "\\|") %>% 
                                                    left_join(UserIdGenreBiases.638wtanhTRLIM12.75CVRT, by = c('userId', 'genres')) %>%
                                                    group_by(userId, movieId, genreId) %>% 
                                                    summarise(b_ugId_cvrt = sum(tanh.638_b_ug_cvrt)) %>%
                                                    replace_na(list(userId = 0, movieId = 0, genreId = 0, b_ugId_cvrt = 0))

# User Specific genreId Biases formatted in table ready for left_join in Complete Model predictions for RMSE Accuracy determination  
# UserIdGenreBiases.638wtanhTRLIM12.75CVRT_holdout
# A tibble: 999,999 × 4
# Groups:   userId, movieId [999,999]
#    userId movieId genreId b_ugId_cvrt
#     <int>   <dbl>   <int>       <dbl>
#  1      1     231      14     0.107  
#  2      1     480      31     0.134  
#  3      1     586      61     0.152  
#  4      2     151      33    -0.00473
#  5      2     858      66     0      
#  6      2    1544     180     0      
#  7      3     590      22    -0.0355 
#  8      3    4995     266     0.177  
#  9      4      34     141     0.137  
# 10      4     432     151    -0.0304 
# ... with 999,989 more rows

# Calculate the predicted ratings for final_holdout_test based on the Complete Model per Equations 1.0 & 1.1 in Section 2.2 of the Project Report ----
# pca_residuals_PC106LRMCimputed_model adds the corresponding cross validated fully regularized  movie bias, user bias & genre bias,
# the L_t = 0.05 regularized movie time bin bias, the TRLIM-regularized tanh-transformed w-weighted user-specific movie-genre bias,  
# and the LRMC imputed sum of latent factors from the 1st 106 principle components to the mean rating of all movies:
pca_residuals_PC106LRMCimputed_model <- final_holdout_test %>% select(userId, movieId, genreId, bin_2_year) %>% 
                                        left_join(movie_time_bias_Lt.05reg_edx_train, by = c('movieId', 'bin_2_year')) %>% 
                                        replace_na(list(userId = 0, movieId = 0, genreId = 0, bin_2_year = 0, b_i_Lt = 0)) %>%
                                        left_join(movie_bias_FMR_edx_train, by='movieId') %>%
                                        left_join(user_bias_FMR_edx_train, by='userId') %>% 
                                        left_join(genre_bias_FMR_edx_train, by='genreId') %>% 
                                        left_join(UserIdGenreBiases.638wtanhTRLIM12.75CVRT_holdout, by = c('userId', 'movieId', 'genreId')) %>%
                                        left_join(LatentFactorSum_PC106LRMCimputed_dt_long, by = c('userId', 'movieId')) %>% 
                                        mutate(r_hat = mu_hat_edx_train + b_i_fmr + b_i_Lt + b_u_fmr + b_g_fmr + b_ugId_cvrt + pc106) %>% 
                                        pull(r_hat)

# Adjust (clip) predictions from pca_residuals_PC106LRMCimputed_model that are below 0.5 to be equal to min of 0.5 
# and adjust (clip) predictions that are above 5.0 to be equal to max of 5.0:
adj_pca_residuals_PC106LRMCimputed_model <- pmax(0.5, pca_residuals_PC106LRMCimputed_model)
adj_pca_residuals_PC106LRMCimputed_model <- pmin(5.0, adj_pca_residuals_PC106LRMCimputed_model)

# Confirm ratings distribution of predictions from adj_pca_residuals_PC106LRMCimputed_model
# summary(adj_pca_residuals_PC106LRMCimputed_model)
#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#  0.500   3.151   3.644   3.572   4.071   5.000

# RMSE for the adjusted (clipped) pca_residuals_PC106LRMCimputed_model final_holdout_test dataset predictions 
# against the actual ratings in the final_holdout_test dataset:
adj_pca_residuals_PC106LRMCimputed_rmse_holdout <- RMSE(final_holdout_test$rating, adj_pca_residuals_PC106LRMCimputed_model)
adj_pca_residuals_PC106LRMCimputed_rmse_holdout
# [1] 0.792085


