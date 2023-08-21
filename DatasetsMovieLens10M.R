##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# https://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# dim(movielens)
# [1] 10000054        6

# movielens %>% group_by(userId) %>% summarise(UserRatings = n()) %>% summary()
#     userId       UserRatings    
# Min.   :    1   Min.   :  20.0  
# 1st Qu.:17943   1st Qu.:  35.0  
# Median :35799   Median :  69.0  
# Mean   :35782   Mean   : 143.1  
# 3rd Qu.:53620   3rd Qu.: 156.0  
# Max.   :71567   Max.   :7359.0

# library(lubridate)
# library(data.table)
# movielens_datetime <- movielens %>% mutate(date_and_time_UTC = as_datetime(timestamp)) %>% 
#                       select(userId, movieId, rating, date_and_time_UTC, title, genres)
# as.data.table(movielens_datetime)
#           userId movieId rating   date_and_time_UTC                                        title                                      genres
#        1:      1     122      5 1996-08-02 11:24:06                             Boomerang (1992)                              Comedy|Romance
#        2:      1     185      5 1996-08-02 10:58:45                              Net, The (1995)                       Action|Crime|Thriller
#        3:      1     231      5 1996-08-02 10:56:32                         Dumb & Dumber (1994)                                      Comedy
#        4:      1     292      5 1996-08-02 10:57:01                              Outbreak (1995)                Action|Drama|Sci-Fi|Thriller
#        5:      1     316      5 1996-08-02 10:56:32                              Stargate (1994)                     Action|Adventure|Sci-Fi
#       ---                                                                                                                                   
# 10000050:  71567    2107      1 1998-12-02 06:35:53         Halloween H20: 20 Years Later (1998)                             Horror|Thriller
# 10000051:  71567    2126      2 1998-12-03 01:39:03                            Snake Eyes (1998)               Action|Crime|Mystery|Thriller
# 10000052:  71567    2294      5 1998-12-02 05:52:48                                  Antz (1998) Adventure|Animation|Children|Comedy|Fantasy
# 10000053:  71567    2338      2 1998-12-02 05:53:36 I Still Know What You Did Last Summer (1998)                     Horror|Mystery|Thriller
# 10000054:  71567    2384      2 1998-12-02 05:56:13                 Babe: Pig in the City (1998)                             Children|Comedy

# min(movielens_datetime$date_and_time_UTC)
# [1] "1995-01-09 11:46:49 UTC"

# max(movielens_datetime$date_and_time_UTC)
# [1] "2009-01-05 05:02:16 UTC"

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # set.seed(1, sample.kind="Rounding") if using R 3.6 or later, otherwise set.seed(1) if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# dim(final_holdout_test)
# [1] 999999      6

# final_holdout_test[1:10,]
#    userId movieId rating  timestamp                                                   title                                  genres
# 1       1     231    5.0  838983392                                    Dumb & Dumber (1994)                                  Comedy
# 2       1     480    5.0  838983653                                    Jurassic Park (1993)        Action|Adventure|Sci-Fi|Thriller
# 3       1     586    5.0  838984068                                       Home Alone (1990)                         Children|Comedy
# 4       2     151    3.0  868246450                                          Rob Roy (1995)                Action|Drama|Romance|War
# 5       2     858    2.0  868245645                                   Godfather, The (1972)                             Crime|Drama
# 6       2    1544    3.0  868245920 Lost World: Jurassic Park, The (Jurassic Park 2) (1997) Action|Adventure|Horror|Sci-Fi|Thriller
# 7       3     590    3.5 1136075494                               Dances with Wolves (1990)                 Adventure|Drama|Western
# 8       3    4995    4.5 1133571200                                Beautiful Mind, A (2001)                   Drama|Mystery|Romance
# 9       4      34    5.0  844416936                                             Babe (1995)           Children|Comedy|Drama|Fantasy
# 10      4     432    3.0  844417070     City Slickers II: The Legend of Curly s Gold (1994)                Adventure|Comedy|Western

# length(unique(final_holdout_test$movieId))
# [1] 9809

# length(unique(final_holdout_test$title))
# [1] 9808

# length(unique(final_holdout_test$genres))
# [1] 773

# length(unique(final_holdout_test$userId))
# [1] 69534

# Find rows removed from final hold-out test
removed <- anti_join(temp, final_holdout_test)

# removed
#   userId movieId rating  timestamp                         title                 genres
# 1  16929   39412    3.0 1221160134    Living 'til the End (2005)                  Drama
# 2  20306   63826    4.0 1228431590               Splinter (2008) Action|Horror|Thriller
# 3  30445    8394    0.5 1200074027           Hi-Line, The (1999)                  Drama
# 4  32620   33140    3.5 1173562747         Down and Derby (2005)        Children|Comedy
# 5  40976   61913    3.0 1227767528           Africa addio (1966)            Documentary
# 6  59269   63141    2.0 1226443318 Rockin' in the Rockies (1945) Comedy|Musical|Western
# 7  60713    4820    2.0 1119156754  Won t Anybody Listen? (2000)            Documentary
# 8  64621   39429    2.5 1201248182                Confess (2005)         Drama|Thriller

# Add rows removed from final hold-out test set back into edx set
edx <- rbind(edx, removed)

# edx[1:36,]   # Sample of all of userId 1 and userId 2 movies
#    userId movieId rating timestamp                                                        title                                      genres
# 1       1     122      5 838985046                                             Boomerang (1992)                              Comedy|Romance
# 2       1     185      5 838983525                                              Net, The (1995)                       Action|Crime|Thriller
# 4       1     292      5 838983421                                              Outbreak (1995)                Action|Drama|Sci-Fi|Thriller
# 5       1     316      5 838983392                                              Stargate (1994)                     Action|Adventure|Sci-Fi
# 6       1     329      5 838983392                                Star Trek: Generations (1994)               Action|Adventure|Drama|Sci-Fi
# 7       1     355      5 838984474                                      Flintstones, The (1994)                     Children|Comedy|Fantasy
# 8       1     356      5 838983653                                          Forrest Gump (1994)                    Comedy|Drama|Romance|War
# 9       1     362      5 838984885                                      Jungle Book, The (1994)                  Adventure|Children|Romance
# 10      1     364      5 838983707                                        Lion King, The (1994)  Adventure|Animation|Children|Drama|Musical
# 11      1     370      5 838984596                    Naked Gun 33 1/3: The Final Insult (1994)                               Action|Comedy
# 12      1     377      5 838983834                                                 Speed (1994)                     Action|Romance|Thriller
# 13      1     420      5 838983834                                 Beverly Hills Cop III (1994)                Action|Comedy|Crime|Thriller
# 14      1     466      5 838984679                                  Hot Shots! Part Deux (1993)                           Action|Comedy|War
# 16      1     520      5 838984679                             Robin Hood: Men in Tights (1993)                                      Comedy
# 17      1     539      5 838984068                                  Sleepless in Seattle (1993)                        Comedy|Drama|Romance
# 19      1     588      5 838983339                                               Aladdin (1992) Adventure|Animation|Children|Comedy|Musical
# 20      1     589      5 838983778                            Terminator 2: Judgment Day (1991)                               Action|Sci-Fi
# 21      1     594      5 838984679                       Snow White and the Seven Dwarfs (1937)    Animation|Children|Drama|Fantasy|Musical
# 22      1     616      5 838984941                                       Aristocats, The (1970)                          Animation|Children
# 23      2     110      5 868245777                                            Braveheart (1995)                            Action|Drama|War
# 25      2     260      5 868244562 Star Wars: Episode IV - A New Hope (a.k.a. Star Wars) (1977)                     Action|Adventure|Sci-Fi
# 26      2     376      3 868245920                                       River Wild, The (1994)                             Action|Thriller
# 27      2     539      3 868246262                                  Sleepless in Seattle (1993)                        Comedy|Drama|Romance
# 28      2     590      5 868245608                                    Dances with Wolves (1990)                     Adventure|Drama|Western
# 29      2     648      2 868244699                                   Mission: Impossible (1996)           Action|Adventure|Mystery|Thriller
# 30      2     719      3 868246191                                          Multiplicity (1996)                                      Comedy
# 31      2     733      3 868244562                                             Rock, The (1996)                   Action|Adventure|Thriller
# 32      2     736      3 868244698                                               Twister (1996)           Action|Adventure|Romance|Thriller
# 33      2     780      3 868244698                         Independence Day (a.k.a. ID4) (1996)                 Action|Adventure|Sci-Fi|War
# 34      2     786      3 868244562                                                Eraser (1996)                       Action|Drama|Thriller
# 35      2     802      2 868244603                                            Phenomenon (1996)                               Drama|Romance
# 37      2    1049      3 868245920                           Ghost and the Darkness, The (1996)                            Action|Adventure
# 38      2    1073      3 868244562                   Willy Wonka & the Chocolate Factory (1971)             Children|Comedy|Fantasy|Musical
# 39      2    1210      4 868245644            Star Wars: Episode VI - Return of the Jedi (1983)                     Action|Adventure|Sci-Fi
# 40      2    1356      3 868244603                              Star Trek: First Contact (1996)            Action|Adventure|Sci-Fi|Thriller
# 41      2    1391      3 868246006                                         Mars Attacks! (1996)                        Action|Comedy|Sci-Fi

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# length(unique(edx$movieId))
# [1] 10677

# length(unique(edx$title))
# [1] 10676

# length(unique(edx$genres))
# [1] 797

# length(unique(edx$userId))
# [1] 69878

# Identify the unique movie-genres and add corresponding genreId numbers to edx dataset
edx_genreId <- data.table(genreId = seq(1:797), genres = unique(edx$genres))

# edx_genreId
#      genreId                             genres
#   1:       1                     Comedy|Romance
#   2:       2              Action|Crime|Thriller
#   3:       3       Action|Drama|Sci-Fi|Thriller
#   4:       4            Action|Adventure|Sci-Fi
#   5:       5      Action|Adventure|Drama|Sci-Fi
# ---                                           
# 793:     793          Animation|Documentary|War
# 794:     794 Adventure|Animation|Musical|Sci-Fi
# 795:     795         Fantasy|Mystery|Sci-Fi|War
# 796:     796                 Action|War|Western
# 797:     797                  Adventure|Mystery

# Adding corresponding genreId numbers to edx dataset
edx <- data.table(left_join(edx, edx_genreId, by = 'genres'))

# edx
#          userId movieId rating  timestamp                         title                        genres genreId
#       1:      1     122    5.0  838985046              Boomerang (1992)                Comedy|Romance       1
#       2:      1     185    5.0  838983525               Net, The (1995)         Action|Crime|Thriller       2
#       3:      1     292    5.0  838983421               Outbreak (1995)  Action|Drama|Sci-Fi|Thriller       3
#       4:      1     316    5.0  838983392               Stargate (1994)       Action|Adventure|Sci-Fi       4
#       5:      1     329    5.0  838983392 Star Trek: Generations (1994) Action|Adventure|Drama|Sci-Fi       5
#      ---                                                                                                     
# 9000051:  32620   33140    3.5 1173562747         Down and Derby (2005)               Children|Comedy      61
# 9000052:  40976   61913    3.0 1227767528           Africa addio (1966)                   Documentary     128
# 9000053:  59269   63141    2.0 1226443318 Rockin' in the Rockies (1945)        Comedy|Musical|Western     597
# 9000054:  60713    4820    2.0 1119156754  Won't Anybody Listen? (2000)                   Documentary     128
# 9000055:  64621   39429    2.5 1201248182                Confess (2005)                Drama|Thriller      49

# Partition the 80% Training and 10% Testing Datasets 
# 11.1112% of edx will produce a 10% Training Dataset equivalent in size (999999 rows) to the 10% final_holdout_test dataset with 999999 rows.
set.seed(1, sample.kind = "Rounding")
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.111112, list = FALSE)
edx_train <- edx[-edx_test_index, ]
edx_temp <- edx[edx_test_index, ]

# dim(edx_train)
# [1] 8000039       7  # PRE-rbind of edx_removed (see below for final dimensions)

# dim(edx_temp)
# [1] 1000016       7

# Make sure userId and movieId in the edx_test set are also in edx_train set
edx_test <- edx_temp %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Final dimensions of edx_test
# dim(edx_test)
# [1] 999999      7

# edx_test
#         userId movieId rating timestamp                                            title                                      genres genreId
#      1:      1     185      5 838983525                                  Net, The (1995)                       Action|Crime|Thriller       2
#      2:      1     520      5 838984679                 Robin Hood: Men in Tights (1993)                                      Comedy      14
#      3:      1     588      5 838983339                                   Aladdin (1992) Adventure|Animation|Children|Comedy|Musical      16
#      4:      2     376      3 868245920                           River Wild, The (1994)                             Action|Thriller      21
#      5:      2    1049      3 868245920               Ghost and the Darkness, The (1996)                            Action|Adventure      29
#     ---                                                                                                                                     
# 999995:  71567    1982      1 912580553                                 Halloween (1978)                                      Horror     115
# 999996:  71567    1986      1 912580553 Halloween 5: The Revenge of Michael Myers (1989)                                      Horror     115
# 999997:  71567    2012      3 912580722               Back to the Future Part III (1990)                       Comedy|Sci-Fi|Western     198
# 999998:  71567    2126      2 912649143                                Snake Eyes (1998)               Action|Crime|Mystery|Thriller     367
# 999999:  71567    2294      5 912577968                                      Antz (1998) Adventure|Animation|Children|Comedy|Fantasy      65

# length(unique(edx_test$movieId))
# [1] 9779

# length(unique(edx_test$title))
# [1] 9778

# length(unique(edx_test$genres))
# [1] 765

# length(unique(edx_test$userId))
# [1] 68548

# Find rows removed from the edx_test 
edx_removed <- anti_join(edx_temp, edx_test)

# edx_removed
#     userId movieId rating  timestamp                                                    title             genres genreId
#  1:   6905   60880    4.0 1222805003                    Family Game, The (Kazoku gêmu) (1983)       Comedy|Drama      43
#  2:  11680    6634    2.5 1060853936          Rowing with the Wind (Remando al viento) (1988)      Drama|Romance      28
#  3:  25055   63760    4.0 1228444806                                        Bellissima (1951)              Drama      34
#  4:  30840   61862    2.5 1229365440                               In Bed (En la cama) (2005)       Comedy|Drama      43
#  5:  38270   61695    4.5 1228254523                                          Ladrones (2007)              Drama      34
#  6:  40215    4075    1.0 1176305292         Monkey s Tale, A (Les Château des singes) (1999) Animation|Children      19
#  7:  47976    5676    2.5 1152215168                               Young Unknowns, The (2000)              Drama      34
#  8:  53315    6634    3.0 1110686356          Rowing with the Wind (Remando al viento) (1988)      Drama|Romance      28
#  9:  53315   31692    4.0 1122294562                                        Uncle Nino (2003)             Comedy      14
# 10:  56915   48899    4.0 1225708209                      Man of Straw (Untertan, Der) (1951)       Comedy|Drama      43
# 11:  59269    3383    3.0 1106423259                                         Big Fella (1937)      Drama|Musical      70
# 12:  59269   64897    3.0 1230162557                                            Mr. Wu (1927)              Drama      34
# 13:  59269   64903    3.5 1230162521               Nazis Strike, The (Why We Fight, 2) (1943)    Documentary|War     513
# 14:  59342   61768    0.5 1230070861                                Accused (Anklaget) (2005)              Drama      34
# 15:  63134   54318    2.5 1222631928 Cruel Story of Youth (Seishun zankoku monogatari) (1960)              Drama      34
# 16:  65308   64611    3.5 1230097394                                Forgotten One, The (1990)           Thriller      81
# 17:  67385    7537    2.5 1188277406                                Du côté de la côte (1958)        Documentary     128

# Add rows removed from the edx_test set back into the edx_train set
edx_train <- rbind(edx_train, edx_removed)

# Final dimensions of edx_train
# dim(edx_train)
# [1] 8000056       7

# edx_train
#          userId movieId rating  timestamp                                                    title                        genres genreId
#       1:      1     122    5.0  838985046                                         Boomerang (1992)                Comedy|Romance       1
#       2:      1     292    5.0  838983421                                          Outbreak (1995)  Action|Drama|Sci-Fi|Thriller       3
#       3:      1     316    5.0  838983392                                          Stargate (1994)       Action|Adventure|Sci-Fi       4
#       4:      1     329    5.0  838983392                            Star Trek: Generations (1994) Action|Adventure|Drama|Sci-Fi       5
#       5:      1     355    5.0  838984474                                  Flintstones, The (1994)       Children|Comedy|Fantasy       6
#      ---                                                                                                                                
# 8000052:  59269   64903    3.5 1230162521               Nazis Strike, The (Why We Fight, 2) (1943)               Documentary|War     513
# 8000053:  59342   61768    0.5 1230070861                                Accused (Anklaget) (2005)                         Drama      34
# 8000054:  63134   54318    2.5 1222631928 Cruel Story of Youth (Seishun zankoku monogatari) (1960)                         Drama      34
# 8000055:  65308   64611    3.5 1230097394                                Forgotten One, The (1990)                      Thriller      81
# 8000056:  67385    7537    2.5 1188277406                                Du côté de la côte (1958)                   Documentary     128

# length(unique(edx_train$movieId))
# [1] 10677

# length(unique(edx_train$title))
# [1] 10676

# length(unique(edx_train$genres))
# [1] 797

# length(unique(edx_train$userId))
# [1] 69878



