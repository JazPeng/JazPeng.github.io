---
layout: default
title:  "Scraping reviews from Google, Tripadvisor, DesignMyNight and Facebook"
permalink: "/scraping_reviews/"
---

# <center>Scraping reviews from Google, Tripadvisor, DesignMyNight and Facebook</center>
## <center>R</center>

One evening, my partner and I were having a conversation about how difficult it was for him to get an all-round view of how his business, a cocktail bar called Suckerpunch, was doing in terms of ratings. There are four main sites on which people tend to leave his business reviews; Google, Facebook, DesignMyNight and TripAdvisor. While he can access all four of these platforms individually, there was no concise way to consolidate all the reviews and get a net promoter score.

I decided to make a project out of doing this for him. The idea began with just getting the average score across all platforms and developed into how we could use the text in those reviews to improve the customer experience and identify pain points. We also talked about doing some time-series analysis to if there were any business developments (such as a refurb or a management change) had any significant effect. The final product will be a shiny app that provides all of this data in an easy-to-interpret interface.

But first, I had to get the data I needed from each source.

## TripAdvisor
TripAdvisor does have an API, but it was not appropriate for my use case. After contacting them to make sure I was not violating their terms of service, I used web scraping to get the data I needed.

```R
library(xml2)
library(rvest)
library(dplyr)
library(httr)
library(readr)
library(stringr)
```

I wanted to automate the scraping as much as possible by creating a function that iterates through every relevant URL and returns the relevant data. First, I had to identify the convention of the URLs and then create a list of them.


```R
tripadvisor_urls <- c()

for (n in seq(0, 80, 10)) {
    url <- paste('https://www.tripadvisor.co.uk/Attraction_Review-g186306-d9756771-Reviews-or',n,
           '-Suckerpunch_St_Albans-St_Albans_Hertfordshire_England.html', sep = "")
    tripadvisor_urls <- c(tripadvisor_urls, url)
    }
```

Next, I created a function that can easily be reused to get the same data for different establishments - perhaps one day I may want to do similar analysis for competitors, or for his future businesses.


```R
get_one_page <- function(x) {
    reviews <- x %>%
      read_html() %>%
      html_nodes("#REVIEWS .innerBubble")

    id <- reviews %>%
      html_node(".quote a") %>%
      html_attr("id")

    headline_quote <- reviews %>%
      html_node(".quote span") %>%
      html_text()

    rating <- reviews %>%
        html_nodes(".ratingInfo") %>%
        as.character() %>%
        substr(63, 63) %>%
        as.numeric()

    date <- reviews %>%
      html_node(".rating .ratingDate") %>%
      html_attr("title")

    data.frame(id, as.Date(date, '%d %B %Y'), rating, headline_quote)
    }
```


```R
all_pages <- lapply(tripadvisor_urls, get_one_page)
tripadvisor_data <- plyr::ldply(all_pages, data.frame)
```

While the above method got me a lot of the content that I needed, using it to scrape the actual content of the reviews only yields truncated text. To get the full text, I had to create a separate function that loops through each individual review's URL, using the id column of the data frame I created.


```R
full_review_urls <- c()

for (id in tripadvisor_data$id) {
    url_id <- substr(id, 3,nchar(id))
    url <- paste('https://www.tripadvisor.co.uk/ShowUserReviews-g186306-d9756771-r',
                 url_id,'-Suckerpunch_St_Albans-St_Albans_Hertfordshire_England.html', sep = "")
    full_review_urls <- c(full_review_urls, url)
    }
```


```R
get_full_review <- function(x) {
    reviews <- x %>%
      read_html() %>%
      html_node("#REVIEWS .innerBubble")

    id <- reviews %>%
      html_node(".quote a") %>%
      html_attr("id")

    review <- reviews %>%
      html_node(".entry .partial_entry") %>%
      html_text()

    data.frame(id, review)
}
```


```R
full_reviews_dfs <- lapply(full_review_urls, get_full_review)
full_reviews__final <- plyr::ldply(full_reviews_dfs, data.frame)
```

I now have two data frames - one with my full review text and one with my metadata. I made sure that there was an id column in both to act as a unique identifier so I could merge them into one.


```R
tripadvisor <- left_join(tripadvisor_data, full_reviews__final, by = 'id')
```


```R
write.csv(tripadvisor, file = "ta_reviews.csv")
```

## Facebook
Facebook was by far the easiest and simplest platform I used. I simply set up my page access tokens [here](https://developers.facebook.com/tools/explorer/) and created a URL from their URL builder. With just a few lines of code I got everything I required.


```R
access_token <- ****
```


```R
fb_url <- paste0("https://graph.facebook.com/v3.0/me?fields=ratings.limit(20000)%7Bcreated_time%2Crating%2Creview_text%2Creviewer%7D&access_token=",access_token)
```


```R
facebook_function <- function(x) {
    returned <- GET(x)
    c <- content(returned)$ratings
    new <- data.frame()
        for (i in c[[1]]) {
            df <- data.frame(as.list(unlist(i)))
            new <- bind_rows(new, df)
    }
    return(new)
}
```


```R
fb_result <- facebook_function(fb_url)
```


```R
write.csv(fb_result, file = "fb_reviews.csv")
```

## DesignMyNight
DesignMyNight had no terms of service to abide by, or API to access. Web scraping the public "Suckerpunch" DesignMyNight page was not an option as there were only a limited number of reviews showing at one time. To access every review, I had to web scrape from the admin page which required credentials.

```R
url <- "https://www.designmynight.com/dmn-admin/page/5718e8fe7fb8d76363cea4ce/edit"
pgsession <- html_session(url)
pgform <- html_form(pgsession)
```


```R
filled_form <- set_values(pgform[[1]],
                          "email" = ****,
                          "password" = ****)
```


```R
filled_form$url <- ""
submit_form(pgsession, filled_form)
review_page <- jump_to(pgsession, "https://www.designmynight.com/dmn-admin/page/5718e8fe7fb8d76363cea4ce/edit")
```


```R
reviews <- read_html(review_page) %>%
    html_nodes(".review-details") %>%
    html_text()
```


```R
read_html(review_page) %>%
    html_nodes(".review-details")
```

Once I had the reviews, I needed to manipulate the text to get the appropriate data in the right columns.

```R
date_split <- str_split(reviews, "Username", simplify = TRUE)
date <- str_replace(date_split[,1], "Date", "")

name_split <- str_split(date_split[,2], "Rating", simplify = TRUE)
name <- name_split[,1]

rating_split <- str_split(name_split[,2], "Review", simplify = TRUE)
rating <- rating_split[,1]

review <- str_replace(rating_split[,2], "Venue replyAdd reply", "")

reviews_df <- data.frame(date, name, rating, review)
```


```R
write.csv(reviews_df, file = "dmn_reviews.csv")
```

## Google Reviews
This was by far the most time consuming and inefficient method I used. While Google do have a business API, they refused myself and my partners' application for access without explanation. Only five reviews are available without paying a fee, so I had to resort to the what felt like quite a primitive method; I copied and pasted all the reviews for the business into a txt file and performed regular expressions to get the data I needed.

While this method is not scalable, it did give me a chance to practice string manipulation in R which I actually found to be relatively simple and straightforward.

```R
google <- read_delim("google_reviews.txt", delim = "/n", col_names = FALSE)
```

## Separate into 5 star, 4 star, 3 star, 2 star and 1 star dataframes
The copying and pasting didn't actually carry over the rating number, so I had to separate the ratings into the appropriate ratings sections. I found that these sections were easily separated at the pattern " star".


```R
filter(google, grepl(" star", X1))
```


```R
star_rows <- grep(" star", google$X1)
star_rows
```


```R
google_5 <- google[2:234, "X1"]
google_4 <- google[236:370, "X1"]
google_3 <- google[372:394, "X1"]
google_1 <- google[396:nrow(google), "X1"]
```

### Here I tidied the data by separating out each individual review, dropped all the rows I didn't need, transposed the reviews and added a column with the appropriate rating.
### Five star
To make the process as painless as possible, I defined numerous functions that could be resused with the other star ratings. This function first locates all indices of the rows that contain "Like", which appear to signal the split between each unique reviews. It then uses these locations to split the data frames.


```R
sep_on_likes <- function(google_star) {
    review_list <- list()

    pattern <- c("Like", "1$")
    likes <- grep(paste(pattern,collapse="|"), google_star$X1)


    for (i in 1:length(likes)) {
        if (likes[i] == likes[1]) {
            review_list <- c(review_list, list(google_star[1:likes[i]-1, "X1"]))
        }
        else if (likes[i] == likes[length(likes)]) {
            review_list <- c(review_list, list(google_star[likes[i]:nrow(google_star), "X1"]))
            }
        else {
            review_list <- c(review_list, list(google_star[likes[i]:likes[i+1], "X1"]))
        }
        }
    return(review_list)
}
```


```R
sep_on_likes_5 <- sep_on_likes(google_5)
```

When the individual reviews were in data frame format, I saw that the last element in the list has a bunch of reviews that could not be "liked" (potentially because that are ratings without review text). I separated them out from the rest and work with the "likeable" ones for now.

For the "likeable" ones, I saw that second, fifth and sixth element are always the information that I need (name, date and review text). I created a function that turns those into data frames where each of these elements are columns.


```R
loop_5 <- sep_on_likes_5[1:length(sep_on_likes_5)-1]
no_review_5 <- sep_on_likes_5[length(sep_on_likes_5)]
```


```R
make_df <- function(loop) {
    df <- NULL

    for (each in 1:length(loop)) {
        if (unlist(loop_5[each])[1] != "Like") {
            unlist <- list(unique(unlist(loop[each]))[c(1,3,4)])
        }
        else {
            unlist <- list(unique(unlist(loop[each]))[c(2,4,5)])
        }
    df <- rbind(df, data.frame(unlist[[1]][[1]], unlist[[1]][[2]], unlist[[1]][[3]]))
    }
    colnames(df) <- c("name", "date", "review")
    return(df)
}
```


```R
five_stars_df <- make_df(loop_5)
five_stars_df <- five_stars_df %>%
    filter(name != 1)
```

Then I worked on the "unlikeable" reviews that have no text. Rather than being able to be separated by "Like", I saw that the rows that contain "* ago" were the best ones to use to separate the text. I defined a function that puts all those reviews into a data frame and then appended them to the original five_stars_df.


```R
make_df_no_text <- function(x) {
    df_final <- NULL
    ux <- unlist(x)
    idx <- cumsum(grepl(" ago", ux)) - grepl(" ago", ux)
    x <- split(ux, idx)
    for (i in 1:length(x)) {
            y <- unlist(x[i])[c(2,length(unlist(x[i])))]
            df <- t(data.frame(y))
            rownames(df) <- c()
            colnames(df) <- c("name", "date")
            df <- as.data.frame(df)
            df$review <- ""
            df_final <- rbind(df_final, df)

            }
    return(as.data.frame(df_final))
}
```


```R
no_text_5 <- make_df_no_text(no_review_5)
five_stars_df <- rbind(five_stars_df, no_text_5)
five_stars_df$rating <- 5
```

#### Four star


```R
sep_on_likes_4 <- sep_on_likes(google_4)
loop_4 <- sep_on_likes_4[1:length(sep_on_likes_4)-1]
no_review_4 <- sep_on_likes_4[length(sep_on_likes_4)]
four_stars_df <- make_df(loop_4)
no_text_4 <- make_df_no_text(no_review_4)
```

The below is one row that slipped through the function, so it requires some manual wrangling.


```R
make_df(loop_4[10])
```


```R
a <- t(data.frame(list(unique(unlist(loop_4[10]))[c(2,4,5)])))
rownames(a) <- c()
colnames(a) <- c("name", "date", "review")
a
```


```R
four_stars_df <- four_stars_df %>%
    filter(name != "Like") %>%
    rbind(a)
four_stars_df$rating <- 4
```

### Three star


```R
sep_on_likes_3 <- sep_on_likes(google_3)
loop_3 <- sep_on_likes_3[1:length(sep_on_likes_3)-1]
no_review_3 <- sep_on_likes_3[length(sep_on_likes_3)]
three_stars_df <- make_df(loop_3)
no_text_3 <- make_df_no_text(no_review_3)
three_stars_df <- rbind(three_stars_df, no_text_3)
three_stars_df$rating <- 3
```

### One star


```R
sep_on_likes_1 <- sep_on_likes(google_1)
one_star_df <- make_df(sep_on_likes_1[1])
one_star_df$rating <- 1
```

### Make a master data frame to extract as a CSV for further analysis

```R
all_reviews <- rbind(five_stars_df, four_stars_df, three_stars_df, one_star_df)
all_reviews
```


```R
write.csv(all_reviews, "google_reviews.csv")
```

The next step is combining all the above data frames into one, cleaning the data and getting some initial insights out of them. See my next post to see how I did it.
