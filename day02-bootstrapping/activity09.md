Activity 9 - Bootstrapping
================

## Load the necessary packages

``` r
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.2 ──
    ## ✔ ggplot2 3.3.6     ✔ purrr   0.3.4
    ## ✔ tibble  3.2.1     ✔ dplyr   1.1.1
    ## ✔ tidyr   1.2.0     ✔ stringr 1.4.1
    ## ✔ readr   2.1.2     ✔ forcats 0.5.2
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

``` r
library(tidymodels)
```

    ## ── Attaching packages ────────────────────────────────────── tidymodels 1.0.0 ──
    ## ✔ broom        1.0.0     ✔ rsample      1.1.0
    ## ✔ dials        1.0.0     ✔ tune         1.0.0
    ## ✔ infer        1.0.3     ✔ workflows    1.0.0
    ## ✔ modeldata    1.0.0     ✔ workflowsets 1.0.0
    ## ✔ parsnip      1.0.1     ✔ yardstick    1.0.0
    ## ✔ recipes      1.0.1     
    ## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
    ## ✖ scales::discard() masks purrr::discard()
    ## ✖ dplyr::filter()   masks stats::filter()
    ## ✖ recipes::fixed()  masks stringr::fixed()
    ## ✖ dplyr::lag()      masks stats::lag()
    ## ✖ yardstick::spec() masks readr::spec()
    ## ✖ recipes::step()   masks stats::step()
    ## • Use tidymodels_prefer() to resolve common conflicts.

## Create the data

``` r
# Create a data frame/tibble named sim_dat
sim_dat <- tibble(
# Explain what next line is doing
  x1 = runif(20, -5, 5),
# Explain what next line is doing
  x2 = runif(20, 0, 100),
# Explain what next line is doing
  x3 = rbinom(20, 1, 0.5)
  )

b0 <- 2
b1 <- 0.25
b2 <- -0.5
b3 <- 1
sigma <- 1.5

errors <- rnorm(20, 0, sigma)

sim_dat <- sim_dat %>% 
  mutate(
    y = b0 + b1*x1 + b2*x2 + b3*x3 + errors,
    x3 = case_when(
      x3 == 0 ~ "No",
      TRUE ~ "Yes"
      )
    )
```

## Traditional MLR model

``` r
mlr_fit <- linear_reg() %>%
  set_mode("regression") %>% 
  set_engine("lm") %>% 
  fit(y ~ x1 + x2 + x3, data = sim_dat)

# Also include the confidence intervals for our estimated slope parameters
tidy(mlr_fit, conf.int = TRUE)
```

    ## # A tibble: 4 × 7
    ##   term        estimate std.error statistic  p.value conf.low conf.high
    ##   <chr>          <dbl>     <dbl>     <dbl>    <dbl>    <dbl>     <dbl>
    ## 1 (Intercept)    2.41     0.773       3.12 6.63e- 3   0.772      4.05 
    ## 2 x1             0.289    0.114       2.54 2.19e- 2   0.0475     0.530
    ## 3 x2            -0.502    0.0135    -37.1  5.97e-17  -0.531     -0.473
    ## 4 x3Yes          1.20     0.776       1.55 1.41e- 1  -0.445      2.85

## Bootstrapping

``` r
# Set a random seed value so we can obtain the same "random" results
set.seed(631)

# Generate the 2000 bootstrap samples
boot_samps <- sim_dat %>% 
  bootstraps(times = 2000)

boot_samps
```

    ## # Bootstrap sampling 
    ## # A tibble: 2,000 × 2
    ##    splits          id           
    ##    <list>          <chr>        
    ##  1 <split [20/8]>  Bootstrap0001
    ##  2 <split [20/6]>  Bootstrap0002
    ##  3 <split [20/6]>  Bootstrap0003
    ##  4 <split [20/6]>  Bootstrap0004
    ##  5 <split [20/10]> Bootstrap0005
    ##  6 <split [20/10]> Bootstrap0006
    ##  7 <split [20/7]>  Bootstrap0007
    ##  8 <split [20/6]>  Bootstrap0008
    ##  9 <split [20/8]>  Bootstrap0009
    ## 10 <split [20/6]>  Bootstrap0010
    ## # … with 1,990 more rows

``` r
boot_samps$splits[[1]] %>% analysis()
```

    ## # A tibble: 20 × 4
    ##         x1    x2 x3          y
    ##      <dbl> <dbl> <chr>   <dbl>
    ##  1 -2.41   79.0  No    -37.4  
    ##  2  0.338  74.8  Yes   -31.9  
    ##  3 -4.81   45.2  No    -21.9  
    ##  4  0.0786  1.54 Yes     1.31 
    ##  5  3.46   53.2  No    -24.8  
    ##  6 -4.82   64.3  Yes   -32.3  
    ##  7 -4.62   96.2  Yes   -45.3  
    ##  8 -0.0644 92.1  No    -43.9  
    ##  9 -4.62   47.6  No    -23.0  
    ## 10  4.86   74.7  Yes   -32.2  
    ## 11  4.86   74.7  Yes   -32.2  
    ## 12 -3.03    7.88 No      0.984
    ## 13  4.04   60.4  No    -25.7  
    ## 14 -4.62   47.6  No    -23.0  
    ## 15 -2.41   79.0  No    -37.4  
    ## 16  0.338  74.8  Yes   -31.9  
    ## 17  0.0786  1.54 Yes     1.31 
    ## 18  3.46   53.2  No    -24.8  
    ## 19  0.0786  1.54 Yes     1.31 
    ## 20  3.46   53.2  No    -24.8

``` r
# Create a function that fits a fixed MLR model to one split dataset
fit_mlr_boots <- function(split) {
  lm(y ~ x1 + x2 + x3, data = analysis(split))
}

# Fit the model to each split and store the information
# Also, obtain the tidy model information
boot_models <- boot_samps %>% 
  mutate(
    model = map(splits, fit_mlr_boots),
    coef_info = map(model, tidy)
    )

boots_coefs <- boot_models %>% 
  unnest(coef_info)

boots_coefs
```

    ## # A tibble: 8,000 × 8
    ##    splits         id            model  term     estim…¹ std.e…² stati…³  p.value
    ##    <list>         <chr>         <list> <chr>      <dbl>   <dbl>   <dbl>    <dbl>
    ##  1 <split [20/8]> Bootstrap0001 <lm>   (Interc…   1.64   0.809    2.02  6.01e- 2
    ##  2 <split [20/8]> Bootstrap0001 <lm>   x1         0.236  0.101    2.34  3.28e- 2
    ##  3 <split [20/8]> Bootstrap0001 <lm>   x2        -0.488  0.0117 -41.6   9.69e-18
    ##  4 <split [20/8]> Bootstrap0001 <lm>   x3Yes      1.09   0.690    1.58  1.34e- 1
    ##  5 <split [20/6]> Bootstrap0002 <lm>   (Interc…   3.30   1.04     3.16  6.05e- 3
    ##  6 <split [20/6]> Bootstrap0002 <lm>   x1         0.360  0.114    3.14  6.27e- 3
    ##  7 <split [20/6]> Bootstrap0002 <lm>   x2        -0.512  0.0146 -35.0   1.54e-16
    ##  8 <split [20/6]> Bootstrap0002 <lm>   x3Yes      0.538  0.797    0.675 5.09e- 1
    ##  9 <split [20/6]> Bootstrap0003 <lm>   (Interc…   3.28   0.829    3.95  1.14e- 3
    ## 10 <split [20/6]> Bootstrap0003 <lm>   x1         0.370  0.121    3.05  7.65e- 3
    ## # … with 7,990 more rows, and abbreviated variable names ¹​estimate, ²​std.error,
    ## #   ³​statistic

``` r
boot_int <- int_pctl(boot_models, statistics = coef_info, alpha = 0.05)
boot_int
```

    ## # A tibble: 4 × 6
    ##   term         .lower .estimate .upper .alpha .method   
    ##   <chr>         <dbl>     <dbl>  <dbl>  <dbl> <chr>     
    ## 1 (Intercept)  0.911      2.45   4.46    0.05 percentile
    ## 2 x1           0.0815     0.291  0.486   0.05 percentile
    ## 3 x2          -0.537     -0.504 -0.480   0.05 percentile
    ## 4 x3Yes       -0.131      1.31   2.84    0.05 percentile

``` r
ggplot(boots_coefs, aes(x = estimate)) +
  geom_histogram(bins = 30) +
  facet_wrap( ~ term, scales = "free") +
  geom_vline(data = boot_int, aes(xintercept = .lower), col = "blue") +
  geom_vline(data = boot_int, aes(xintercept = .upper), col = "blue")
```

![](activity09_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->
