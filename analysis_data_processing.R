options(dplyr.summarise.inform = FALSE)

library(tidyverse)
library(tidybayes)
library(posterior)
library(jsonlite)
library(kableExtra)
library(gt)
source("plot_utilities.R")
source("table_functions.R")

# Function to save plot as PNG
save_plot_as_png <- function(plot, filename, width = 10, height = 8) {
    # Create plots directory if it doesn't exist
    if (!dir.exists("shiny_plots")) {
        dir.create("shin_plots")
    }

    # Save the plot
    ggsave(
        filename = file.path("shiny_plots", filename),
        plot = plot,
        width = width,
        height = height,
        dpi = 300
    )
}

# Function to load and process data for a specific model rank
load_model_data <- function(
    model_rank,
    prefix = "preg_deaths",
    suffix = "quarterly_all"
) {
    # Load the base data
    df <- read_csv(sprintf('results/df_%s_%s.csv', prefix, suffix))

    # Process the base data
    df <- df |>
        group_by(state) |>
        mutate(banned_state = 1 * any(exposed == 1)) |>
        ungroup()

    df$start_date <- df$date
    df$end_date <- df$date + months(1) - days(1)

    # Rename columns
    df <- df |>
        rename_with(
            ~ {
                suffix_part <- str_remove(., paste0("^", prefix, "_"))
                paste0("deaths_", suffix_part)
            },
            .cols = starts_with(prefix)
        ) |>
        select(
            date,
            state,
            starts_with("deaths_"),
            starts_with("births_"),
            exposed,
            banned_state,
            start_date,
            end_date
        )

    # Load model samples
    csv_prefix <- paste0("results/Poisson_", prefix)
    all_samples <- read_csv(sprintf(
        "%s_%s_%s.csv",
        csv_prefix,
        model_rank,
        suffix
    ))

    # Merge draws and data
    agg_category_name = "total"
    merged_df <- merge_draws_and_data(
        df,
        all_samples,
        categories = c("nhwhite", "nhblack", "hispanic", "nhother"),
        agg_category_name = agg_category_name,
        outcome_string = "deaths",
        denom_string = "births"
    )

    # Add model rank to the dataframe
    merged_df$model_rank <- model_rank

    return(merged_df)
}

# # Load data for all model ranks
# model_ranks <- 1:5 # Adjust this range based on your actual model ranks
# all_merged_df <- map_dfr(model_ranks, load_model_data)

# Define the resolutions and death types to process

# covid_exclusions <- c("", "_with_covid")
covid_exclusions <- c("")
# resolutions <- c("biannual", "quarterly", "monthly")
resolutions <- c("quarterly")
death_types <- c("preg_deaths", "mat_deaths")

# Generate and save tables
for (covid in covid_exclusions) {
    for (rank in 1:6) {
        for (death_type in death_types) {
            for (resolution in resolutions) {
                print(sprintf(
                    "Type: %s, Resolution: %s, Rank: %d, %s",
                    death_type,
                    resolution,
                    rank,
                    covid
                ))
                print("--------------------------------")

                # Check if files exist for this combination
                data_file <- sprintf(
                    'results/df_%s_%s_all%s.csv',
                    death_type,
                    resolution,
                    covid
                )
                model_file <- sprintf(
                    'results/Poisson_%s_%s_%s_all%s.csv',
                    death_type,
                    rank,
                    resolution,
                    covid
                )

                # Skip if files don't exist
                if (!file.exists(data_file) || !file.exists(model_file)) {
                    print("Skipping because results don't exist")
                    next
                }

                # Load data for this death type and resolution
                df <- read_csv(data_file)

                # Process the base data
                df <- df |>
                    group_by(state) |>
                    mutate(banned_state = 1 * any(exposed == 1)) |>
                    ungroup()

                df$start_date <- df$date
                # Set end_date based on resolution
                df$end_date <- case_when(
                    resolution == "monthly" ~ df$date + months(1) - days(1),
                    resolution == "bimonthly" ~ df$date + months(2) - days(1),
                    resolution == "quarterly" ~ df$date + months(3) - days(1),
                    resolution == "biannual" ~ df$date + months(6) - days(1),
                    resolution == "annual" ~ df$date + months(12) - days(1),
                    TRUE ~ df$date + months(1) - days(1) # default to monthly
                )

                # Rename columns
                df <- df |>
                    rename_with(
                        ~ {
                            suffix_part <- str_remove(
                                .,
                                paste0("^", death_type, "_")
                            )
                            paste0("deaths_", suffix_part)
                        },
                        .cols = starts_with(death_type)
                    ) |>
                    select(
                        date,
                        state,
                        starts_with("deaths_"),
                        starts_with("births_"),
                        exposed,
                        banned_state,
                        start_date,
                        end_date
                    )

                # Load model samples
                all_samples <- read_csv(model_file)

                # Merge draws and data
                agg_category_name = "total"
                merged_df <- merge_draws_and_data(
                    df,
                    all_samples,
                    categories = c("nhwhite", "nhblack", "hispanic", "nhother"),
                    agg_category_name = agg_category_name,
                    outcome_string = "deaths",
                    denom_string = "births"
                )
                # Create quantiles data for each model rank
                quantiles_df <- merged_df %>%
                    group_by(category, state, date) %>%
                    summarize(
                        ypred_mean = mean(ypred),
                        ypred_lower = quantile(ypred, 0.025),
                        ypred_upper = quantile(ypred, 0.975),
                        deaths = mean(deaths),
                        exposure_code = first(exposure_code),
                        banned_state = first(banned_state)
                    ) %>%
                    ungroup()

                # Add model rank to the dataframe
                merged_df$model_rank <- rank

                # Create PPC plots directory if it doesn't exist
                ppc_dir <- file.path("shiny_plots", "ppc")
                if (!dir.exists(ppc_dir)) {
                    dir.create(ppc_dir)
                }

                # Define states and categories for PPC
                ppc_states <- c("TX", "Ban States", "Unexposed States")
                categories <- c(
                    "total",
                    "nhwhite",
                    "nhblack",
                    "hispanic",
                    "nhother"
                )

                # Generate and save RMSE PPC plots
                rmse_res <- make_rmse_ppc_plot(
                    merged_df %>% filter(state %in% ppc_states),
                    categories = categories,
                    outcome = "deaths"
                )
                ggsave(
                    file.path(
                        ppc_dir,
                        sprintf(
                            "rmse_ppc_rank%d_%s_%s%s.png",
                            rank,
                            death_type,
                            resolution,
                            covid
                        )
                    ),
                    plot = rmse_res$rmse_plt,
                    width = 10,
                    height = 8
                )

                # Generate and save Absolute Residuals PPC plots
                abs_res <- make_abs_res_ppc_plot(
                    merged_df %>% filter(state %in% ppc_states),
                    categories = categories,
                    outcome = "deaths"
                )
                ggsave(
                    file.path(
                        ppc_dir,
                        sprintf(
                            "abs_res_ppc_rank%d_%s_%s%s.png",
                            rank,
                            death_type,
                            resolution,
                            covid
                        )
                    ),
                    plot = abs_res$max_plt,
                    width = 10,
                    height = 8
                )

                # Generate and save ACF PPC plots for different lags
                for (lag in c(1, 2, 4)) {
                    acf_ppc <- make_acf_ppc_plot(
                        merged_df %>% filter(state %in% ppc_states),
                        categories = categories,
                        lag = lag,
                        outcome = "deaths"
                    )
                    ggsave(
                        file.path(
                            ppc_dir,
                            sprintf(
                                "acf%d_ppc_rank%d_%s_%s%s.png",
                                lag,
                                rank,
                                death_type,
                                resolution,
                                covid
                            )
                        ),
                        plot = acf_ppc$acf_plt,
                        width = 10,
                        height = 8
                    )
                }

                # Generate and save Unit Correlation PPC plots
                uc_ppcs_obj <- make_unit_corr_ppc_plot(
                    merged_df,
                    categories = categories,
                    outcome = "deaths"
                )
                ggsave(
                    file.path(
                        ppc_dir,
                        sprintf(
                            "unit_corr_ppc_rank%d_%s_%s%s.png",
                            rank,
                            death_type,
                            resolution,
                            covid
                        )
                    ),
                    plot = uc_ppcs_obj$eval_plt,
                    width = 10,
                    height = 8
                )

                # Save PPC results to a file
                ppc_results_file <- file.path(ppc_dir, "ppc_results.txt")
                all_pass <- all(
                    !(any(rmse_res$pval < 0.1) | any(rmse_res$pval > 0.9)),
                    !(any(abs_res$pval < 0.1) | any(abs_res$pval > 0.9)),
                    !(any(acf_ppc$pval < 0.1) | any(acf_ppc$pval > 0.9)),
                    !(any(uc_ppcs_obj$pval < 0.1) | any(uc_ppcs_obj$pval > 0.9))
                )
                results_str <- sprintf(
                    "Type: %s, Resolution: %s, Rank: %d, All pass = %s\n",
                    death_type,
                    resolution,
                    rank,
                    all_pass
                )
                write_lines(results_str, ppc_results_file, append = TRUE)

                # Create mortality table
                mortality_table <- make_mortality_table(
                    merged_df |> mutate(type = "Race and ethnicity"),
                    target_state = "States With Bans"
                )

                # Save mortality table
                gtsave(
                    mortality_table,
                    sprintf(
                        "shiny_plots/tables/mortality_table_rank%d_%s_%s%s.png",
                        rank,
                        death_type,
                        resolution,
                        covid
                    ),
                    zoom = 4
                )

                # Create state table
                state_table <- make_state_table(
                    merged_df,
                    target_category = "total"
                )

                # Save state table
                gtsave(
                    state_table,
                    sprintf(
                        "shiny_plots/tables/state_table_rank%d_%s_%s%s.png",
                        rank,
                        death_type,
                        resolution,
                        covid
                    ),
                    zoom = 4
                )

                # Generate and save model fit plots for each state and category
                banned_states <- merged_df %>%
                    filter(exposure_code == 1 | state == "Unexposed States") %>%
                    pull(state) %>%
                    unique()

                for (state in banned_states) {
                    for (category in unique(merged_df$category)) {
                        # Create plot using make_all_te_plots
                        plts <- make_all_te_plots(
                            merged_df,
                            quantiles_df,
                            state_name = state,
                            category = category,
                            target = "deaths"
                        )

                        # Add title annotation
                        plts <- plts + plot_annotation(title = category)

                        # Save plot
                        save_plot_as_png(
                            plts,
                            sprintf(
                                "model_fit/model_fit_rank%d_%s_%s_%s_%s%s.png",
                                rank,
                                death_type,
                                resolution,
                                state,
                                category,
                                covid
                            ),
                            width = 10,
                            height = 10
                        )
                    }
                }
            }
        }
    }
}

# # Generate and save mortality rate explorer plots
# for (death_type in c(
#     "mat",
#     "mat_all",
#     "preg",
#     "preglate",
#     "pregearly",
#     "wra"
# )) {
#     for (show_covid in c(TRUE, FALSE)) {
#         for (race_display in c("facet", "single")) {
#             for (selected_race in c(
#                 "total",
#                 "nhwhite",
#                 "nhblack",
#                 "hispanic",
#                 "nhother"
#             )) {
#                 # Create plot
#                 p <- plot_mortality_rates(
#                     data = grouped_death_rate,
#                     death_type_filter = death_type,
#                     show_covid_separately = show_covid,
#                     start_date = "2016-01-01",
#                     facet_by_race = race_display == "facet",
#                     race_to_show = selected_race,
#                     exclude_states = NULL,
#                     plot_relative_rates = FALSE,
#                     normalize_before_date = "2020-01-01"
#                 )

#                 # Save plot
#                 filename <- sprintf(
#                     "mortality_rates/mortality_rates_%s_covid%s_race%s_%s.png",
#                     death_type,
#                     ifelse(show_covid, "show", "hide"),
#                     race_display,
#                     selected_race
#                 )
#                 save_plot_as_png(p, filename, width = 12, height = 10)
#             }
#         }
#     }
# }
