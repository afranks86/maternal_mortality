######################################################
## LOAD DATA
options(dplyr.summarise.inform = FALSE)

library(tidyverse)
library(tidybayes)
library(posterior)
library(jsonlite)
library(kableExtra)
library(gt)
source("plot_utilities.R")


for (prefix in c(
    # "late_preg_deaths",
    "preg_deaths"
    #    "mat_deaths"
    # "early_preg_deaths"
)) {
    for (model_rank in 1:6) {
        #, "quarterly_placebo_time_2020-04-01")
        for (suffix in c("quarterly")) {
            csv_prefix <- paste0("results/Poisson_", prefix)
            df <- read_csv(sprintf('results/df_%s_%s.csv', prefix, suffix))

            df <- df |> fill(exposed, .direction = "up")

            df <- df |>
                group_by(state) |>
                mutate(banned_state = 1 * any(exposed == 1)) |>
                ungroup()
            df$start_date <- df$date
            # df$end_date <- df$date + months(6) - days(1)
            df$end_date <- df$date + months(1) - days(1)

            # Rename columns that start with prefix to deaths_ + suffix
            df <- df |>
                rename_with(
                    ~ {
                        # Extract the suffix part after the prefix
                        suffix_part <- str_remove(., paste0("^", prefix, "_"))
                        # Create new name with "deaths_" + suffix
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

            #######################################################
            ## Load all datasets - looping over previx, model_rank and suffix

            all_samples <- read_csv(sprintf(
                "%s_%s_%s.csv",
                csv_prefix,
                model_rank,
                suffix
            ))

            agg_category_name = "total"

            merged_df <- merge_draws_and_data(
                df,
                all_samples,
                categories = c("total"), # categories=c("nhwhite", "nhblack", "hispanic", "nhother"),
                agg_category_name = agg_category_name,
                outcome_string = "deaths",
                denom_string = "births"
            )

            draws_mat <- all_samples %>% as_draws_matrix()

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

            #######################################################
            ## Make state table

            make_state_table <- function(
                merged_df,
                target_category = "total",
                target = "deaths",
                denom = "births",
                rate_normalizer = 100000,
                tab_caption = NULL,
                footnote_text = "",
                digits = 1
            ) {
                merged_df |> filter(category == target_category) -> merged_df

                # Determine death type based on prefix for proper labeling
                death_type_labels <- list(
                    mat_deaths = list(
                        singular = "maternal death",
                        plural = "maternal deaths",
                        title_case = "Maternal",
                        caption = "Expected difference in maternal deaths (count and rate) in states that banned abortion in months affected by bans."
                    ),
                    preg_deaths = list(
                        singular = "pregnancy-associated death",
                        plural = "pregnancy-associated deaths",
                        title_case = "Pregnancy-Associated",
                        caption = "Expected difference in pregnancy-associated deaths (count and rate) in states that banned abortion in months affected by bans."
                    ),
                    preg_no_mat = list(
                        singular = "pregnancy-associated (excluding maternal) death",
                        plural = "pregnancy-associated (excluding maternal) deaths",
                        title_case = "Pregnancy-Associated (Excluding Maternal)",
                        caption = "Expected difference in pregnancy-associated (excluding maternal) deaths (count and rate) in states that banned abortion in months affected by bans."
                    )
                )

                # Get the current prefix from the global environment
                current_prefix <- get("prefix", envir = parent.frame())

                # Get the appropriate labels
                labels <- death_type_labels[[current_prefix]]
                if (is.null(labels)) {
                    # Default to pregnancy-associated if prefix not recognized
                    labels <- death_type_labels[["preg_deaths"]]
                }

                # Use provided caption or generate based on prefix
                if (is.null(tab_caption)) {
                    tab_caption <- labels$caption #paste("Table %i.", labels$caption)
                }

                # Use provided footnote or default text
                if (is.null(footnote_text)) {
                    footnote_text <- paste0(
                        "Exposed months include January 2022 through December 2023 for Texas and January 2023 through December 2023 for other 13 states that banned abortion at 6 weeks or completely.<br>
            † Expected differences are computed using the Bayesian hierarchical panel data model described in the methods section. We report expected counts and rates as the observed count (or rate) minus the expected difference.
            The estimates for each row are computed using the same Bayesian model; note however that the expected percent change will not necessarily equal the percent change in expected(known as Jensen's inequality). States are ordered by expected percent change.<br>
            * Categories for which the 95% credible intervals of the expected difference excludes zero."
                    )
                }

                merged_df <- merged_df %>%
                    filter(exposure_code == 1) %>%
                    mutate(denom = births)

                # Create an aggregate entry for "States with bans (excl. Texas)" before grouping
                # First, identify all ban states except Texas
                all_banned_states <- unique(merged_df$state[
                    !merged_df$state %in%
                        c(
                            "States With Bans",
                            "States With Bans (excluding Texas)",
                            "States Without Bans"
                        )
                ])
                banned_states_excl_tx <- all_banned_states[
                    all_banned_states != "TX"
                ]

                ## Create a special merged_df_excl_tx for aggregation
                merged_df_all_banned <- merged_df %>%
                    filter(state %in% all_banned_states) %>%
                    mutate(state = "States With Bans")

                ## Create a special merged_df_excl_tx for aggregation
                merged_df_banned_excl_tx <- merged_df %>%
                    filter(state %in% banned_states_excl_tx) %>%
                    mutate(state = "States With Bans (excl. Texas)")

                merged_df <- merged_df |> filter(state %in% all_banned_states)

                # Add this to the original merged_df
                merged_df <- bind_rows(
                    merged_df,
                    merged_df_all_banned,
                    merged_df_banned_excl_tx
                )

                # Continue with the rest of the function
                table_df <- merged_df %>%
                    ungroup() %>%
                    ## Aggregate over time
                    group_by(state, .draw) %>%
                    summarize(
                        ypred = sum(ypred),
                        outcome = sum(ifelse(
                            is.na(.data[[target]]),
                            round(exp(mu)),
                            .data[[target]]
                        )),
                        treated = sum(exp(mu_treated)),
                        untreated = sum(exp(mu)),
                        denom = sum(denom, na.rm = TRUE),
                        treated_rate = treated / denom * rate_normalizer,
                        untreated_rate = untreated / denom * rate_normalizer,
                        outcome_rate = round(
                            outcome / denom * rate_normalizer,
                            digits
                        ),
                        outcome_diff = round(treated - untreated)
                    ) %>%
                    ungroup() %>%
                    ## Compute quantiles of effects
                    group_by(state) %>%
                    summarize(
                        ypred_mean = mean(ypred),
                        outcome = round(
                            mean(outcome),
                            digits = ifelse(mean(outcome < 1), 2, 0)
                        ),
                        outcome_diff_mean = round(
                            mean(outcome_diff),
                            digits = ifelse(mean(outcome < 1), 2, 0)
                        ),
                        outcome_diff_lower = round(quantile(
                            outcome_diff,
                            0.025
                        )),
                        outcome_diff_upper = round(quantile(
                            outcome_diff,
                            0.975
                        )),
                        outcome_rate = mean(outcome_rate),
                        ypred_lower = quantile(ypred, 0.025),
                        ypred_upper = quantile(ypred, 0.975),
                        treated_mean = mean(treated),
                        treated_lower = quantile(treated, 0.025),
                        treated_upper = quantile(treated, 0.975),
                        untreated_mean = mean(untreated),
                        untreated_lower = quantile(untreated, 0.025),
                        untreated_upper = quantile(untreated, 0.975),
                        treated_rate_mean = mean(treated_rate),
                        treated_rate_lower = quantile(treated_rate, 0.025),
                        treated_rate_upper = quantile(treated_rate, 0.975),
                        untreated_rate_mean = mean(untreated_rate),
                        untreated_rate_lower = quantile(untreated_rate, 0.025),
                        untreated_rate_upper = quantile(untreated_rate, 0.975),
                        causal_effect_diff_mean = mean(
                            treated_rate - untreated_rate
                        ),
                        causal_effect_diff_lower = quantile(
                            treated_rate - untreated_rate,
                            0.025
                        ),
                        causal_effect_diff_upper = quantile(
                            treated_rate - untreated_rate,
                            0.975
                        ),
                        causal_effect_ratio_mean = mean(
                            treated_rate / untreated_rate
                        ),
                        causal_effect_ratio_lower = quantile(
                            treated_rate / untreated_rate,
                            0.025
                        ),
                        causal_effect_ratio_upper = quantile(
                            treated_rate / untreated_rate,
                            0.975
                        ),
                        denom = mean(denom),
                        pval = 2 *
                            min(
                                mean(untreated_rate > treated_rate),
                                mean(untreated < treated)
                            )
                    )

                table_df <- table_df %>%
                    mutate(
                        # ypred_mean_rate = ypred_mean / years / (denom / rate_normalizer),
                        outcome_rate = round(outcome_rate, digits),
                        rate_diff = round(causal_effect_diff_mean, digits),
                        rate_diff_lower = round(
                            causal_effect_diff_lower,
                            digits
                        ),
                        rate_diff_upper = round(
                            causal_effect_diff_upper,
                            digits
                        ),
                        mult_change = causal_effect_ratio_mean,
                        mult_change_lower = causal_effect_ratio_lower,
                        mult_change_upper = causal_effect_ratio_upper,
                        # Only apply ordering to non-aggregated states
                        is_aggregate = state %in%
                            c(
                                "States With Bans",
                                "States With Bans (excl. Texas)"
                            ),
                        # Create a new column for ordering where aggregate rows get special values
                        order_value = ifelse(is_aggregate, Inf, mult_change),
                        # Convert to factor with order based on mult_change only for individual states
                        state = fct_reorder(
                            as.factor(state),
                            order_value,
                            .fun = median
                        ),
                        # Apply renaming for aggregated states
                        state = fct_recode(
                            state,
                            `States with bans` = "States With Bans",
                            `States with bans (excl. Texas)` = "States With Bans (excl. Texas)"
                        )
                    ) %>%
                    select(-is_aggregate, -order_value) %>% # Remove temporary columns
                    arrange(desc(state))

                table_df <- table_df %>%
                    mutate(untreated_mean = round(untreated_mean, 0)) %>%
                    mutate(
                        untreated_rate_mean = round(untreated_rate_mean, digits)
                    ) %>%
                    mutate(
                        death_counts_str = paste0(
                            outcome_diff_mean,
                            " (",
                            outcome_diff_lower,
                            ", ",
                            outcome_diff_upper,
                            ")"
                        )
                    ) %>%
                    mutate(
                        death_rate_abs_str = paste0(
                            rate_diff,
                            " (",
                            rate_diff_lower,
                            ", ",
                            rate_diff_upper,
                            ")"
                        )
                    ) %>%
                    mutate(
                        death_rate_pct_str = paste0(
                            round(100 * (mult_change - 1), digits),
                            " (",
                            round(100 * (mult_change_lower - 1), digits),
                            ", ",
                            round(100 * (mult_change_upper - 1), digits),
                            ")"
                        )
                    ) %>%
                    ungroup()

                pvals <- pval_rows <- table_df %>% pull(pval)
                pval_rows <- which(pvals < 0.05)
                table_df <- table_df %>%
                    mutate(
                        state = paste0(
                            state,
                            ifelse(rate_diff_lower >= 0, "*", "")
                        )
                    )

                # Ensure the aggregated rows appear at the top in their own specific order
                # Create row types: 1=States with bans, 2=States with bans (excl. TX), 3=individual states
                table_df <- table_df %>%
                    mutate(
                        is_aggregated = grepl("^States with bans", state),
                        row_order = case_when(
                            state %in%
                                c("States with bans*", "States with bans") ~
                                1,
                            state %in%
                                c(
                                    "States with bans (excl. Texas)*",
                                    "States with bans (excl. Texas)"
                                ) ~
                                2,
                            TRUE ~ 3
                        )
                    ) %>%
                    arrange(row_order, desc(mult_change)) %>% # Sort by row_order first, then by mult_change
                    select(-c(is_aggregated, row_order)) # Remove helper columns

                table_df %>%
                    select(
                        state,
                        denom,
                        outcome,
                        outcome_diff_mean,
                        outcome_rate,
                        rate_diff,
                        death_counts_str,
                        death_rate_abs_str,
                        death_rate_pct_str
                    ) %>%
                    mutate(
                        expected_outcome = outcome - outcome_diff_mean,
                        expected_rate = round(outcome_rate - rate_diff, digits)
                    ) %>%
                    mutate(outcome = as.character(outcome)) %>%
                    select(-c("outcome_diff_mean", "rate_diff")) %>%
                    gt(rowname_col = "state") |>
                    tab_header(
                        title = tab_caption
                    ) |>
                    ## ROW OPERATIONS
                    tab_row_group(
                        label = md("**Aggregated**"),
                        rows = str_detect(state, "States with bans")
                    ) |>
                    tab_row_group(
                        label = md("**States with bans**"),
                        rows = !str_detect(state, "States with bans")
                    ) |>
                    row_group_order(
                        groups = c(
                            md("**Aggregated**"),
                            md("**States with bans**")
                        )
                    ) |>

                    ### COLUMN OPERATIONS
                    tab_spanner(
                        label = paste(
                            labels$title_case,
                            "mortality rate (per 100,000 live births)"
                        ),
                        columns = c(
                            outcome_rate,
                            expected_rate,
                            death_rate_abs_str,
                            death_rate_pct_str
                        )
                    ) |>
                    tab_spanner(
                        label = labels$title_case,
                        columns = c(outcome, expected_outcome, death_counts_str)
                    ) |>
                    cols_label(
                        denom = "Births",
                        outcome_rate = "Observed",
                        expected_rate = "Expected",
                        death_rate_abs_str = html(
                            "Expected difference<br>(95% CI)"
                        ),
                        death_rate_pct_str = html(
                            "Expected percent change<br>(95% CI)"
                        ),
                        outcome = "Observed",
                        expected_outcome = "Expected",
                        death_counts_str = html(
                            "Expected difference<br>(95% CI)"
                        ),
                    ) -> table_df

                ## Styling
                table_df |>
                    tab_options(table.align = "left", heading.align = "left") |>
                    cols_align(align = "left") |>
                    cols_hide(state) |>
                    tab_options(table.font.size = 8) |>
                    opt_vertical_padding(scale = 0.5) |>
                    cols_width(
                        state ~ px(125),
                        death_rate_abs_str ~ px(100),
                        death_rate_pct_str ~ px(100),
                        outcome_rate ~ px(50),
                        death_counts_str ~ px(100),
                        outcome ~ px(50),
                        denom ~ px(50),
                        expected_outcome ~ px(50),
                        expected_rate ~ px(50)
                    ) -> table_df_final

                table_df_final |> tab_footnote(html(footnote_text))
            }

            total_states_tab <- make_state_table(merged_df)

            total_states_tab |>
                gtsave(
                    filename = sprintf(
                        "figs/tables/states_table_%s_%s_%i.png",
                        prefix,
                        suffix,
                        model_rank
                    ),
                    zoom = 4
                )

            #######################################################
            ## Model Fit Plots

            banned_states <- merged_df %>%
                filter(exposure_code == 1 | state == "States Without Bans") %>%
                pull(state) %>%
                unique()

            # banned_states <- sort(unique(merged_df$state))
            for (s in banned_states) {
                for (c in unique(merged_df$category)) {
                    plts <- make_all_te_plots(
                        merged_df,
                        quantiles_df,
                        state_name = s,
                        category = c,
                        target = "deaths"
                    )
                    print(plts + plot_annotation(title = c))
                    ggsave(
                        sprintf(
                            "figs/fits/%s_te_plot_%s_%i_%s_%s.png",
                            prefix,
                            s,
                            model_rank,
                            suffix,
                            c
                        ),
                        width = 10,
                        height = 10
                    )
                }
            }

            #######################################################
            # Posterior Predictive Checking

            categories <- c("total")
            ppc_states <- c(
                "States With Bans (excluding Texas)",
                "Texas",
                "States Without Bans"
            )
            ppc_states <- unique(merged_df$state[merged_df$banned_state == 1])
            ppc_outcome <- "deaths"

            rmse_res <- make_rmse_ppc_plot(
                merged_df %>% filter(state %in% ppc_states),
                categories = categories,
                outcome = "deaths"
            )

            ggsave(
                sprintf(
                    "figs/ppcs/%s_rmse_ppc_%i_%s.png",
                    prefix,
                    model_rank,
                    suffix
                ),
                plot = rmse_res$rmse_plt,
                width = 10,
                height = 10
            )
            rmse_pval_bool <- !(any(rmse_res$pval < 0.1) |
                any(rmse_res$pval > 0.9))

            abs_res <- make_abs_res_ppc_plot(
                merged_df %>% filter(state %in% ppc_states),
                categories = categories,
                outcome = "deaths"
            )
            ggsave(
                sprintf(
                    "figs/ppcs/%s_abs_res_ppc_%i_%s.png",
                    prefix,
                    model_rank,
                    suffix
                ),
                plot = abs_res$max_plt,
                width = 10,
                height = 10
            )

            abs_res_pval_bool <- !(any(abs_res$pval < 0.1) |
                any(abs_res$pval > 0.9))

            acf_ppc4 <- make_acf_ppc_plot(
                merged_df %>% filter(state %in% ppc_states),
                categories = categories,
                lag = 4,
                outcome = "deaths"
            )
            ggsave(
                sprintf(
                    "figs/ppcs/%s_acf_ppc_%i_%s.png",
                    prefix,
                    model_rank,
                    suffix
                ),
                plot = acf_ppc4$acf_plt,
                width = 10,
                height = 10
            )

            acf_ppc2 <- make_acf_ppc_plot(
                merged_df %>% filter(state %in% ppc_states),
                categories = categories,
                lag = 2,
                outcome = "deaths"
            )
            ggsave(
                sprintf(
                    "figs/ppcs/%s_acf_ppc2_%i_%s.png",
                    prefix,
                    model_rank,
                    suffix
                ),
                plot = acf_ppc2$acf_plt,
                width = 10,
                height = 10
            )

            acf_ppc1 <- make_acf_ppc_plot(
                merged_df %>% filter(state %in% ppc_states),
                categories = categories,
                lag = 1,
                outcome = "deaths"
            )
            ggsave(
                sprintf(
                    "figs/ppcs/%s_acf_ppc1_%i_%s.png",
                    prefix,
                    model_rank,
                    suffix
                ),
                plot = acf_ppc1$acf_plt,
                width = 10,
                height = 10
            )

            acf_pval_bool4 <- !(any(acf_ppc4$pval < 0.1) |
                any(acf_ppc4$pval > 0.9))
            acf_pval_bool2 <- !(any(acf_ppc2$pval < 0.1) |
                any(acf_ppc2$pval > 0.9))
            acf_pval_bool1 <- !(any(acf_ppc1$pval < 0.1) |
                any(acf_ppc1$pval > 0.9))

            uc_ppcs_obj <- make_unit_corr_ppc_plot(
                merged_df |> filter(!startsWith(state, "States")),
                categories = categories,
                outcome = "deaths"
            )

            ggsave(
                sprintf(
                    "figs/ppcs/%s_uc_ppc_%i_%s.png",
                    prefix,
                    model_rank,
                    suffix
                ),
                plot = uc_ppcs_obj$eval_plt,
                width = 10,
                height = 10
            )

            uc_pval_bool <- !(any(uc_ppcs_obj$pval < 0.1) |
                any(uc_ppcs_obj$pval > 0.9))

            all_pass <- all(
                rmse_pval_bool,
                abs_res_pval_bool,
                acf_pval_bool4,
                acf_pval_bool2,
                acf_pval_bool1,
                uc_pval_bool
            )

            ppc_results <- data.frame(
                prefix = prefix,
                model_rank = model_rank,
                suffix = suffix,
                rmse_pval_bool = rmse_pval_bool,
                abs_res_pval_bool = abs_res_pval_bool,
                acf_pval_bool4 = acf_pval_bool4,
                acf_pval_bool2 = acf_pval_bool2,
                acf_pval_bool1 = acf_pval_bool1,
                uc_pval_bool = uc_pval_bool,
                all_pass = all_pass
            )

            ppc_results |>
                write_csv(
                    file = sprintf(
                        "figs/ppcs/ppc_results_%s_%s_%i.csv",
                        prefix,
                        suffix,
                        model_rank
                    )
                )
        }
    }
}
