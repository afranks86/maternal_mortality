make_mortality_table <- function(
    merged_df,
    target_state = "Texas",
    target = "deaths",
    denom = "births",
    rate_normalizer = 100000,
    tab_caption = NULL,
    digits = 1
) {
    # Determine death type based on prefix for proper labeling
    death_type_labels <- list(
        mat_deaths = list(
            singular = "maternal death",
            plural = "maternal deaths",
            title_case = "Maternal",
            caption = "Expected difference† in maternal deaths (count and rate) in states that banned abortion in months affected by bans."
        ),
        preg_deaths = list(
            singular = "pregnancy-associated death",
            plural = "pregnancy-associated deaths",
            title_case = "Pregnancy-Associated",
            caption = "Expected difference† in pregnancy-associated deaths (count and rate) in states that banned abortion in months affected by bans."
        ),
        preg_no_mat = list(
            singular = "pregnancy-associated (excluding maternal) death",
            plural = "pregnancy-associated (excluding maternal) deaths",
            title_case = "Pregnancy-Associated (Excluding Maternal)",
            caption = "Expected difference† in pregnancy-associated (excluding maternal) deaths (count and rate) in states that banned abortion in months affected by bans."
        )
    )

    # Get the current prefix from the global environment
    current_prefix <- "preg_deaths" # Default to preg_deaths

    # Get the appropriate labels
    labels <- death_type_labels[[current_prefix]]
    if (is.null(labels)) {
        # Default to pregnancy-associated if prefix not recognized
        labels <- death_type_labels[["preg_deaths"]]
    }

    # Use provided caption or generate based on prefix
    if (is.null(tab_caption)) {
        tab_caption <- paste("Table 1.", labels$caption)
    }

    if (target_state == "States With Bans") {
        merged_df <- merged_df %>%
            filter(exposure_code == 1) %>%
            mutate(denom = births)

        # Create an aggregate entry for "States with bans (excl. Texas)" before grouping
        all_banned_states <- unique(merged_df$state[
            !(merged_df$state %in%
                c(
                    "States With Bans",
                    "States With Bans (excluding Texas)",
                    "Unexposed States"
                ))
        ])
        banned_states_excl_tx <- all_banned_states[all_banned_states != "TX"]

        merged_df_all_banned <- merged_df %>%
            filter(state %in% all_banned_states) %>%
            mutate(state = "States With Bans")

        merged_df_banned_excl_tx <- merged_df %>%
            filter(state %in% banned_states_excl_tx) %>%
            mutate(state = "States With Bans (excl. Texas)")

        merged_df <- merged_df |> filter(state %in% all_banned_states)

        merged_df <- bind_rows(
            merged_df,
            merged_df_all_banned,
            merged_df_banned_excl_tx
        )

        merged_df <- merged_df |>
            filter(state == "States With Bans") |>
            mutate(type = ifelse(category == "total", "total", "race"))
    } else {
        merged_df <- merged_df %>%
            filter(state == target_state, exposure_code == 1) %>%
            mutate(denom = .data[[denom]])
    }

    table_df <- merged_df %>%
        ungroup() %>%
        group_by(type, category, .draw) %>%
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
            outcome_rate = round(outcome / denom * rate_normalizer, digits),
            outcome_diff = round(treated - untreated)
        ) %>%
        ungroup() %>%
        group_by(type, category) %>%
        summarize(
            ypred_mean = mean(ypred),
            outcome = round(
                mean(outcome),
                digits = ifelse(mean(outcome < 10), digits, 0)
            ),
            outcome_diff_mean = round(
                mean(outcome_diff),
                digits = ifelse(mean(outcome < 1), digits, 0)
            ),
            outcome_diff_lower = round(quantile(outcome_diff, 0.025)),
            outcome_diff_upper = round(quantile(outcome_diff, 0.975)),
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
            causal_effect_diff_mean = mean(treated_rate - untreated_rate),
            causal_effect_diff_lower = quantile(
                treated_rate - untreated_rate,
                0.025
            ),
            causal_effect_diff_upper = quantile(
                treated_rate - untreated_rate,
                0.975
            ),
            causal_effect_ratio_mean = mean(treated_rate / untreated_rate),
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
            outcome_rate = round(outcome_rate, digits),
            rate_diff = round(causal_effect_diff_mean, digits),
            rate_diff_lower = round(causal_effect_diff_lower, digits),
            rate_diff_upper = round(causal_effect_diff_upper, digits),
            mult_change = causal_effect_ratio_mean,
            mult_change_lower = causal_effect_ratio_lower,
            mult_change_upper = causal_effect_ratio_upper
        )

    table_df <- table_df %>%
        mutate(untreated_mean = round(untreated_mean, 0)) %>%
        mutate(untreated_rate_mean = round(untreated_rate_mean, digits)) %>%
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
        ungroup() %>%
        filter((type != "total" & category != "Total") | type == "total")

    pvals <- pval_rows <- table_df %>% pull(pval)
    pval_rows <- which(pvals < 0.05)
    table_df <- table_df %>%
        mutate(
            category = paste0(category, ifelse(rate_diff_lower > 0, "*", ""))
        )

    table_df %>%
        select(
            type,
            category,
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
            expected_rate = outcome_rate - rate_diff
        ) %>%
        mutate(outcome = as.character(outcome)) %>%
        select(-c("outcome_diff_mean", "rate_diff")) %>%
        mutate(outcome = ifelse(as.numeric(outcome) < 10, NA, outcome)) %>%
        mutate(outcome_rate = ifelse(is.na(outcome), NA, outcome_rate)) %>%
        gt(rowname_col = "category") |>
        tab_header(
            title = tab_caption
        ) |>
        tab_row_group(
            label = "Race and ethnicity",
            rows = type == "race"
        ) |>
        row_group_order(groups = c(NA, "Race and ethnicity")) |>
        tab_spanner(
            label = "Mortality rate (per 100,000 live births)",
            columns = c(
                outcome_rate,
                expected_rate,
                death_rate_abs_str,
                death_rate_pct_str
            )
        ) |>
        tab_spanner(
            label = "Deaths",
            columns = c(outcome, expected_outcome, death_counts_str)
        ) |>
        cols_label(
            denom = "Births",
            outcome_rate = "Observed",
            expected_rate = "Expected",
            death_rate_abs_str = html("Expected difference<br>(95% CI)"),
            death_rate_pct_str = html("Expected percent change<br>(95% CI)"),
            outcome = "Observed",
            expected_outcome = "Expected",
            death_counts_str = html("Expected difference<br>(95% CI)"),
            category = ""
        ) |>
        tab_stub_indent(
            rows = category != "Total",
            indent = 5
        ) |>
        tab_options(table.align = "left", heading.align = "left") |>
        cols_align(align = "left") |>
        cols_hide(c(type, category)) |>
        tab_options(table.font.size = 8) |>
        opt_vertical_padding(scale = 0.5) |>
        cols_width(
            category ~ px(125),
            death_rate_abs_str ~ px(100),
            death_rate_pct_str ~ px(100),
            outcome_rate ~ px(50),
            death_counts_str ~ px(100),
            outcome ~ px(50),
            denom ~ px(50),
            expected_outcome ~ px(50),
            expected_rate ~ px(50)
        ) |>
        sub_missing()
}

make_state_table <- function(
    merged_df,
    target_category = "total",
    target = "deaths",
    denom = "births",
    rate_normalizer = 100000,
    tab_caption = NULL,
    footnote_text = NULL
) {
    merged_df |> filter(category == target_category) -> merged_df

    # Determine death type based on prefix for proper labeling
    death_type_labels <- list(
        mat_deaths = list(
            singular = "maternal death",
            plural = "maternal deaths",
            title_case = "Maternal",
            caption = "Expected difference† in maternal deaths (count and rate) in states that banned abortion in months affected by bans."
        ),
        preg_deaths = list(
            singular = "pregnancy-associated death",
            plural = "pregnancy-associated deaths",
            title_case = "Pregnancy-Associated",
            caption = "Expected difference† in pregnancy-associated deaths (count and rate) in states that banned abortion in months affected by bans."
        ),
        preg_no_mat = list(
            singular = "pregnancy-associated (excluding maternal) death",
            plural = "pregnancy-associated (excluding maternal) deaths",
            title_case = "Pregnancy-Associated (Excluding Maternal)",
            caption = "Expected difference† in pregnancy-associated (excluding maternal) deaths (count and rate) in states that banned abortion in months affected by bans."
        )
    )

    # Get the current prefix from the global environment
    current_prefix <- "preg_deaths" # Default to preg_deaths

    # Get the appropriate labels
    labels <- death_type_labels[[current_prefix]]
    if (is.null(labels)) {
        # Default to pregnancy-associated if prefix not recognized
        labels <- death_type_labels[["preg_deaths"]]
    }

    # Use provided caption or generate based on prefix
    if (is.null(tab_caption)) {
        tab_caption <- paste("Table %i.", labels$caption)
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
    all_banned_states <- unique(merged_df$state[
        merged_df$state != "States With Bans" &
            !merged_df$state %in% c("Unexposed States")
    ])
    banned_states_excl_tx <- all_banned_states[all_banned_states != "TX"]

    merged_df_all_banned <- merged_df %>%
        filter(state %in% all_banned_states) %>%
        mutate(state = "States With Bans")

    merged_df_banned_excl_tx <- merged_df %>%
        filter(state %in% banned_states_excl_tx) %>%
        mutate(state = "States With Bans (excl. Texas)")

    merged_df <- merged_df |> filter(state %in% all_banned_states)

    merged_df <- bind_rows(
        merged_df,
        merged_df_all_banned,
        merged_df_banned_excl_tx
    )

    table_df <- merged_df %>%
        ungroup() %>%
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
            outcome_rate = round(outcome / denom * rate_normalizer, 2),
            outcome_diff = round(treated - untreated)
        ) %>%
        ungroup() %>%
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
            outcome_diff_lower = round(quantile(outcome_diff, 0.025)),
            outcome_diff_upper = round(quantile(outcome_diff, 0.975)),
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
            causal_effect_diff_mean = mean(treated_rate - untreated_rate),
            causal_effect_diff_lower = quantile(
                treated_rate - untreated_rate,
                0.025
            ),
            causal_effect_diff_upper = quantile(
                treated_rate - untreated_rate,
                0.975
            ),
            causal_effect_ratio_mean = mean(treated_rate / untreated_rate),
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
            outcome_rate = round(outcome_rate, 2),
            rate_diff = round(causal_effect_diff_mean, 2),
            rate_diff_lower = round(causal_effect_diff_lower, 2),
            rate_diff_upper = round(causal_effect_diff_upper, 2),
            mult_change = causal_effect_ratio_mean,
            mult_change_lower = causal_effect_ratio_lower,
            mult_change_upper = causal_effect_ratio_upper,
            is_aggregate = state %in%
                c("States With Bans", "States With Bans (excl. Texas)"),
            order_value = ifelse(is_aggregate, Inf, mult_change),
            state = fct_reorder(as.factor(state), order_value, .fun = median),
            state = fct_recode(
                state,
                `States with bans` = "States With Bans",
                `States with bans (excl. Texas)` = "States With Bans (excl. Texas)"
            )
        ) %>%
        select(-is_aggregate, -order_value) %>%
        arrange(desc(state))

    table_df <- table_df %>%
        mutate(untreated_mean = round(untreated_mean, 0)) %>%
        mutate(untreated_rate_mean = round(untreated_rate_mean, 2)) %>%
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
                round(100 * (mult_change - 1), 2),
                " (",
                round(100 * (mult_change_lower - 1), 2),
                ", ",
                round(100 * (mult_change_upper - 1), 2),
                ")"
            )
        ) %>%
        ungroup()

    pvals <- pval_rows <- table_df %>% pull(pval)
    pval_rows <- which(pvals < 0.05)
    table_df <- table_df %>%
        mutate(state = paste0(state, ifelse(rate_diff_lower >= 0, "*", "")))

    table_df <- table_df %>%
        mutate(
            is_aggregated = grepl("^States with bans", state),
            row_order = case_when(
                state %in% c("States with bans*", "States with bans") ~ 1,
                state %in%
                    c(
                        "States with bans (excl. Texas)*",
                        "States with bans (excl. Texas)"
                    ) ~
                    2,
                TRUE ~ 3
            )
        ) %>%
        arrange(row_order, desc(mult_change)) %>%
        select(-c(is_aggregated, row_order))

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
            expected_rate = outcome_rate - rate_diff
        ) %>%
        mutate(outcome = as.character(outcome)) %>%
        select(-c("outcome_diff_mean", "rate_diff")) %>%
        gt(rowname_col = "state") |>
        tab_header(
            title = tab_caption
        ) |>
        tab_row_group(
            label = md("**Aggregated**"),
            rows = str_detect(state, "States with bans")
        ) |>
        tab_row_group(
            label = md("**States with bans**"),
            rows = !str_detect(state, "States with bans")
        ) |>
        row_group_order(
            groups = c(md("**Aggregated**"), md("**States with bans**"))
        ) |>
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
            death_rate_abs_str = html("Expected difference<br>(95% CI)"),
            death_rate_pct_str = html("Expected percent change<br>(95% CI)"),
            outcome = "Observed",
            expected_outcome = "Expected",
            death_counts_str = html("Expected difference<br>(95% CI)"),
        ) |>
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
        ) |>
        tab_footnote(html(footnote_text))
}
