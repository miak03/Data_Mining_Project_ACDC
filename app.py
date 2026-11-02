"""Interactive Streamlit dashboard for exploring ACDC member attributes."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import load_config, load_data
from src.preprocessing import canonicalize_boolean_columns, canonicalize_categories


st.set_page_config(
    page_title="ACDC Member Explorer",
    page_icon="📊",
    layout="wide",
)

MISSING_LABEL = "(Missing)"
NUMERICAL_CHART_COLOR = "#8B7DBA"
CATEGORICAL_CHART_COLOR = "#39796b"
LIFETIME_VALUE_BINS = [
    float("-inf"),
    0,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    float("inf"),
]
LIFETIME_VALUE_BUCKET_LABELS = [
    "≤ €0",
    "> €0 – €500",
    "> €500 – €1,000",
    "> €1,000 – €2,500",
    "> €2,500 – €5,000",
    "> €5,000 – €10,000",
    "> €10,000 – €25,000",
    "> €25,000",
]
EXCLUDED_COLUMNS = {
    "member_id",
    "membership_start",
    "membership_end",
    "last_badge_swipe",
}


def clear_chart_selection() -> None:
    """Reset the keyed chart widget to clear its selected points."""
    st.session_state["category_chart_generation"] += 1


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows and normalize categories and booleans without changing numbers."""
    config = load_config()
    raw_data = load_data(config)
    duplicates_removed = int(raw_data.duplicated().sum())
    members = raw_data.drop_duplicates(keep="first").reset_index(drop=True)

    diagnostics = config["diagnostics"]
    members = canonicalize_categories(
        members,
        diagnostics.get("canonical_categories", {}),
        diagnostics.get("placeholder_tokens", []),
    )
    boolean_columns = list(diagnostics.get("boolean_columns", []))
    if "uses_print" not in boolean_columns:
        boolean_columns.append("uses_print")
    members = canonicalize_boolean_columns(members, boolean_columns)
    return members, duplicates_removed


def numeric_lifetime_values(values: pd.Series) -> pd.Series:
    """Parse lifetime values for summary statistics without changing displayed source values."""
    normalized = (
        values.astype("string")
        .str.replace(r"[^\d,.\-]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(normalized, errors="coerce")


def lifetime_value_buckets(values: pd.Series) -> pd.Series:
    """Group lifetime values into fixed euro ranges, retaining missing values."""
    numeric_values = numeric_lifetime_values(values)
    buckets = pd.cut(
        numeric_values,
        bins=LIFETIME_VALUE_BINS,
        labels=LIFETIME_VALUE_BUCKET_LABELS,
        include_lowest=True,
    )
    if numeric_values.isna().any():
        buckets = buckets.cat.add_categories([MISSING_LABEL]).fillna(MISSING_LABEL)
    return buckets


def category_options(data: pd.DataFrame, column: str) -> list[str]:
    """Return display values for a categorical field, including missing values."""
    values = sorted(data[column].dropna().astype(str).unique().tolist())
    if data[column].isna().any():
        values.append(MISSING_LABEL)
    return values


def filter_category(
    data: pd.DataFrame,
    column: str,
    selected_values: list[str],
) -> pd.DataFrame:
    """Filter a category while allowing missing values to be selected explicitly."""
    if not selected_values:
        return data

    mask = data[column].astype("string").isin(
        [value for value in selected_values if value != MISSING_LABEL]
    )
    if MISSING_LABEL in selected_values:
        mask |= data[column].isna()
    return data.loc[mask]


def display_category(value: object) -> str:
    return MISSING_LABEL if pd.isna(value) else str(value)


def selected_plotly_points(chart_key: str) -> list[dict]:
    """Return points selected in an interactive Plotly chart."""
    chart_state = st.session_state.get(chart_key, {})
    selection = chart_state.get("selection", {})
    return selection.get("points", [])


@st.cache_data
def csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")


st.session_state.setdefault("category_chart_generation", 0)

try:
    members, duplicates_removed = load_dashboard_data()
except (FileNotFoundError, KeyError, OSError, pd.errors.ParserError) as error:
    st.error(f"Could not load the configured member dataset: {error}")
    st.stop()

numeric_columns = [
    column
    for column in members.select_dtypes(include="number").columns
    if column not in EXCLUDED_COLUMNS
]
categorical_columns = [
    column
    for column in members.select_dtypes(include=["object", "category", "bool"]).columns
    if column not in EXCLUDED_COLUMNS and column != "lifetime_value"
]
breakdown_columns = categorical_columns + (
    ["lifetime_value"] if "lifetime_value" in members.columns else []
)

st.markdown(
    """
    <div style="display:flex;align-items:center;gap:1.1rem;padding:1.1rem 0 .4rem">
      <div style="flex:1">
        <div style="color:#7663a6;font-size:.8rem;font-weight:700;letter-spacing:.14em">
          AMAZING CREATIVE DESK COWORKING
        </div>
        <h1 style="margin:.25rem 0">Member Explorer</h1>
        <div style="color:#68747b">Explore who our members are, how they use the space,
        and how membership patterns differ across segments.</div>
      </div>
      <svg viewBox="0 0 100 100" width="92" height="92" role="img"
           aria-label="ACDC logo">
        <circle cx="50" cy="50" r="46" fill="#F3F0F8" stroke="#7663A6"
                stroke-width="3"/>
        <g fill="#51456F" font-family="Arial, sans-serif" font-size="22"
           font-weight="700" text-anchor="middle">
          <text x="50" y="29">C</text>
          <text x="29" y="57">A</text>
          <text x="71" y="57">D</text>
          <text x="50" y="85">C</text>
        </g>
      </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header(":material/filter_alt: Filter members")
    st.caption("Filters apply to every chart and table.")

    filtered = members
    if "membership_start" in members:
        membership_start_dates = pd.to_datetime(
            members["membership_start"],
            errors="coerce",
        )
        valid_start_dates = membership_start_dates.dropna()
        if valid_start_dates.empty:
            st.info("No valid membership start dates are available to filter.")
        else:
            start_date = st.date_input(
                "Start date",
                value=valid_start_dates.min().date(),
                min_value=valid_start_dates.min().date(),
                max_value=valid_start_dates.max().date(),
                key="filter_start_date",
            )
            end_date = st.date_input(
                "End date",
                value=valid_start_dates.max().date(),
                min_value=valid_start_dates.min().date(),
                max_value=valid_start_dates.max().date(),
                key="filter_end_date",
            )
            include_missing_start_dates = st.checkbox(
                "Include members without a valid start date",
                value=True,
                key="filter_missing_start_dates",
            )
            if start_date > end_date:
                st.error("Start date must be on or before end date.")
                filtered = filtered.iloc[0:0]
            else:
                date_mask = membership_start_dates.loc[filtered.index].between(
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date),
                )
                if include_missing_start_dates:
                    date_mask |= membership_start_dates.loc[filtered.index].isna()
                filtered = filtered.loc[date_mask]

    for column, title in (
        ("country", "Country"),
        ("industry", "Industry"),
        ("plan_tier", "Plan tier"),
        ("gender", "Gender"),
    ):
        if column in categorical_columns:
            selected = st.multiselect(
                title,
                options=category_options(members, column),
                key=f"filter_{column}",
                placeholder="All",
            )
            filtered = filter_category(filtered, column, selected)

    if "is_team_account" in categorical_columns:
        team_options = category_options(members, "is_team_account")
        selected_teams = st.multiselect(
            "Account type",
            options=team_options,
            format_func=lambda value: {
                "True": "Team",
                "False": "Individual",
                MISSING_LABEL: MISSING_LABEL,
            }.get(value, value),
            placeholder="All",
        )
        filtered = filter_category(filtered, "is_team_account", selected_teams)

    if "age" in numeric_columns and filtered["age"].notna().any():
        age_min = float(filtered["age"].min())
        age_max = float(filtered["age"].max())
        if age_min < age_max:
            age_range = st.slider(
                "Age range",
                min_value=int(age_min),
                max_value=int(age_max),
                value=(int(age_min), int(age_max)),
            )
            include_missing_age = st.checkbox(
                "Include members with no recorded age",
                value=True,
            )
            age_mask = filtered["age"].between(*age_range)
            if include_missing_age:
                age_mask |= filtered["age"].isna()
            filtered = filtered.loc[age_mask]

default_breakdown_column = (
    "plan_tier" if "plan_tier" in breakdown_columns else breakdown_columns[0]
)
breakdown_column = st.session_state.get("category_column", default_breakdown_column)
category_chart_key = (
    f"category_distribution_{st.session_state['category_chart_generation']}"
)
selected_categories = [
    point["y"]
    for point in selected_plotly_points(category_chart_key)
    if "y" in point
]
if breakdown_column == "lifetime_value":
    if selected_categories:
        selected_buckets = lifetime_value_buckets(filtered["lifetime_value"])
        filtered = filtered.loc[selected_buckets.isin(selected_categories)]
elif selected_categories:
    filtered = filter_category(filtered, breakdown_column, selected_categories)

st.markdown("### At a glance")
metric_columns = st.columns(4)
metric_columns[0].metric("Members", f"{len(filtered):,}")
if "age" in filtered:
    metric_columns[1].metric(
        "Median age",
        f"{filtered['age'].median():.0f}" if filtered["age"].notna().any() else "—",
    )
if "avg_weekly_visits" in filtered:
    metric_columns[2].metric(
        "Avg. weekly visits",
        (
            f"{filtered['avg_weekly_visits'].mean():.1f}"
            if filtered["avg_weekly_visits"].notna().any()
            else "—"
        ),
    )
if "lifetime_value" in filtered:
    median_value = numeric_lifetime_values(filtered["lifetime_value"]).median()
    metric_columns[3].metric(
        "Median lifetime value",
        f"€{median_value:,.0f}" if pd.notna(median_value) else "—",
    )

if filtered.empty:
    st.info("No members match these filters. Adjust or clear a filter to continue.")
    st.stop()

overview_tab, data_tab = st.tabs(["Overview", "Filtered records"])

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Numerical distribution")
        histogram_column = st.selectbox(
            "Choose a numeric attribute",
            numeric_columns,
            index=numeric_columns.index("age") if "age" in numeric_columns else 0,
            key="histogram_column",
        )
        histogram = px.histogram(
            filtered,
            x=histogram_column,
            nbins=30,
            color_discrete_sequence=[NUMERICAL_CHART_COLOR],
            labels={histogram_column: histogram_column.replace("_", " ").title()},
        )
        histogram.update_layout(
            xaxis_title=histogram_column.replace("_", " ").title(),
            yaxis_title="Members",
            showlegend=False,
            bargap=0.08,
        )
        st.plotly_chart(histogram, width="stretch")

    with right:
        st.subheader("Categorical/Boolean distribution")
        breakdown_column = st.selectbox(
            "Break down members by",
            breakdown_columns,
            index=(
                breakdown_columns.index("plan_tier")
                if "plan_tier" in breakdown_columns
                else 0
            ),
            key="category_column",
            on_change=clear_chart_selection,
        )
        if breakdown_column == "lifetime_value":
            bucket_labels = lifetime_value_buckets(filtered["lifetime_value"])
            category_order = list(bucket_labels.cat.categories)
            counts = (
                bucket_labels.value_counts(sort=False)
                .rename_axis("Lifetime value range")
                .reset_index(name="Members")
            )
            category_column = "Lifetime value range"
        else:
            category_order = None
            category_column = breakdown_column
            counts = (
                filtered[category_column]
                .map(display_category)
                .value_counts()
                .rename_axis(category_column)
                .reset_index(name="Members")
            )
            counts = counts.head(15)
        if category_order is None:
            counts = counts.sort_values("Members")
        bar_chart = px.bar(
            counts,
            x="Members",
            y=category_column,
            orientation="h",
            color_discrete_sequence=[CATEGORICAL_CHART_COLOR],
            labels={category_column: category_column.replace("_", " ").title()},
        )
        bar_chart.update_layout(
            yaxis_title=None,
            xaxis_title="Members",
            showlegend=False,
            clickmode="event+select",
        )
        if category_order is not None:
            bar_chart.update_yaxes(
                categoryorder="array",
                categoryarray=category_order,
            )
        st.plotly_chart(
            bar_chart,
            width="stretch",
            key=category_chart_key,
            on_select="rerun",
            selection_mode="points",
        )
        if selected_categories:
            st.caption(f"Focused on: {', '.join(selected_categories)}")
            st.button(
                "Clear chart selection",
                key="clear_chart_selection",
                on_click=clear_chart_selection,
            )

    st.caption(
        "Category charts show up to the 15 most common values. "
        "Click a category to focus all charts and records on that group. "
        "Sidebar filters also apply."
    )

with data_tab:
    st.subheader("Filtered member records")
    st.caption(f"Showing {len(filtered):,} rows after applying the sidebar filters.")
    st.dataframe(filtered, width="stretch", hide_index=True)
    st.download_button(
        "Download filtered records as CSV",
        data=csv_bytes(filtered),
        file_name="acdc_filtered_members.csv",
        mime="text/csv",
    )
