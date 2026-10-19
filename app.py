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
EXCLUDED_COLUMNS = {
    "member_id",
    "membership_start",
    "membership_end",
    "last_badge_swipe",
}


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, int, dict]:
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
    return members, duplicates_removed, diagnostics


def numeric_lifetime_values(values: pd.Series) -> pd.Series:
    """Parse lifetime values for summary statistics without changing displayed source values."""
    normalized = (
        values.astype("string")
        .str.replace(r"[^\d,.\-]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(normalized, errors="coerce")


def lifetime_value_buckets(values: pd.Series) -> pd.Series:
    """Group parseable lifetime values into quantile ranges, retaining missing values."""
    numeric_values = numeric_lifetime_values(values)
    labels = pd.Series(MISSING_LABEL, index=values.index, dtype="object")
    unique_values = numeric_values.nunique()
    if unique_values == 0:
        return labels

    buckets = pd.qcut(
        numeric_values.dropna(),
        q=min(10, unique_values),
        duplicates="drop",
    )
    labels.loc[buckets.index] = buckets.map(
        lambda interval: f"€{interval.left:,.0f} – €{interval.right:,.0f}"
    ).astype(str)
    return labels


def build_quality_report(
    members: pd.DataFrame,
    diagnostics: dict,
) -> pd.DataFrame:
    """Describe numeric and date anomalies while preserving their original values."""
    issue_rows: list[dict[str, object]] = []
    id_column = diagnostics.get("id_column")

    def add_issues(column: str, mask: pd.Series, issue: str) -> None:
        if not mask.any():
            return
        for index, value in members.loc[mask, column].items():
            row: dict[str, object] = {
                "column": column,
                "observed_value": str(value),
                "issue": issue,
            }
            if id_column and id_column in members.columns:
                row[id_column] = members.at[index, id_column]
            issue_rows.append(row)

    for column, bounds in diagnostics.get("validity_rules", {}).items():
        if column not in members.columns:
            continue
        numeric = (
            numeric_lifetime_values(members[column])
            if column == "lifetime_value"
            else pd.to_numeric(members[column], errors="coerce")
        )
        invalid = numeric.notna() & (
            (numeric < bounds.get("min", float("-inf")))
            | (numeric > bounds.get("max", float("inf")))
        )
        add_issues(
            column,
            invalid,
            f"Outside configured range [{bounds.get('min', '−∞')}, "
            f"{bounds.get('max', '∞')}]",
        )
        not_numeric = members[column].notna() & numeric.isna()
        add_issues(column, not_numeric, "Non-empty value could not be parsed as a number")

    for rule in diagnostics.get("date_order_rules", []):
        start_column = rule["start_column"]
        end_column = rule["end_column"]
        if start_column not in members.columns or end_column not in members.columns:
            continue
        start_dates = pd.to_datetime(members[start_column], errors="coerce")
        end_dates = pd.to_datetime(members[end_column], errors="coerce")
        ignored_end_dates = pd.to_datetime(
            rule.get("ignore_end_values", []),
            errors="coerce",
        )
        invalid_order = (
            start_dates.notna()
            & end_dates.notna()
            & end_dates.lt(start_dates)
            & ~end_dates.isin(ignored_end_dates)
        )
        for column in (start_column, end_column):
            malformed = members[column].notna() & pd.to_datetime(
                members[column],
                errors="coerce",
            ).isna()
            add_issues(column, malformed, "Non-empty value could not be parsed as a date")
        add_issues(end_column, invalid_order, f"End date is before {start_column}")

        for sentinel in rule.get("ignore_end_values", []):
            sentinel_mask = members[end_column].astype("string").eq(str(sentinel))
            add_issues(
                end_column,
                sentinel_mask,
                "Configured sentinel date (retained; often represents no end date)",
            )

    columns = ["column", "observed_value", "issue"]
    if id_column and id_column in members.columns:
        columns.insert(0, id_column)
    return pd.DataFrame(issue_rows, columns=columns)


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


@st.cache_data
def csv_bytes(data: pd.DataFrame) -> bytes:
    return data.to_csv(index=False).encode("utf-8")


try:
    members, duplicates_removed, diagnostics = load_dashboard_data()
except (FileNotFoundError, KeyError, OSError, pd.errors.ParserError) as error:
    st.error(f"Could not load the configured member dataset: {error}")
    st.stop()

quality_report = build_quality_report(members, diagnostics)
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
    <div style="padding: 1.1rem 0 0.4rem">
      <div style="color:#55736b;font-size:.8rem;font-weight:700;letter-spacing:.14em">
        AMAZING CREATIVE DESK COWORKING
      </div>
      <h1 style="margin:.25rem 0">Member Explorer</h1>
      <div style="color:#68747b">Explore who our members are, how they use the space,
      and how membership patterns differ across segments.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Filter members")
    st.caption("Filters apply to every chart and table.")

    filtered = members
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

    if "membership_end" in members:
        membership_status = st.selectbox(
            "Membership status",
            ["All", "Active", "Former"],
        )
        if membership_status != "All":
            active_members = members["membership_end"].astype(str).eq("1970-01-01")
            filtered = filtered.loc[
                active_members.loc[filtered.index]
                if membership_status == "Active"
                else ~active_members.loc[filtered.index]
            ]

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

overview_tab, explore_tab, data_tab, quality_tab = st.tabs(
    ["Overview", "Explore a segment", "Filtered records", "Data quality"]
)

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Attribute distribution")
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
            color_discrete_sequence=["#39796b"],
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
        st.subheader("Member mix")
        breakdown_column = st.selectbox(
            "Break down members by",
            breakdown_columns,
            index=(
                breakdown_columns.index("plan_tier")
                if "plan_tier" in breakdown_columns
                else 0
            ),
            key="category_column",
        )
        if breakdown_column == "lifetime_value":
            bucket_labels = lifetime_value_buckets(filtered["lifetime_value"])
            counts = (
                bucket_labels.value_counts(sort=False)
                .rename_axis("Lifetime value range")
                .reset_index(name="Members")
            )
            category_column = "Lifetime value range"
        else:
            category_column = breakdown_column
            counts = (
                filtered[category_column]
                .map(display_category)
                .value_counts()
                .rename_axis(category_column)
                .reset_index(name="Members")
            )
            counts = counts.head(15)
        counts = counts.sort_values("Members")
        bar_chart = px.bar(
            counts,
            x="Members",
            y=category_column,
            orientation="h",
            color_discrete_sequence=["#e0a64d"],
            labels={category_column: category_column.replace("_", " ").title()},
        )
        bar_chart.update_layout(yaxis_title=None, xaxis_title="Members")
        st.plotly_chart(bar_chart, width="stretch")

    st.caption(
        "Category charts show up to the 15 most common values. "
        "All charts reflect the filters selected in the sidebar."
    )

with explore_tab:
    st.subheader("Compare and drill into member segments")
    st.caption("Choose a segment below to compare its profile with the filtered population.")
    segment_column = st.selectbox(
        "Segment by",
        categorical_columns,
        index=(
            categorical_columns.index("plan_tier")
            if "plan_tier" in categorical_columns
            else 0
        ),
        key="segment_column",
    )
    segment_value = st.selectbox(
        "Choose a segment",
        category_options(filtered, segment_column),
        key="segment_value",
    )
    segment_members = filter_category(filtered, segment_column, [segment_value])

    segment_metrics = st.columns(3)
    segment_metrics[0].metric("Members in segment", f"{len(segment_members):,}")
    if "avg_weekly_visits" in segment_members:
        visits = segment_members["avg_weekly_visits"].mean()
        segment_metrics[1].metric(
            "Avg. weekly visits",
            f"{visits:.1f}" if pd.notna(visits) else "—",
        )
    if "lifetime_value" in segment_members:
        value = numeric_lifetime_values(segment_members["lifetime_value"]).median()
        segment_metrics[2].metric(
            "Median lifetime value",
            f"€{value:,.0f}" if pd.notna(value) else "—",
        )

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.markdown("**Age profile**")
        if "age" in segment_members and segment_members["age"].notna().any():
            age_chart = px.histogram(
                segment_members,
                x="age",
                nbins=25,
                color_discrete_sequence=["#39796b"],
                labels={"age": "Age"},
            )
            age_chart.update_layout(xaxis_title="Age", yaxis_title="Members")
            st.plotly_chart(age_chart, width="stretch")
        else:
            st.info("Age data is not available for this segment.")
    with detail_right:
        st.markdown("**Visit frequency**")
        if (
            "avg_weekly_visits" in segment_members
            and segment_members["avg_weekly_visits"].notna().any()
        ):
            visits_chart = px.histogram(
                segment_members,
                x="avg_weekly_visits",
                nbins=25,
                color_discrete_sequence=["#e0a64d"],
                labels={"avg_weekly_visits": "Average weekly visits"},
            )
            visits_chart.update_layout(
                xaxis_title="Average weekly visits",
                yaxis_title="Members",
            )
            st.plotly_chart(visits_chart, width="stretch")
        else:
            st.info("Visit data is not available for this segment.")

    st.markdown("**Segment members**")
    st.dataframe(segment_members, width="stretch", hide_index=True)
    st.download_button(
        "Download segment as CSV",
        data=csv_bytes(segment_members),
        file_name="acdc_member_segment.csv",
        mime="text/csv",
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

with quality_tab:
    st.subheader("Data quality and unusual values")
    st.caption(
        "Values outside configured ranges, unusual dates, and unparsable values are "
        "reported here, not corrected or removed. The data table and charts retain "
        "the source numeric values."
    )

    if diagnostics.get("id_column") in filtered.columns:
        filtered_ids = set(filtered[diagnostics["id_column"]])
        filtered_quality = quality_report.loc[
            quality_report[diagnostics["id_column"]].isin(filtered_ids)
        ]
    else:
        filtered_quality = quality_report

    quality_metrics = st.columns(4)
    range_issue_count = int(
        filtered_quality["issue"].str.startswith("Outside configured range").sum()
    )
    sentinel_count = int(
        filtered_quality["issue"]
        .str.startswith("Configured sentinel date")
        .sum()
    )
    parse_issue_count = int(
        filtered_quality["issue"].str.contains("could not be parsed").sum()
    )
    quality_metrics[0].metric("Exact duplicate rows removed", f"{duplicates_removed:,}")
    quality_metrics[1].metric("Out-of-range values", f"{range_issue_count:,}")
    quality_metrics[2].metric("Sentinel dates retained", f"{sentinel_count:,}")
    quality_metrics[3].metric("Unparsed values", f"{parse_issue_count:,}")

    st.caption(
        f"Showing {len(filtered_quality):,} quality findings for the currently "
        "filtered members. Existing missing values are not included as findings."
    )
    st.dataframe(filtered_quality, width="stretch", hide_index=True)
    st.download_button(
        "Download quality findings as CSV",
        data=csv_bytes(filtered_quality),
        file_name="acdc_data_quality_findings.csv",
        mime="text/csv",
    )
