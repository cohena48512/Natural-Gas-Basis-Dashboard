import io
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Natural Gas Market Intelligence",
    page_icon="🗺️",
    layout="wide",
)

# -----------------------------
# State lookup and map positions
# -----------------------------
STATE_CODES = {
    "Alabama":"AL", "Alaska":"AK", "Arizona":"AZ", "Arkansas":"AR",
    "California":"CA", "Colorado":"CO", "Connecticut":"CT", "Delaware":"DE",
    "Florida":"FL", "Georgia":"GA", "Hawaii":"HI", "Idaho":"ID",
    "Illinois":"IL", "Indiana":"IN", "Iowa":"IA", "Kansas":"KS",
    "Kentucky":"KY", "Louisiana":"LA", "Maine":"ME", "Maryland":"MD",
    "Massachusetts":"MA", "Michigan":"MI", "Minnesota":"MN",
    "Mississippi":"MS", "Missouri":"MO", "Montana":"MT", "Nebraska":"NE",
    "Nevada":"NV", "New Hampshire":"NH", "New Jersey":"NJ",
    "New Mexico":"NM", "New York":"NY", "North Carolina":"NC",
    "North Dakota":"ND", "Ohio":"OH", "Oklahoma":"OK", "Oregon":"OR",
    "Pennsylvania":"PA", "Rhode Island":"RI", "South Carolina":"SC",
    "South Dakota":"SD", "Tennessee":"TN", "Texas":"TX", "Utah":"UT",
    "Vermont":"VT", "Virginia":"VA", "Washington":"WA",
    "West Virginia":"WV", "Wisconsin":"WI", "Wyoming":"WY",
}
CODE_TO_STATE = {v: k for k, v in STATE_CODES.items()}

# Approximate label coordinates (longitude, latitude), for readable state labels.
STATE_COORDS = {
    "AL":(-86.8,32.8),"AZ":(-111.7,34.2),"CA":(-120.0,37.0),
    "CO":(-105.5,39.0),"FL":(-82.0,28.0),"GA":(-83.5,32.5),
    "IA":(-93.5,42.0),"IL":(-89.0,40.0),"IN":(-86.1,40.0),
    "KS":(-98.0,38.5),"KY":(-85.3,37.8),"LA":(-91.5,31.0),
    "MA":(-71.8,42.3),"MD":(-76.5,39.0),"MI":(-85.5,44.0),
    "MN":(-94.5,46.0),"MS":(-89.7,32.7),"NC":(-79.5,35.5),
    "ND":(-100.5,47.5),"NJ":(-74.5,40.1),"NM":(-106.0,34.5),
    "NY":(-75.5,42.9),"OH":(-82.8,40.3),"OK":(-97.0,35.5),
    "OR":(-120.5,44.0),"PA":(-77.8,41.0),"SC":(-80.8,33.8),
    "TN":(-86.0,35.8),"TX":(-99.0,31.0),"VA":(-79.0,37.5),
    "WA":(-120.5,47.5),"WV":(-80.5,38.6),"WY":(-107.5,43.0),
    "MO":(-92.5,38.5),"NE":(-99.8,41.5),"MT":(-110.5,47.0),
    "ID":(-114.5,44.2),"NV":(-116.8,39.3),"UT":(-111.5,39.3),
    "AR":(-92.5,34.8),"WI":(-89.8,44.5),"SD":(-100.0,44.5),
}

# -----------------------------
# Helpers
# -----------------------------
def clean_column_name(name):
    return re.sub(r"\s+", " ", str(name).strip()).lower()

def find_column(df, candidates):
    lookup = {clean_column_name(c): c for c in df.columns}
    for candidate in candidates:
        key = clean_column_name(candidate)
        if key in lookup:
            return lookup[key]
    return None

def parse_state(value):
    if pd.isna(value):
        return None
    # Your source may list multiple states separated by "/".
    first_state = str(value).split("/")[0].strip()
    if first_state in STATE_CODES:
        return STATE_CODES[first_state]
    upper = first_state.upper()
    if upper in CODE_TO_STATE:
        return upper
    # Handle common variants such as "NY - New York".
    match = re.search(r"\b([A-Z]{2})\b", upper)
    if match and match.group(1) in CODE_TO_STATE:
        return match.group(1)
    return None

def money_or_number(series):
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")

@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes, file_name):
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))

def prepare_data(raw):
    hub_col = find_column(raw, ["Hub", "Location", "Pricing Hub", "Trading Hub"])
    state_col = find_column(raw, ["State", "States"])
    region_col = find_column(raw, ["Region", "Market Region"])
    after_col = find_column(raw, ["Price After Basis", "Price After", "All-in Price"])
    basis_col = find_column(raw, ["Basis Differential", "Basis", "Basis Diff"])
    date_col = find_column(raw, ["Date", "Month", "Pricing Date", "Effective Date"])

    missing = []
    for label, col in [
        ("Hub", hub_col), ("State", state_col), ("Region", region_col),
        ("Price After Basis", after_col), ("Basis Differential", basis_col)
    ]:
        if col is None:
            missing.append(label)
    if missing:
        raise ValueError(
            "Could not find these required columns: "
            + ", ".join(missing)
            + ". Check the Excel column headers."
        )

    data = pd.DataFrame()
    data["Hub"] = raw[hub_col].astype(str).str.strip()
    data["State"] = raw[state_col].astype(str).str.split("/").str[0].str.strip()
    data["State Code"] = raw[state_col].apply(parse_state)
    data["Region"] = raw[region_col].astype(str).str.strip()
    data["Price After Basis"] = money_or_number(raw[after_col])
    data["Basis Differential"] = money_or_number(raw[basis_col])

    if date_col is not None:
        data["Date"] = pd.to_datetime(raw[date_col], errors="coerce")
    else:
        data["Date"] = pd.NaT

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["Hub", "Region"])
    data = data[data["Hub"].ne("") & data["Region"].ne("")]
    data["Regional Average - Price After Basis"] = (
        data.groupby("Region")["Price After Basis"].transform("mean")
    )
    data["Deviation from Regional Average"] = (
        data["Price After Basis"]
        - data["Regional Average - Price After Basis"]
    )
    data["Deviation (%)"] = np.where(
        data["Regional Average - Price After Basis"].abs() > 1e-12,
        data["Deviation from Regional Average"]
        / data["Regional Average - Price After Basis"].abs() * 100,
        np.nan,
    )
    return data, date_col is not None

def fmt_num(value):
    return "N/A" if pd.isna(value) else f"${value:,.3f}"

# -----------------------------
# Header and upload
# -----------------------------
st.title("Natural Gas Market Intelligence")
st.caption("Geographic pricing • Regional comparisons • Hub-level analysis")

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader(
        "Upload your Excel or CSV file",
        type=["xlsx", "xls", "csv"],
        help="Required columns: Hub, State, Region, Price After Basis, Basis Differential. Date is optional."
    )
    st.divider()
    st.caption("The dashboard calculates regional averages from the uploaded rows. Confirm that each row represents the intended pricing observation.")

if uploaded_file is None:
    st.info("Upload your natural gas basis workbook in the sidebar to populate the dashboard.")
    st.markdown(
        """
        **Expected columns**
        - `Hub`
        - `State`
        - `Region`
        - `Price After Basis`
        - `Basis Differential`
        - `Date` (optional; recommended for future historical analysis)
        """
    )
    st.stop()

try:
    raw_df = read_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
    df, has_date = prepare_data(raw_df)
except Exception as exc:
    st.error(f"Unable to load the workbook: {exc}")
    st.stop()

if df.empty:
    st.warning("No usable rows were found after cleaning the uploaded data.")
    st.stop()

# -----------------------------
# Filters
# -----------------------------
with st.sidebar:
    st.header("Filters")
    regions = sorted(df["Region"].dropna().unique().tolist())
    selected_regions = st.multiselect("Region", regions, default=regions)

    available_dates = df["Date"].dropna()
    if has_date and not available_dates.empty:
        date_values = sorted(available_dates.dt.date.unique())
        if len(date_values) > 1:
            selected_dates = st.date_input(
                "Date range",
                value=(min(date_values), max(date_values)),
                min_value=min(date_values),
                max_value=max(date_values),
            )
            if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                start_date, end_date = selected_dates
            else:
                start_date = end_date = selected_dates
        else:
            start_date = end_date = date_values[0]
            st.caption(f"Data date: {date_values[0]}")
    else:
        start_date = end_date = None
        st.caption("No usable Date column detected.")

filtered = df[df["Region"].isin(selected_regions)].copy()
if start_date is not None and end_date is not None:
    filtered = filtered[
        filtered["Date"].dt.date.between(start_date, end_date)
        | filtered["Date"].isna()
    ].copy()

if filtered.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

# Recalculate regional metrics after filters so comparisons reflect current selection.
filtered["Regional Average - Price After Basis"] = (
    filtered.groupby("Region")["Price After Basis"].transform("mean")
)
filtered["Deviation from Regional Average"] = (
    filtered["Price After Basis"]
    - filtered["Regional Average - Price After Basis"]
)
filtered["Deviation (%)"] = np.where(
    filtered["Regional Average - Price After Basis"].abs() > 1e-12,
    filtered["Deviation from Regional Average"]
    / filtered["Regional Average - Price After Basis"].abs() * 100,
    np.nan,
)

# -----------------------------
# KPI cards
# -----------------------------
st.subheader("Market Overview")
valid_prices = filtered["Price After Basis"].dropna()
valid_basis = filtered["Basis Differential"].dropna()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Average Price After Basis", fmt_num(valid_prices.mean()) if not valid_prices.empty else "N/A")
k2.metric("Average Basis Differential", fmt_num(valid_basis.mean()) if not valid_basis.empty else "N/A")
k3.metric("Hubs", f"{filtered['Hub'].nunique():,}")
k4.metric("Regions", f"{filtered['Region'].nunique():,}")

if has_date and filtered["Date"].notna().any():
    date_label = filtered["Date"].dropna().dt.strftime("%b %Y").drop_duplicates().tolist()
    st.caption("Date coverage: " + ", ".join(date_label[:6]) + (" ..." if len(date_label) > 6 else ""))

st.divider()

# -----------------------------
# Geographic map
# -----------------------------
st.subheader("Geographic Price Analysis")
map_metric = st.selectbox(
    "Map metric",
    [
        "Price After Basis",
        "Basis Differential",
        "Deviation from Regional Average",
    ],
    index=0,
)
map_data = (
    filtered.dropna(subset=["State Code", map_metric])
    .groupby("State Code", as_index=False)
    .agg(
        **{
            map_metric: (map_metric, "mean"),
            "Hub Count": ("Hub", "nunique"),
        }
    )
)
if map_data.empty:
    st.warning("No recognized state values are available to plot. Check the State column.")
else:
    map_data["Longitude"] = map_data["State Code"].map(lambda x: STATE_COORDS.get(x, (np.nan, np.nan))[0])
    map_data["Latitude"] = map_data["State Code"].map(lambda x: STATE_COORDS.get(x, (np.nan, np.nan))[1])
    fig_map = px.choropleth(
        map_data,
        locations="State Code",
        locationmode="USA-states",
        color=map_metric,
        scope="usa",
        color_continuous_scale="RdYlGn_r",
        hover_name="State Code",
        hover_data={"State Code": False, map_metric: ":.3f", "Hub Count": True},
        labels={map_metric: map_metric},
    )
    label_data = map_data.dropna(subset=["Longitude", "Latitude"])
    fig_map.add_trace(
        go.Scattergeo(
            lon=label_data["Longitude"],
            lat=label_data["Latitude"],
            text=label_data["State Code"],
            mode="text",
            textfont={"size": 10, "color": "black", "family": "Arial Black"},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_map.update_layout(
        height=600,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        geo={"scope": "usa", "showland": True, "landcolor": "#f3f4f6"},
        coloraxis_colorbar={"title": map_metric},
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("State values are averages of the selected hub observations in that state. A hub listing multiple states is assigned to the first state listed.")

st.divider()

# -----------------------------
# Regional comparison and hub ranking
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("Regional Comparison")
    regional_metric = st.selectbox(
        "Regional metric",
        ["Price After Basis", "Basis Differential"],
        key="regional_metric",
    )
    regional = (
        filtered.groupby("Region", as_index=False)[regional_metric]
        .mean()
        .sort_values(regional_metric, ascending=False)
    )
    fig_region = px.bar(
        regional,
        x="Region",
        y=regional_metric,
        color=regional_metric,
        color_continuous_scale="RdYlGn_r",
        text_auto=".3f",
        title=f"Average {regional_metric} by Region",
    )
    fig_region.update_layout(
        height=420,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title="Region",
        yaxis_title=regional_metric,
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
    )
    st.plotly_chart(fig_region, use_container_width=True)

with right:
    st.subheader("Hub Ranking")
    ranking_metric = st.selectbox(
        "Rank hubs by",
        ["Price After Basis", "Basis Differential", "Deviation from Regional Average"],
        key="ranking_metric",
    )
    ranking = (
        filtered.groupby(["Hub", "Region"], as_index=False)[ranking_metric]
        .mean()
        .sort_values(ranking_metric, ascending=False)
    )
    show_n = st.slider("Number of hubs", min_value=5, max_value=min(30, max(5, len(ranking))), value=min(10, max(5, len(ranking))))
    ranking_view = ranking.head(show_n)
    fig_rank = px.bar(
        ranking_view.sort_values(ranking_metric),
        x=ranking_metric,
        y="Hub",
        color=ranking_metric,
        color_continuous_scale="RdYlGn_r",
        orientation="h",
        hover_data=["Region"],
        title=f"Top {min(show_n, len(ranking))} Hubs",
    )
    fig_rank.update_layout(
        height=420,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title=ranking_metric,
        yaxis_title="",
        margin={"l": 10, "r": 10, "t": 55, "b": 10},
    )
    st.plotly_chart(fig_rank, use_container_width=True)

st.divider()

# -----------------------------
# Hub deviation analysis
# -----------------------------
st.subheader("Hub Deviation from Regional Average")
st.write(
    "Positive values indicate a hub is above its selected-region average; "
    "negative values indicate it is below. This comparison uses Price After Basis."
)
deviation = (
    filtered.groupby(["Hub", "Region"], as_index=False)
    .agg(
        **{
            "Price After Basis": ("Price After Basis", "mean"),
            "Regional Average": ("Regional Average - Price After Basis", "mean"),
            "Deviation": ("Deviation from Regional Average", "mean"),
        }
    )
    .sort_values("Deviation", ascending=False)
)
fig_dev = px.bar(
    deviation,
    x="Hub",
    y="Deviation",
    color="Deviation",
    color_continuous_scale="RdYlGn_r",
    hover_data=["Region", "Price After Basis", "Regional Average"],
    title="Hub Premium / Discount vs. Regional Average",
)
fig_dev.add_hline(y=0, line_dash="dash", line_color="black")
fig_dev.update_layout(
    height=450,
    xaxis_title="Hub",
    yaxis_title="Price difference",
    coloraxis_showscale=False,
    margin={"l": 10, "r": 10, "t": 55, "b": 90},
)
st.plotly_chart(fig_dev, use_container_width=True)

# -----------------------------
# Simple anomaly screening
# -----------------------------
st.subheader("Potential Price Outliers")
st.caption(
    "Screening only: flags observations outside 1.5 × the interquartile range "
    "within each region. These are investigation candidates, not proof of a market anomaly."
)

def flag_iqr(group):
    values = group["Price After Basis"].dropna()
    if len(values) < 4:
        group["Potential Outlier"] = False
        return group
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    group["Potential Outlier"] = (
        group["Price After Basis"].lt(low)
        | group["Price After Basis"].gt(high)
    )
    return group

outlier_data = filtered.groupby("Region", group_keys=False).apply(flag_iqr)
outliers = outlier_data[outlier_data["Potential Outlier"]].copy()
if outliers.empty:
    st.success("No observations were flagged by this regional IQR screen.")
else:
    st.warning(f"{len(outliers)} observation(s) flagged for review.")
    st.dataframe(
        outliers[["Hub", "State", "Region", "Price After Basis", "Basis Differential"] + (["Date"] if has_date else [])],
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# -----------------------------
# Data table and downloads
# -----------------------------
st.subheader("Underlying Data")
display_columns = [
    "Hub", "State", "State Code", "Region", "Price After Basis",
    "Basis Differential", "Regional Average - Price After Basis",
    "Deviation from Regional Average", "Deviation (%)",
]
if has_date:
    display_columns.append("Date")

st.dataframe(
    filtered[display_columns].sort_values(["Region", "Hub"]),
    use_container_width=True,
    hide_index=True,
)

csv_bytes = filtered[display_columns].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered analysis (CSV)",
    data=csv_bytes,
    file_name="natural_gas_basis_analysis.csv",
    mime="text/csv",
)

st.caption(
    "Methodology: state and regional values are arithmetic averages of the uploaded observations. "
    "If the workbook contains duplicate hubs, multiple delivery points, or different units, "
    "review those records before interpreting averages."
)
