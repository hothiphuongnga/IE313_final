from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATA_PATH = Path(__file__).with_name("wine.csv")
PALETTE = ["#8C2942", "#2D7167", "#D69A3A", "#526A8A", "#A65B3F"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

st.set_page_config(
    page_title="Wine Lab | Phân tích dữ liệu",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');
    :root {
        --ink: #202925;
        --muted: #66716a;
        --paper: #f5f5ef;
        --line: #dedfd6;
        --wine: #8c2942;
        --leaf: #2d7167;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    [data-testid="stHeader"] { background: rgba(245,245,239,.92); }
    [data-testid="stSidebar"] { background: #e9eee8; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h2 { font-family: 'DM Serif Display', serif; font-size: 1.45rem; }
    .block-container { max-width: 1440px; padding-top: 3.75rem; padding-bottom: 3rem; }
    .eyebrow { color: var(--wine); font-size: .73rem; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; }
    .hero-title { font-family: 'DM Serif Display', serif; font-size: 2.7rem; line-height: 1.08; margin: .25rem 0 .45rem; }
    .hero-copy { color: var(--muted); font-size: .98rem; margin-bottom: 1.3rem; }
    [data-testid="stMetric"] { background: #fffefa; border: 1px solid var(--line); border-radius: 7px; padding: 16px 18px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'DM Serif Display', serif; }
    [data-testid="stTabs"] button { font-weight: 600; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 5px; }
    div[data-testid="stAlert"] { border-radius: 5px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def format_feature(feature: str) -> str:
    return feature.replace("_", " ")


def make_histogram(data: pd.DataFrame, feature: str, bins: int) -> go.Figure:
    figure = px.histogram(
        data,
        x=feature,
        color="class",
        nbins=bins,
        barmode="overlay",
        opacity=0.78,
        color_discrete_sequence=PALETTE,
        labels={feature: format_feature(feature), "class": "Lớp"},
        marginal="box",
    )
    figure.update_layout(
        height=430,
        legend_title_text="Lớp rượu",
        bargap=0.08,
        xaxis_title=format_feature(feature),
        yaxis_title="Số mẫu",
    )
    return figure


st.sidebar.markdown("<div class='eyebrow'>Wine dataset · 01</div>", unsafe_allow_html=True)
st.sidebar.markdown("## Bảng điều khiển")
st.sidebar.caption("Chọn dữ liệu và phạm vi mẫu để khám phá.")
uploaded_file = st.sidebar.file_uploader("Nạp tệp CSV", type=["csv"])

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        data_name = uploaded_file.name
    else:
        df = load_default_data()
        data_name = DATA_PATH.name
except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"Không thể đọc dữ liệu: {error}")
    st.stop()

if "class" not in df.columns:
    st.error("Tệp CSV cần có cột nhãn `class` để phân tích theo lớp.")
    st.stop()

numeric_features = df.select_dtypes(include="number").columns.drop("class", errors="ignore").tolist()
if not numeric_features:
    st.error("Tệp CSV cần có ít nhất một thuộc tính dạng số ngoài cột `class`.")
    st.stop()

classes = sorted(df["class"].dropna().unique().tolist(), key=str)
selected_classes = st.sidebar.multiselect("Lớp rượu", classes, default=classes)
filtered_df = df[df["class"].isin(selected_classes)].copy()

range_feature = st.sidebar.selectbox(
    "Lọc theo thuộc tính",
    ["Không lọc"] + numeric_features,
    format_func=lambda value: "Không lọc" if value == "Không lọc" else format_feature(value),
)
if range_feature != "Không lọc" and not filtered_df.empty:
    minimum = float(filtered_df[range_feature].min())
    maximum = float(filtered_df[range_feature].max())
    if minimum < maximum:
        selected_range = st.sidebar.slider(
            f"Phạm vi {format_feature(range_feature)}",
            min_value=minimum,
            max_value=maximum,
            value=(minimum, maximum),
        )
        filtered_df = filtered_df[
            filtered_df[range_feature].between(selected_range[0], selected_range[1])
        ]

st.sidebar.divider()
st.sidebar.caption(f"Nguồn dữ liệu: {data_name}\n\n{len(df):,} dòng ban đầu")

st.markdown("<div class='eyebrow'>PHÂN TÍCH VÀ TRỰC QUAN DỮ LIỆU</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-title'>Wine Lab</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='hero-copy'>Khám phá đặc trưng hóa học của rượu vang · {data_name} · "
    f"{len(filtered_df):,} / {len(df):,} mẫu đang hiển thị</div>",
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.warning("Không có mẫu phù hợp với bộ lọc hiện tại. Hãy chọn lại lớp hoặc phạm vi giá trị.")
    st.stop()

missing_count = int(df.isna().sum().sum())
duplicate_count = int(df.duplicated().sum())
class_count = filtered_df["class"].nunique()
metric_columns = st.columns(4)
metric_columns[0].metric("Mẫu đang xem", f"{len(filtered_df):,}", f"/{len(df):,} mẫu")
metric_columns[1].metric("Thuộc tính số", len(numeric_features), "không tính nhãn lớp")
metric_columns[2].metric("Số lớp", class_count, "trong bộ lọc hiện tại")
metric_columns[3].metric("Ô dữ liệu thiếu", missing_count, f"{duplicate_count} dòng trùng")

overview_tab, feature_tab, relation_tab, quality_tab = st.tabs(
    ["Tổng quan", "Phân phối", "Mối quan hệ", "Dữ liệu"]
)

with overview_tab:
    left_column, right_column = st.columns([0.92, 1.35], gap="large")
    with left_column:
        st.subheader("Cơ cấu mẫu theo lớp")
        counts = filtered_df["class"].value_counts().sort_index().rename_axis("class").reset_index(name="Số mẫu")
        figure = px.bar(
            counts,
            x="class",
            y="Số mẫu",
            color="class",
            text="Số mẫu",
            color_discrete_sequence=PALETTE,
            labels={"class": "Lớp rượu"},
        )
        figure.update_layout(showlegend=False, height=360, xaxis_title="Lớp rượu", yaxis_title="Số mẫu")
        figure.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(figure, width="stretch")
    with right_column:
        st.subheader("Phân bố Alcohol theo lớp")
        figure = px.box(
            filtered_df,
            x="class",
            y="Alcohol" if "Alcohol" in filtered_df.columns else numeric_features[0],
            color="class",
            points="outliers",
            color_discrete_sequence=PALETTE,
            labels={"class": "Lớp rượu"},
        )
        figure.update_layout(showlegend=False, height=360, xaxis_title="Lớp rượu")
        st.plotly_chart(figure, width="stretch")

    st.subheader("Giá trị trung bình theo lớp")
    mean_table = filtered_df.groupby("class")[numeric_features].mean().T
    mean_table.index = [format_feature(feature) for feature in mean_table.index]
    st.dataframe(mean_table.round(2), width="stretch")

with feature_tab:
    control_column, chart_column = st.columns([0.28, 0.72], gap="large")
    with control_column:
        st.subheader("Thuộc tính")
        feature = st.selectbox(
            "Chọn biến cần khảo sát",
            numeric_features,
            format_func=format_feature,
            key="distribution_feature",
        )
        bins = st.slider("Số khoảng histogram", min_value=5, max_value=40, value=18)
        stats = filtered_df[feature].describe().rename(
            {"count": "Số mẫu", "mean": "Trung bình", "std": "Độ lệch chuẩn", "min": "Nhỏ nhất", "25%": "Q1", "50%": "Trung vị", "75%": "Q3", "max": "Lớn nhất"}
        )
        st.dataframe(stats.round(3).to_frame(name="Giá trị"), width="stretch")
    with chart_column:
        st.subheader(f"Phân phối {format_feature(feature)}")
        st.plotly_chart(make_histogram(filtered_df, feature, bins), width="stretch")
        st.caption("Histogram cho hình dạng phân phối; box plot giúp so sánh trung vị, độ phân tán và ngoại lệ giữa các lớp.")

with relation_tab:
    relation_left, relation_right = st.columns([0.24, 0.76], gap="large")
    with relation_left:
        st.subheader("Thiết lập")
        x_feature = st.selectbox(
            "Trục X",
            numeric_features,
            index=numeric_features.index("Alcohol") if "Alcohol" in numeric_features else 0,
            format_func=format_feature,
        )
        y_default = numeric_features.index("Color_intensity") if "Color_intensity" in numeric_features else min(1, len(numeric_features) - 1)
        y_feature = st.selectbox(
            "Trục Y",
            numeric_features,
            index=y_default,
            format_func=format_feature,
        )
        show_trend = st.checkbox("Hiển thị đường xu hướng", value=True)
    with relation_right:
        st.subheader("Biểu đồ phân tán")
        figure = px.scatter(
            filtered_df,
            x=x_feature,
            y=y_feature,
            color="class",
            hover_data=numeric_features[:5],
            color_discrete_sequence=PALETTE,
            labels={x_feature: format_feature(x_feature), y_feature: format_feature(y_feature), "class": "Lớp"},
        )
        if show_trend:
            for class_value in filtered_df["class"].dropna().unique():
                class_data = filtered_df[filtered_df["class"] == class_value]
                if len(class_data) > 1 and class_data[x_feature].nunique() > 1:
                    coefficients = np.polyfit(class_data[x_feature], class_data[y_feature], 1)
                    x_values = np.linspace(class_data[x_feature].min(), class_data[x_feature].max(), 50)
                    figure.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=np.polyval(coefficients, x_values),
                            mode="lines",
                            name=f"Xu hướng lớp {class_value}",
                            line={"dash": "dot", "width": 2},
                            hoverinfo="skip",
                        )
                    )
        figure.update_layout(height=520, legend_title_text="Lớp rượu")
        st.plotly_chart(figure, width="stretch")

    correlation = filtered_df[numeric_features].corr()
    upper_triangle = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    strongest_pairs = upper_triangle.stack().rename("Tương quan Pearson").to_frame()
    strongest_pairs = strongest_pairs.reindex(
        strongest_pairs["Tương quan Pearson"].abs().sort_values(ascending=False).index
    ).head(5)
    st.subheader("Các cặp tương quan mạnh nhất")
    if strongest_pairs.empty:
        st.info("Cần ít nhất hai thuộc tính số để tính tương quan.")
    else:
        strongest_pairs.index = [f"{format_feature(left)} × {format_feature(right)}" for left, right in strongest_pairs.index]
        st.dataframe(strongest_pairs.round(3), width="stretch")

    st.subheader("Ma trận tương quan Pearson")
    heatmap = px.imshow(
        correlation,
        text_auto=".2f",
        color_continuous_scale=["#8c2942", "#f5f5ef", "#2d7167"],
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels={"x": "Thuộc tính", "y": "Thuộc tính", "color": "Pearson"},
    )
    heatmap.update_layout(height=max(520, 38 * len(numeric_features)), margin={"l": 10, "r": 10, "t": 30, "b": 100})
    heatmap.update_xaxes(tickangle=-45, tickfont={"size": 10})
    heatmap.update_yaxes(tickfont={"size": 10})
    st.plotly_chart(heatmap, width="stretch")
    st.caption("Pearson gần +1: đồng biến tuyến tính; gần -1: nghịch biến; gần 0: ít quan hệ tuyến tính. Tương quan không hàm ý quan hệ nhân quả.")

with quality_tab:
    missing_by_column = df.isna().sum().rename("Giá trị thiếu").to_frame()
    missing_by_column["Tỷ lệ thiếu (%)"] = (df.isna().mean() * 100).round(2)
    quality_left, quality_right = st.columns([0.46, 0.54], gap="large")
    with quality_left:
        st.subheader("Kiểm tra chất lượng")
        st.dataframe(missing_by_column, width="stretch")
        st.caption(f"Kiểu dữ liệu: {df.dtypes.astype(str).value_counts().to_dict()}")
    with quality_right:
        st.subheader("Thống kê mô tả")
        st.dataframe(df[numeric_features].describe().round(2), width="stretch")
        st.caption(f"Dòng trùng lặp toàn bộ: {duplicate_count}")
        if duplicate_count:
            with st.expander("Xem các dòng trùng lặp"):
                st.dataframe(df[df.duplicated(keep=False)], width="stretch")

st.divider()
st.subheader("Bảng dữ liệu đang lọc")
st.dataframe(filtered_df, width="stretch", hide_index=True)
csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Tải dữ liệu đã lọc (.csv)",
    data=csv_data,
    file_name="wine_filtered.csv",
    mime="text/csv",
    type="primary",
)