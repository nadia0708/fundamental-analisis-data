
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =====================================================
# 1. KONFIGURASI DASHBOARD
# =====================================================

st.set_page_config(
    page_title="E-Commerce Data Analysis",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background-color: #F5F7FA;
        border: 1px solid #E5E7EB;
        padding: 16px;
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: #253858;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. MEMBACA DATASET OTOMATIS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():

    orders = pd.read_csv(
        BASE_DIR / "orders_dataset.csv"
    )

    order_items = pd.read_csv(
        BASE_DIR / "order_items_dataset.csv"
    )

    products = pd.read_csv(
        BASE_DIR / "products_dataset.csv"
    )

    # Menggabungkan orders dan order_items
    gabungan_data = pd.merge(
        orders,
        order_items,
        on="order_id",
        how="inner"
    )

    # Menggabungkan hasil dengan products
    main_data = pd.merge(
        gabungan_data,
        products,
        on="product_id",
        how="inner"
    )

    # Menyimpan informasi sebelum cleaning
    total_rows = len(main_data)
    missing_values = main_data.isnull().sum()
    duplicate_values = main_data.duplicated().sum()

    # Cleaning missing values sesuai notebook
    main_data.dropna(axis=0, inplace=True)

    # Mengubah tanggal menjadi datetime
    main_data["order_purchase_timestamp"] = pd.to_datetime(
        main_data["order_purchase_timestamp"],
        errors="coerce"
    )

    main_data["price"] = pd.to_numeric(
        main_data["price"],
        errors="coerce"
    )

    main_data["freight_value"] = pd.to_numeric(
        main_data["freight_value"],
        errors="coerce"
    )

    # Membersihkan nilai yang gagal dikonversi
    main_data.dropna(
        subset=[
            "order_purchase_timestamp",
            "price",
            "freight_value"
        ],
        inplace=True
    )

    # Memfilter periode Januari-Desember 2017
    data_2017 = main_data[
        (main_data["order_purchase_timestamp"] >= "2017-01-01") &
        (main_data["order_purchase_timestamp"] < "2018-01-01")
    ].copy()

    # Membuat kolom bulan
    data_2017["month"] = (
        data_2017["order_purchase_timestamp"]
        .dt.to_period("M")
    )

    # Menghitung freight ratio
    data_2017["freight_ratio"] = np.where(
        data_2017["price"] > 0,
        (
            data_2017["freight_value"] /
            data_2017["price"]
        ) * 100,
        np.nan
    )

    audit = {
        "total_rows": total_rows,
        "missing_values": missing_values,
        "duplicate_values": duplicate_values,
        "clean_rows": len(main_data),
        "data_2017_rows": len(data_2017)
    }

    return data_2017, audit


# Menangani apabila dataset tidak ditemukan
try:
    with st.spinner("Memuat dataset..."):
        data_2017, audit = load_data()

except FileNotFoundError as e:
    st.error("Dataset tidak ditemukan!")
    st.warning(
        "Pastikan ketiga file CSV berada dalam folder yang sama "
        "dengan dashboard.py."
    )
    st.code("""
E-Commerce-Dashboard/
│
├── dashboard.py
├── orders_dataset.csv
├── order_items_dataset.csv
└── products_dataset.csv
    """)
    st.stop()

except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca dataset: {e}")
    st.stop()


if data_2017.empty:
    st.warning("Tidak terdapat data untuk periode tahun 2017.")
    st.stop()


# =====================================================
# 3. MEMBUAT DATA ANALISIS
# =====================================================

# Analisis pertanyaan 1
explore_status = (
    data_2017.groupby(
        ["month", "order_status"]
    )["order_id"]
    .nunique()
    .reset_index(name="total_orders")
)

status_summary = (
    data_2017.groupby("order_status")["order_id"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index(name="total_orders")
)

monthly_orders = (
    data_2017.groupby("month")["order_id"]
    .nunique()
    .reset_index(name="total_orders")
)

# Analisis pertanyaan 2
data_shipping = data_2017[
    (data_2017["price"] > 0) &
    (data_2017["freight_value"] >= 0)
].copy()

explore_shipping = (
    data_shipping.groupby(
        ["month", "product_category_name"]
    )
    .agg(
        avg_freight=("freight_value", "mean"),
        total_freight=("freight_value", "sum"),
        total_price=("price", "sum")
    )
    .reset_index()
)

explore_shipping["freight_ratio"] = (
    explore_shipping["total_freight"] /
    explore_shipping["total_price"]
) * 100

# Top 10 sesuai metode notebook
top_freight = (
    explore_shipping.groupby("product_category_name")["avg_freight"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

top_ratio = (
    explore_shipping.groupby("product_category_name")["freight_ratio"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)


# =====================================================
# 4. SIDEBAR DAN FILTER
# =====================================================

st.sidebar.title("📊 E-Commerce Dashboard")
st.sidebar.caption("Fundamental Analisis Data")

st.sidebar.divider()

st.sidebar.subheader("Filter Data")

month_labels = {
    "2017-01": "Januari",
    "2017-02": "Februari",
    "2017-03": "Maret",
    "2017-04": "April",
    "2017-05": "Mei",
    "2017-06": "Juni",
    "2017-07": "Juli",
    "2017-08": "Agustus",
    "2017-09": "September",
    "2017-10": "Oktober",
    "2017-11": "November",
    "2017-12": "Desember"
}

all_months = list(month_labels.keys())

selected_months = st.sidebar.multiselect(
    "Pilih Bulan",
    options=all_months,
    default=all_months,
    format_func=lambda x: month_labels[x]
)

all_status = sorted(
    data_2017["order_status"].dropna().unique().tolist()
)

selected_status = st.sidebar.multiselect(
    "Pilih Status Pesanan",
    options=all_status,
    default=all_status
)

all_categories = sorted(
    data_2017["product_category_name"]
    .dropna()
    .unique()
    .tolist()
)

selected_categories = st.sidebar.multiselect(
    "Pilih Kategori Produk",
    options=all_categories,
    default=all_categories
)

if st.sidebar.button("Reset Filter", use_container_width=True):
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("Periode analisis: Januari–Desember 2017")


# =====================================================
# 5. MENERAPKAN FILTER
# =====================================================

filtered_data = data_2017[
    data_2017["month"].astype(str).isin(selected_months) &
    data_2017["order_status"].isin(selected_status) &
    data_2017["product_category_name"].isin(selected_categories)
].copy()

if filtered_data.empty:
    st.warning(
        "Tidak ada data yang sesuai dengan filter yang dipilih."
    )
    st.stop()


# Menghitung ulang analisis berdasarkan filter
filtered_status = (
    filtered_data.groupby(
        ["month", "order_status"]
    )["order_id"]
    .nunique()
    .reset_index(name="total_orders")
)

filtered_monthly = (
    filtered_data.groupby("month")["order_id"]
    .nunique()
    .reset_index(name="total_orders")
)

filtered_shipping = filtered_data[
    (filtered_data["price"] > 0) &
    (filtered_data["freight_value"] >= 0)
].copy()

filtered_shipping_summary = (
    filtered_shipping.groupby(
        ["month", "product_category_name"]
    )
    .agg(
        avg_freight=("freight_value", "mean"),
        total_freight=("freight_value", "sum"),
        total_price=("price", "sum")
    )
    .reset_index()
)

filtered_shipping_summary["freight_ratio"] = (
    filtered_shipping_summary["total_freight"] /
    filtered_shipping_summary["total_price"]
) * 100


# =====================================================
# 6. HEADER DASHBOARD
# =====================================================

st.title("E-Commerce Public Dataset")
st.markdown(
    "### Dashboard Analisis Pesanan dan Biaya Pengiriman"
)

st.markdown(
    "Dashboard interaktif untuk menganalisis perkembangan "
    "pesanan dan efisiensi biaya pengiriman selama tahun 2017."
)

st.divider()


# =====================================================
# 7. OVERVIEW / KPI
# =====================================================

st.subheader("📌 Ringkasan Performa")

total_orders = filtered_data["order_id"].nunique()

total_items = len(filtered_data)

total_revenue = filtered_data["price"].sum()

avg_freight = filtered_shipping["freight_value"].mean()

if filtered_shipping["price"].sum() > 0:
    overall_freight_ratio = (
        filtered_shipping["freight_value"].sum() /
        filtered_shipping["price"].sum()
    ) * 100
else:
    overall_freight_ratio = 0

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Pesanan",
    f"{total_orders:,}"
)

col2.metric(
    "Total Item",
    f"{total_items:,}"
)

col3.metric(
    "Total Harga Produk",
    f"R$ {total_revenue:,.2f}"
)

col4.metric(
    "Rata-rata Ongkir",
    f"R$ {avg_freight:,.2f}" if pd.notna(avg_freight) else "-"
)

col5.metric(
    "Rasio Ongkir",
    f"{overall_freight_ratio:.2f}%"
)

st.caption(
    "Catatan: Total pesanan menggunakan jumlah order_id unik. "
    "Total harga produk dihitung dari kolom price."
)


# =====================================================
# 8. TAB DASHBOARD
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Analisis Pesanan",
    "🚚 Biaya Pengiriman",
    "🔍 Eksplorasi Data",
    "💡 Kesimpulan & Rekomendasi"
])


# =====================================================
# TAB 1 - ANALISIS PESANAN
# =====================================================

with tab1:

    st.header("Analisis Perkembangan Pesanan")

    st.markdown(
        "**Pertanyaan Bisnis 1:** "
        "Bagaimana perkembangan jumlah pesanan berdasarkan status "
        "pesanan pada setiap bulan selama periode Januari 2017–"
        "Desember 2017, sebagai dasar untuk mengevaluasi "
        "perkembangan dan kondisi operasional penjualan?"
    )

    st.divider()

    # Membuat pivot table
    pivot_status = filtered_status.pivot(
        index="month",
        columns="order_status",
        values="total_orders"
    ).fillna(0)

    # Memastikan urutan bulan
    month_periods = pd.period_range(
        "2017-01",
        "2017-12",
        freq="M"
    )

    pivot_status = pivot_status.reindex(
        month_periods,
        fill_value=0
    )

    # Filter status yang dipilih
    pivot_status = pivot_status.reindex(
        columns=selected_status,
        fill_value=0
    )

    pivot_status.index = [
        month_labels[str(x)] for x in pivot_status.index
    ]

    # LINE CHART
    st.subheader(
        "1. Perkembangan Jumlah Pesanan Berdasarkan Status"
    )

    line_data = (
        pivot_status.reset_index()
        .rename(columns={"index": "Bulan"})
        .melt(
            id_vars="Bulan",
            var_name="Status Pesanan",
            value_name="Jumlah Pesanan"
        )
    )

    fig1 = px.line(
        line_data,
        x="Bulan",
        y="Jumlah Pesanan",
        color="Status Pesanan",
        markers=True,
        title="Perkembangan Jumlah Pesanan Tahun 2017"
    )

    fig1.update_layout(
        hovermode="x unified",
        xaxis_title="Bulan",
        yaxis_title="Jumlah Pesanan",
        legend_title="Status Pesanan"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # STACKED BAR CHART
    st.subheader(
        "2. Distribusi Jumlah Pesanan Berdasarkan Status"
    )

    fig2 = px.bar(
        line_data,
        x="Bulan",
        y="Jumlah Pesanan",
        color="Status Pesanan",
        barmode="stack",
        title="Distribusi Pesanan Setiap Bulan Tahun 2017"
    )

    fig2.update_layout(
        xaxis_title="Bulan",
        yaxis_title="Jumlah Pesanan",
        legend_title="Status Pesanan"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # TABEL STATUS
    st.subheader("3. Ringkasan Pesanan Berdasarkan Status")

    filtered_status_summary = (
        filtered_data.groupby("order_status")["order_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="total_orders")
    )

    st.dataframe(
        filtered_status_summary,
        use_container_width=True,
        hide_index=True
    )

    # TABEL BULANAN
    st.subheader("4. Ringkasan Pesanan Setiap Bulan")

    monthly_table = (
        filtered_data.groupby("month")["order_id"]
        .nunique()
        .reindex(month_periods, fill_value=0)
        .reset_index()
    )

    monthly_table.columns = [
        "Bulan",
        "Jumlah Pesanan"
    ]

    monthly_table["Bulan"] = monthly_table["Bulan"].astype(str)

    monthly_table["Bulan"] = monthly_table["Bulan"].map(
        month_labels
    )

    st.dataframe(
        monthly_table,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# TAB 2 - ANALISIS BIAYA PENGIRIMAN
# =====================================================

with tab2:

    st.header("Analisis Biaya Pengiriman")

    st.markdown(
        "**Pertanyaan Bisnis 2:** "
        "Bagaimana perbandingan rata-rata biaya pengiriman dan "
        "rasio biaya pengiriman terhadap harga produk berdasarkan "
        "kategori produk pada setiap bulan selama periode "
        "Januari 2017–Desember 2017?"
    )

    st.divider()

    if filtered_shipping_summary.empty:

        st.warning(
            "Tidak ada data biaya pengiriman yang valid."
        )

    else:

        # HEATMAP AVG FREIGHT
        st.subheader(
            "1. Rata-rata Biaya Pengiriman Berdasarkan Kategori Produk"
        )

        freight_pivot = filtered_shipping_summary.pivot(
            index="product_category_name",
            columns="month",
            values="avg_freight"
        )

        selected_periods = [
            pd.Period(x, freq="M")
            for x in selected_months
        ]

        freight_pivot = freight_pivot.reindex(
            columns=selected_periods
        )

        # Ubah tipe data Period menjadi string
        freight_pivot.columns = freight_pivot.columns.astype(str)
        freight_pivot.index = freight_pivot.index.astype(str)
        
        # Membuat grafik ketiga
        fig3 = px.imshow(
            freight_pivot,
            aspect="auto",
            title="Rata-rata Biaya Pengiriman per Kategori dan Bulan"
            )

        fig3.update_layout(
            xaxis_title="Bulan",
            yaxis_title="Kategori Produk"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # HEATMAP FREIGHT RATIO
        st.subheader(
            "2. Rasio Biaya Pengiriman terhadap Harga Produk"
        )

        ratio_pivot = filtered_shipping_summary.pivot(
            index="product_category_name",
            columns="month",
            values="freight_ratio"
        )

        ratio_pivot = ratio_pivot.reindex(
            columns=selected_periods
        )
        
        # Membuat pivot untuk heatmap persentase biaya pengiriman
        freight_ratio_pivot = filtered_shipping_summary.pivot(
            index="product_category_name",
            columns="month",
            values="freight_ratio"
            )
            
        # Mengubah Period menjadi string agar bisa dibaca Plotly
        freight_ratio_pivot.columns = freight_ratio_pivot.columns.astype(str)
        freight_ratio_pivot.index = freight_ratio_pivot.index.astype(str)
        
        # Membuat grafik keempat
        fig4 = px.imshow(
            freight_ratio_pivot,
            aspect="auto",
            title="Persentase Biaya Pengiriman per Kategori dan Bulan",
            labels={
                "x": "Bulan",
                "y": "Kategori Produk",
                "color": "Persentase (%)"
                })
                
        fig4.update_layout(
            xaxis_title="Bulan",
            yaxis_title="Kategori Produk"
            )
            
        st.plotly_chart(
            fig4,
            use_container_width=True
            )

        st.divider()

        # TOP 10 CATEGORY
        st.subheader("3. Kategori Produk dengan Biaya Pengiriman Tertinggi")

        # Mengikuti metode notebook:
        # Mean avg_freight dari agregasi bulanan
        filtered_top_freight = (
            filtered_shipping_summary.groupby(
                "product_category_name"
            )["avg_freight"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        # Mean freight_ratio dari agregasi bulanan
        filtered_top_ratio = (
            filtered_shipping_summary.groupby(
                "product_category_name"
            )["freight_ratio"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        col1, col2 = st.columns(2)

        with col1:

            fig5 = px.bar(
                filtered_top_freight.sort_values("avg_freight"),
                x="avg_freight",
                y="product_category_name",
                orientation="h",
                title="Top 10 Rata-rata Biaya Pengiriman",
                labels={
                    "avg_freight": "Rata-rata Ongkir",
                    "product_category_name": "Kategori Produk"
                },
                color="avg_freight",
                color_continuous_scale="YlOrRd"
            )

            st.plotly_chart(
                fig5,
                use_container_width=True
            )

        with col2:

            fig6 = px.bar(
                filtered_top_ratio.sort_values("freight_ratio"),
                x="freight_ratio",
                y="product_category_name",
                orientation="h",
                title="Top 10 Rasio Biaya Pengiriman",
                labels={
                    "freight_ratio": "Rasio Ongkir (%)",
                    "product_category_name": "Kategori Produk"
                },
                color="freight_ratio",
                color_continuous_scale="RdYlGn_r"
            )

            st.plotly_chart(
                fig6,
                use_container_width=True
            )

        # TABEL
        st.subheader("4. Tabel Ringkasan Biaya Pengiriman")

        shipping_table = filtered_shipping_summary.copy()

        shipping_table["month"] = (
            shipping_table["month"].astype(str)
        )

        shipping_table["month"] = (
            shipping_table["month"].map(month_labels)
        )

        shipping_table = shipping_table.rename(
            columns={
                "month": "Bulan",
                "product_category_name": "Kategori Produk",
                "avg_freight": "Rata-rata Ongkir",
                "total_freight": "Total Ongkir",
                "total_price": "Total Harga Produk",
                "freight_ratio": "Rasio Ongkir (%)"
            }
        )

        st.dataframe(
            shipping_table,
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# TAB 3 - EKSPLORASI DATA
# =====================================================

with tab3:

    st.header("Eksplorasi Data")

    st.markdown(
        "Halaman ini menampilkan informasi kualitas dataset "
        "dan hasil proses data wrangling."
    )

    st.divider()

    # DATA QUALITY
    st.subheader("1. Informasi Kualitas Data")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Data Sebelum Cleaning",
        f"{audit['total_rows']:,}"
    )

    col2.metric(
        "Jumlah Missing Values",
        f"{audit['missing_values'].sum():,}"
    )

    col3.metric(
        "Jumlah Data Duplikat",
        f"{audit['duplicate_values']:,}"
    )

    st.write(
        "Jumlah data setelah cleaning:",
        f"{audit['clean_rows']:,}"
    )

    st.write(
        "Jumlah data periode 2017:",
        f"{audit['data_2017_rows']:,}"
    )

    # MISSING VALUES
    st.subheader("2. Missing Values Setiap Kolom")

    missing_table = (
        audit["missing_values"]
        .reset_index()
    )

    missing_table.columns = [
        "Nama Kolom",
        "Jumlah Missing Values"
    ]

    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True
    )

    # DESCRIPTIVE STATISTICS
    st.subheader("3. Statistik Deskriptif")

    st.dataframe(
        filtered_data[
            ["price", "freight_value"]
        ].describe(),
        use_container_width=True
    )

    # DATA PREVIEW
    st.subheader("4. Preview Dataset")

    st.dataframe(
        filtered_data.head(500),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Preview menampilkan maksimal 500 baris."
    )

    # DOWNLOAD
    st.subheader("5. Download Data")

    csv = filtered_data.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Download Data Hasil Analisis",
        data=csv,
        file_name="ecommerce_analysis_2017.csv",
        mime="text/csv"
    )


# =====================================================
# TAB 4 - CONCLUSION & RECOMMENDATION
# =====================================================

with tab4:

    st.header("Conclusion & Recommendation")

    st.markdown(
        "Kesimpulan berikut dihitung berdasarkan data "
        "dan filter yang sedang dipilih."
    )

    st.divider()

    # CONCLUSION PERTANYAAN 1
    st.subheader("1. Kesimpulan Analisis Pesanan")

    monthly_conclusion = (
        filtered_data.groupby("month")["order_id"]
        .nunique()
    )

    if not monthly_conclusion.empty:

        highest_month = monthly_conclusion.idxmax()
        lowest_month = monthly_conclusion.idxmin()

        st.write(
            f"Jumlah pesanan tertinggi terjadi pada "
            f"**{month_labels[str(highest_month)]}** "
            f"dengan total **{monthly_conclusion.max():,} pesanan**."
        )

        st.write(
            f"Jumlah pesanan terendah terjadi pada "
            f"**{month_labels[str(lowest_month)]}** "
            f"dengan total **{monthly_conclusion.min():,} pesanan**."
        )

        delivered = filtered_data[
            filtered_data["order_status"] == "delivered"
        ]

        if not delivered.empty:

            delivered_monthly = (
                delivered.groupby("month")["order_id"]
                .nunique()
            )

            st.write(
                f"Jumlah pesanan dengan status delivered "
                f"pada data terfilter adalah "
                f"**{delivered['order_id'].nunique():,} pesanan**."
            )

    # CONCLUSION PERTANYAAN 2
    st.divider()

    st.subheader("2. Kesimpulan Analisis Biaya Pengiriman")

    if not filtered_shipping_summary.empty:

        category_summary = (
            filtered_shipping_summary.groupby(
                "product_category_name"
            )
            .agg(
                avg_freight=("avg_freight", "mean"),
                avg_ratio=("freight_ratio", "mean")
            )
        )

        highest_freight_category = (
            category_summary["avg_freight"].idxmax()
        )

        highest_ratio_category = (
            category_summary["avg_ratio"].idxmax()
        )

        st.write(
            f"Kategori dengan rata-rata biaya pengiriman "
            f"tertinggi adalah **{highest_freight_category}**, "
            f"dengan rata-rata sebesar "
            f"**R$ {category_summary.loc[highest_freight_category, 'avg_freight']:.2f}**."
        )

        st.write(
            f"Kategori dengan rasio biaya pengiriman "
            f"tertinggi adalah **{highest_ratio_category}**, "
            f"dengan rasio sebesar "
            f"**{category_summary.loc[highest_ratio_category, 'avg_ratio']:.2f}%**."
        )

    # RECOMMENDATION
    st.divider()

    st.subheader("3. Rekomendasi Action Item")

    st.markdown("""
    **Berdasarkan hasil analisis, rekomendasi yang dapat diberikan:**

    1. Mempertahankan strategi penjualan yang efektif dan meningkatkan pemantauan status pesanan.

    2. Memperkuat koordinasi antara tim penjualan dan logistik agar peningkatan jumlah pesanan dapat diimbangi dengan kualitas pelayanan dan ketepatan pengiriman.

    3. Mengevaluasi biaya pengiriman pada kategori produk dengan biaya pengiriman tertinggi.

    4. Mempertimbangkan alternatif jasa ekspedisi yang lebih efisien.

    5. Memantau rasio biaya pengiriman terhadap harga produk agar biaya logistik tetap proporsional dan efisien.
    """)

    st.info(
        "Catatan: Kesimpulan akan berubah mengikuti filter bulan, "
        "status pesanan, dan kategori produk."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "E-Commerce Public Dataset | Fundamental Analisis Data | 2017"
)
