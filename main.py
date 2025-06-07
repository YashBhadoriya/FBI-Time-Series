import os
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import random

# Page configuration
st.set_page_config(
    page_title="FBI Time Series & AI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    [data-testid="metric-container"] * {
        color: #000000 !important;
        font-weight: bold !important;
    }
    .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .chat-message.user { background-color: #2b313e; color: white; }
    .chat-message.bot { background-color: #475063; color: white; }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data' not in st.session_state:
    st.session_state.data = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

# Load sample data
def load_sample_data():
    data = {
        'YEAR': [2013] * 54 + [2012] * 108,
        'MONTH': ([6,5,4,3,2,1] * 9 + [12,11,10,9,8,7,6,5,4,3,2,1] * 9),
        'TYPE': ['Vehicle Collision or Pedestrian Struck (with Injury)', 'Theft of Vehicle',
                 'Theft of Bicycle', 'Theft from Vehicle', 'Other Theft',
                 'Offence Against a Person', 'Mischief', 'Break and Enter Residential/Other',
                 'Break and Enter Commercial'] * 18,
        'Incident_Counts': [random.randint(10, 100) for _ in range(162)]
    }
    return pd.DataFrame(data)

# Get AI response
def get_ai_response(question, data_summary):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.session_state.api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a data analyst assistant. {data_summary}"},
                {"role": "user", "content": question}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Summary statistics
def create_summary_stats(df):
    df = df.copy()
    df['Incident_Counts'] = pd.to_numeric(df['Incident_Counts'], errors='coerce').fillna(0)
    total_incidents = int(df['Incident_Counts'].sum())
    monthly_data = df.groupby(['YEAR', 'MONTH'])['Incident_Counts'].sum()
    avg_monthly = monthly_data.mean()
    crime_totals = df.groupby('TYPE')['Incident_Counts'].sum()
    most_common_crime = crime_totals.idxmax() if not crime_totals.empty else "No Data"
    peak_month = df.groupby('MONTH')['Incident_Counts'].sum().idxmax()
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    return {
        'total_incidents': total_incidents,
        'avg_monthly': round(avg_monthly, 1),
        'most_common_crime': most_common_crime,
        'peak_month': month_names.get(peak_month, str(peak_month))
    }

# Create charts
def create_charts(df):
    charts = {}
    crime_counts = df.groupby('TYPE')['Incident_Counts'].sum().reset_index()
    charts['crime_types'] = px.bar(crime_counts, x='Incident_Counts', y='TYPE', orientation='h', title="Crime Types Distribution")

    monthly_data = df.groupby(['YEAR', 'MONTH'])['Incident_Counts'].sum().reset_index()
    monthly_data['Date'] = pd.to_datetime(monthly_data[['YEAR', 'MONTH']].assign(DAY=1))
    charts['monthly_trends'] = px.line(monthly_data, x='Date', y='Incident_Counts', title="Monthly Crime Trends")

    yearly_data = df.groupby('YEAR')['Incident_Counts'].sum().reset_index()
    charts['yearly_comparison'] = px.bar(yearly_data, x='YEAR', y='Incident_Counts', title="Yearly Crime Comparison")

    heatmap_data = df.groupby(['MONTH', 'TYPE'])['Incident_Counts'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='TYPE', columns='MONTH', values='Incident_Counts')
    charts['heatmap'] = px.imshow(heatmap_pivot, title="Crime Types by Month Heatmap", aspect="auto")

    top_crimes = df.groupby('TYPE')['Incident_Counts'].sum().nlargest(5).reset_index()
    charts['pie_chart'] = px.pie(top_crimes, values='Incident_Counts', names='TYPE', title="Top 5 Crime Types")
    return charts

# Main app
def main():
    st.title("🚔 Crime Data Dashboard & AI Assistant")

    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        st.session_state.api_key = st.text_input("OpenAI API Key", type="password") or st.session_state.api_key
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'Incident_Counts' not in df.columns:
                    st.error("Missing 'Incident_Counts' column.")
                    return
                df['Incident_Counts'] = pd.to_numeric(df['Incident_Counts'], errors='coerce').fillna(0)
                st.session_state.data = df
                st.success("Data uploaded!")
            except Exception as e:
                st.error(f"Upload error: {e}")
        else:
            st.session_state.data = load_sample_data()
            st.info("Using sample data")

        if st.session_state.data is not None:
            years = st.multiselect("Filter by Year", sorted(st.session_state.data['YEAR'].unique()),
                                   default=sorted(st.session_state.data['YEAR'].unique()))
            crime_types = st.multiselect("Filter by Crime Type", sorted(st.session_state.data['TYPE'].unique()),
                                         default=sorted(st.session_state.data['TYPE'].unique()))

    if st.session_state.data is not None:
        df = st.session_state.data
        filtered_df = df[df['YEAR'].isin(years) & df['TYPE'].isin(crime_types)]

        if filtered_df.empty:
            st.warning("No data for selected filters.")
            return

        # Summary stats
        stats = create_summary_stats(filtered_df)
        st.header("📈 Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Incidents", f"{stats['total_incidents']:,}")
        col2.metric("Avg Monthly", f"{stats['avg_monthly']:.1f}")
        col3.metric("Most Common Crime", stats['most_common_crime'][:30] + "...")
        col4.metric("Peak Month", stats['peak_month'])

        # Charts
        charts = create_charts(filtered_df)
        st.header("📊 Visualizations")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Crime Types", "Trends", "Yearly", "Heatmap", "Top Crimes"])
        tab1.plotly_chart(charts['crime_types'], use_container_width=True)
        tab2.plotly_chart(charts['monthly_trends'], use_container_width=True)
        tab3.plotly_chart(charts['yearly_comparison'], use_container_width=True)
        tab4.plotly_chart(charts['heatmap'], use_container_width=True)
        tab5.plotly_chart(charts['pie_chart'], use_container_width=True)

        # AI Assistant
        st.header("🤖 AI Assistant")
        question = st.text_input("Ask a question about your crime data")
        if st.button("🚀 Ask AI") and st.session_state.api_key:
            summary_text = (
                f"Data Summary:\n"
                f"- Total Incidents: {stats['total_incidents']}\n"
                f"- Avg Monthly: {stats['avg_monthly']}\n"
                f"- Most Common Crime: {stats['most_common_crime']}\n"
                f"- Peak Month: {stats['peak_month']}\n"
                f"- Years: {sorted(filtered_df['YEAR'].unique())}\n"
                f"- Crime Types: {filtered_df['TYPE'].nunique()}"
            )
            with st.spinner("Thinking..."):
                answer = get_ai_response(question, summary_text)
            st.session_state.chat_history.append({"q": question, "a": answer})
        elif question and not st.session_state.api_key:
            st.warning("Please enter your OpenAI API key.")

        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            for chat in reversed(st.session_state.chat_history[-5:]):
                st.write(f"**Q:** {chat['q']}")
                st.write(f"**A:** {chat['a']}")

        # Export
        st.header("💾 Export")
        csv = filtered_df.to_csv(index=False)
        st.download_button("⬇️ Download Filtered Data", data=csv, file_name="filtered_crime_data.csv", mime="text/csv")

        report = (
            f"Crime Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Total Incidents: {stats['total_incidents']}\n"
            f"Average Monthly: {stats['avg_monthly']}\n"
            f"Most Common Crime: {stats['most_common_crime']}\n"
            f"Peak Month: {stats['peak_month']}\n"
            f"Years Analyzed: {', '.join(map(str, sorted(filtered_df['YEAR'].unique())))}\n"
            f"Crime Types: {filtered_df['TYPE'].nunique()}"
        )
        st.download_button("⬇️ Download Report", data=report, file_name="crime_report.txt", mime="text/plain")

if __name__ == "__main__":
    main()
