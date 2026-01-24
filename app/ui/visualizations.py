# visualizations.py - Power BI Style Interactive Charts with Animations
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np
import pygwalker as pyg
import streamlit.components.v1 as components
import os

# Configure Plotly for better performance and styling
pio.templates.default = "plotly_white"

def show_interactive_pygwalker(df):
    """
    PyGWalker interactive visualization - Main feature
    """
    st.subheader("🚀 Interactive Data Visualization (PyGWalker)")
    st.markdown("""
    **Drag and drop to create charts like Tableau!**
    - Drag columns to X/Y axis, Color, Size areas
    - Switch between chart types (bar, line, scatter, etc.)
    - Filter and explore your data interactively
    """)
    
    try:
        # Check if config exists, otherwise use default
        config_path = "./gw_config.json"
        
        # Generate PyGWalker HTML
        pyg_html = pyg.to_html(df, spec=config_path, use_kernel_calc=True)
        
        # Embed in Streamlit
        components.html(pyg_html, height=1000, scrolling=True)
        
    except Exception as e:
        st.error(f"Error loading PyGWalker: {str(e)}")
        st.info("💡 Make sure pygwalker is installed: `pip install pygwalker`")

def show_basic_statistics(df):
    """Enhanced statistics with interactive charts"""
    st.subheader("📈 Interactive Statistics Dashboard")
    
    # Create tabs for different statistical views
    tab1, tab2, tab3 = st.tabs(["📊 Descriptive Stats", "📈 Distribution Overview", "🔍 Missing Data Analysis"])
    
    with tab1:
        # Enhanced descriptive statistics
        stats_df = df.describe(include='all').T
        stats_df = stats_df.round(2)
        
        # Create an interactive table using Plotly
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Column</b>'] + [f'<b>{col}</b>' for col in stats_df.columns],
                fill_color='#1f77b4',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[stats_df.index] + [stats_df[col].values for col in stats_df.columns],
                fill_color=['#f0f2f6', '#ffffff'],
                font=dict(color='black', size=11),
                align='left',
                height=30
            )
        )])
        
        fig.update_layout(
            title="📋 Descriptive Statistics",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Distribution overview for numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Select column for distribution analysis:", numeric_cols, key="dist_col")
            
            # Create subplot with histogram and box plot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=[f'Distribution of {col}', f'Box Plot of {col}'],
                vertical_spacing=0.1
            )
            
            # Histogram with animations
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(),
                    nbinsx=30,
                    name='Distribution',
                    marker_color='#636EFA',
                    opacity=0.8,
                    hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Box plot
            fig.add_trace(
                go.Box(
                    y=df[col].dropna(),
                    name='Box Plot',
                    marker_color='#EF553B',
                    boxpoints='outliers',
                    hovertemplate='<b>Value:</b> %{y}<br><b>Quartile Info:</b> Available on hover<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text=f"📊 Statistical Analysis: {col}",
                animations=[
                    dict(
                        frame=dict(duration=500, redraw=True),
                        transition=dict(duration=300)
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            data = df[col].dropna()
            with col1:
                st.metric("📊 Mean", f"{data.mean():.2f}")
            with col2:
                st.metric("📈 Median", f"{data.median():.2f}")
            with col3:
                st.metric("📏 Std Dev", f"{data.std():.2f}")
            with col4:
                st.metric("📋 Count", f"{len(data):,}")
    
    with tab3:
        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        if missing_data.sum() > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=missing_data.index,
                y=missing_data.values,
                name='Missing Count',
                marker_color='#FF6B6B',
                hovertemplate='<b>Column:</b> %{x}<br><b>Missing:</b> %{y}<br><b>Percentage:</b> %{customdata:.1f}%<extra></extra>',
                customdata=missing_percent.values
            ))
            
            fig.update_layout(
                title='🚨 Missing Data Analysis',
                xaxis_title='Columns',
                yaxis_title='Missing Values Count',
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No missing data found in your dataset!")

def show_bar_chart_interactive(df):
    """Power BI style interactive bar chart with animations"""
    st.subheader("📊 Interactive Bar Chart")
    
    # Get categorical and numerical columns
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if 2 <= df[c].nunique() <= 25]
    num_cols = [c for c in df.select_dtypes(include=['int64', 'float64']).columns]
    
    if cat_cols:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            grouping_col = st.selectbox("Select Category Column", cat_cols)
            
        with col2:
            chart_type = st.selectbox(
                "Chart Type",
                ["Count of Records", "Sum of Values"] if num_cols else ["Count of Records"]
            )
            
        with col3:
            y_col = (
                st.selectbox("Select Value Column", num_cols)
                if chart_type == "Sum of Values" and num_cols
                else "Count"
            )

        # Additional styling options
        col1, col2, col3 = st.columns(3)
        with col1:
            orientation = st.selectbox("Chart Orientation", ["Vertical", "Horizontal"])
            
        with col2:
            color_scheme = st.selectbox(
                "Color Scheme", 
                ["viridis", "plasma", "inferno", "magma", "blues", "reds"]
            )
            
        with col3:
            animate = st.checkbox("Enable Animations", value=True)
            show_values = st.checkbox("Show Values", value=True)

        # Prepare data
        if chart_type == "Count of Records":
            data = df[grouping_col].value_counts().reset_index()
            data.columns = [grouping_col, 'Count']
            y_col = 'Count'
        else:
            data = df.groupby(grouping_col)[y_col].sum().reset_index()

        # Sort data for better visualization
        data = data.sort_values(y_col, ascending=False)

        # Create the chart
        if orientation == "Vertical":
            fig = px.bar(
                data,
                x=grouping_col,
                y=y_col,
                title=f"{chart_type} by {grouping_col}",
                color=y_col,
                color_continuous_scale=color_scheme,
                custom_data=[(data[y_col] / data[y_col].sum() * 100).round(1)]
            )
            
            # Update hover template
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>' +
                             f'<b>{y_col}:</b> %{{y:,.0f}}<br>' +
                             '<b>Percentage:</b> %{customdata[0]:.1f}%<extra></extra>'
            )
        else:
            fig = px.bar(
                data,
                x=y_col,
                y=grouping_col,
                title=f"{chart_type} by {grouping_col}",
                color=y_col,
                color_continuous_scale=color_scheme,
                orientation='h',
                custom_data=[(data[y_col] / data[y_col].sum() * 100).round(1)]
            )
            
            # Update hover template
            fig.update_traces(
                hovertemplate='<b>%{y}</b><br>' +
                             f'<b>{y_col}:</b> %{{x:,.0f}}<br>' +
                             '<b>Percentage:</b> %{customdata[0]:.1f}%<extra></extra>'
            )

        # Add animation if enabled
        if animate:
            fig.update_layout(
                transition_duration=500,
                transition={
                    'duration': 500,
                    'easing': 'cubic-in-out'
                }
            )

        # Show values on bars if enabled
        if show_values:
            if orientation == "Vertical":
                fig.update_traces(
                    texttemplate='%{y:,.0f}',
                    textposition='outside',
                    textfont=dict(size=12, color='black')
                )
            else:
                fig.update_traces(
                    texttemplate='%{x:,.0f}',
                    textposition='outside',
                    textfont=dict(size=12, color='black')
                )

        # Update layout
        fig.update_layout(
            showlegend=False,
            margin=dict(t=30, l=10, r=10, b=10)
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ No suitable categorical columns found for bar chart visualization.")
        st.info("💡 Categorical columns should have between 2 and 25 unique values.")

def show_pie_chart_interactive(df):
    """Power BI style interactive pie chart with animations"""
    st.subheader("🥧 Interactive Pie Chart")
    
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if 2 <= df[c].nunique() <= 10]
    num_cols = [c for c in df.select_dtypes(include=['int64', 'float64']).columns]
    
    if cat_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            grouping_col = st.selectbox("📊 Group by", cat_cols, key="pie_grouping_col")
            chart_type = st.radio("📈 Chart Type", ["Count of Records", "Sum of Values"], key="pie_chart_type")
        
        with col2:
            hole_size = st.slider("🕳️ Donut Hole Size", 0.0, 0.8, 0.3, key="pie_hole")
            show_percentages = st.checkbox("📊 Show Percentages", value=True, key="pie_percentages")
        
        # Prepare data
        if chart_type == "Sum of Values" and num_cols:
            sum_col = st.selectbox("📊 Sum column", num_cols, key="pie_sum_col")
            values = df.groupby(grouping_col)[sum_col].sum()
            title = f"Total {sum_col} by {grouping_col}"
        else:
            values = df[grouping_col].value_counts()
            title = f"Distribution of {grouping_col}"
        
        # Create enhanced pie chart with animations
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3[:len(values)]
        
        fig.add_trace(go.Pie(
            labels=values.index,
            values=values.values,
            hole=hole_size,
            marker=dict(
                colors=colors,
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent' if show_percentages else 'label',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>' +
                         'Value: %{value:,.0f}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>',
            pull=[0.1 if i == values.argmax() else 0 for i in range(len(values))]  # Pull out the largest slice
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            height=500,
            font=dict(family="Arial, sans-serif", size=12),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            ),
            annotations=[
                dict(
                    text=f'Total<br>{values.sum():,}',
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False,
                    font_color="black",
                    font_family="Arial"
                )
            ] if hole_size > 0 else [],
            margin=dict(t=80, b=50, l=50, r=150)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown table
        st.subheader("📋 Detailed Breakdown")
        breakdown_df = pd.DataFrame({
            'Category': values.index,
            'Value': values.values,
            'Percentage': (values.values / values.sum() * 100).round(2)
        }).sort_values('Value', ascending=False)
        
        # Create interactive table
        fig_table = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Category</b>', '<b>Value</b>', '<b>Percentage</b>'],
                fill_color='#1f77b4',
                font=dict(color='white', size=12),
                align='left'
            ),
            cells=dict(
                values=[
                    breakdown_df['Category'],
                    [f"{val:,}" for val in breakdown_df['Value']],
                    [f"{val}%" for val in breakdown_df['Percentage']]
                ],
                fill_color=['#f0f2f6', '#ffffff', '#f0f2f6'],
                font=dict(color='black', size=11),
                align='left',
                height=30
            )
        )])
        
        fig_table.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_table, use_container_width=True)
        
    else:
        st.info("No suitable categorical columns for pie chart (max 10 categories).")

def show_histogram_interactive(df):
    """Power BI style interactive histogram with statistical overlays"""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 0:
        st.subheader("📊 Interactive Histogram with Statistical Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            col = st.selectbox("📊 Choose numeric column", num_cols, key="hist_column_selector")
        with col2:
            bins = st.slider("🔢 Number of bins", min_value=10, max_value=100, value=30, key="hist_bins")
        with col3:
            show_stats = st.checkbox("📈 Show Statistics", value=True, key="hist_stats")
        
        data = df[col].dropna()
        
        # Create subplot with histogram and additional statistical plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'Distribution of {col}',
                f'Box Plot',
                f'Q-Q Plot vs Normal',
                f'Cumulative Distribution'
            ],
            specs=[[{"colspan": 2}, None],
                   [{}, {}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Main histogram with overlays
        fig.add_trace(
            go.Histogram(
                x=data,
                nbinsx=bins,
                name='Distribution',
                marker_color='rgba(99, 110, 250, 0.7)',
                marker_line=dict(color='white', width=1),
                hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<br><b>Density:</b> %{customdata:.3f}<extra></extra>',
                customdata=np.histogram(data, bins=bins)[0] / len(data)
            ),
            row=1, col=1
        )
        
        if show_stats:
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            
            # Add mean line
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top",
                row=1, col=1
            )
            
            # Add median line
            fig.add_vline(
                x=median_val,
                line_dash="dot",
                line_color="green",
                line_width=2,
                annotation_text=f"Median: {median_val:.2f}",
                annotation_position="top left",
                row=1, col=1
            )
            
            # Add standard deviation bands
            fig.add_vrect(
                x0=mean_val - std_val, x1=mean_val + std_val,
                fillcolor="yellow", opacity=0.2,
                annotation_text="±1σ",
                annotation_position="top left",
                row=1, col=1
            )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=data,
                name='Box Plot',
                marker_color='rgba(239, 85, 59, 0.7)',
                boxpoints='outliers',
                hovertemplate='<b>Value:</b> %{y}<br><extra></extra>'
            ),
            row=2, col=1
        )
        
        # Q-Q plot
        from scipy import stats
        qq_data = stats.probplot(data, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color='purple', size=6),
                hovertemplate='<b>Theoretical:</b> %{x:.2f}<br><b>Sample:</b> %{y:.2f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Add Q-Q reference line
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                mode='lines',
                name='Q-Q Line',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"📊 Statistical Analysis: {col}",
            title_x=0.5,
            title_font_size=20,
            showlegend=False,
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced metrics with statistical tests
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Mean", f"{data.mean():.3f}")
        with col2:
            st.metric("📈 Median", f"{data.median():.3f}")
        with col3:
            st.metric("📏 Std Dev", f"{data.std():.3f}")
        with col4:
            st.metric("📐 Skewness", f"{data.skew():.3f}")
        with col5:
            st.metric("📋 Count", f"{len(data):,}")
        
        # Additional statistical information
        st.subheader("📈 Statistical Tests & Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            # Normality test
            shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
            st.metric("🧪 Shapiro-Wilk p-value", f"{shapiro_p:.4f}")
            if shapiro_p > 0.05:
                st.success("✅ Data appears normally distributed")
            else:
                st.warning("⚠️ Data may not be normally distributed")
        
        with col2:
            # Outlier detection
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]
            st.metric("🚨 Outliers Detected", len(outliers))
            if len(outliers) > 0:
                st.warning(f"⚠️ {len(outliers)/len(data)*100:.1f}% of data are outliers")
            else:
                st.success("✅ No significant outliers detected")
            
    else:
        st.info("No numeric columns available for histogram.")

def show_scatter_plot_interactive(df):
    """Power BI style interactive scatter plot with trend analysis"""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(num_cols) >= 2:
        st.subheader("🔵 Interactive Scatter Plot with Trend Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x = st.selectbox("📊 X-axis", num_cols, index=0, key="scatter_x_selector")
        with col2:
            y = st.selectbox("📊 Y-axis", num_cols, index=1 if len(num_cols)>1 else 0, key="scatter_y_selector")
        with col3:
            color_by = st.selectbox("🎨 Color by", ["None"] + list(cat_cols), key="scatter_color")
        with col4:
            size_by = st.selectbox("📏 Size by", ["None"] + list(num_cols), key="scatter_size")
        
        # Additional options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_trendline = st.checkbox("📈 Show Trendline", value=True, key="scatter_trend")
        with col2:
            show_marginals = st.checkbox("📊 Show Marginal Plots", value=False, key="scatter_marginals")
        with col3:
            opacity = st.slider("🎭 Point Opacity", 0.1, 1.0, 0.7, key="scatter_opacity")
        
        # Create the scatter plot
        kwargs = {
            'data_frame': df,
            'x': x,
            'y': y,
            'opacity': opacity,
            'hover_data': {col: True for col in num_cols[:5]}  # Show top 5 numeric columns in hover
        }
        
        if color_by != "None":
            kwargs['color'] = color_by
            kwargs['color_discrete_sequence'] = px.colors.qualitative.Set3
        
        if size_by != "None":
            kwargs['size'] = size_by
            kwargs['size_max'] = 20
        
        if show_trendline:
            kwargs['trendline'] = 'ols'
            kwargs['trendline_color_override'] = 'red'
        
        if show_marginals:
            if color_by != "None":
                fig = px.scatter(marginal_x="histogram", marginal_y="histogram", **kwargs)
            else:
                fig = px.scatter(marginal_x="histogram", marginal_y="histogram", **kwargs)
        else:
            fig = px.scatter(**kwargs)
        
        # Enhanced styling and hover information
        fig.update_traces(
            hovertemplate='<b>' + x + ':</b> %{x}<br>' +
                         '<b>' + y + ':</b> %{y}<br>' +
                         ('<b>' + color_by + ':</b> %{customdata[0]}<br>' if color_by != "None" else '') +
                         '<extra></extra>',
            marker=dict(
                line=dict(width=0.8, color='white')
            )
        )
        
        fig.update_layout(
            title=f'📊 Scatter Plot: {x} vs {y}',
            title_x=0.5,
            title_font_size=20,
            height=600,
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(248,249,250,0.8)',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation and regression analysis
        correlation = df[x].corr(df[y])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Correlation", f"{correlation:.4f}")
        with col2:
            r_squared = correlation ** 2
            st.metric("📊 R² Score", f"{r_squared:.4f}")
        with col3:
            if abs(correlation) > 0.7:
                st.metric("💪 Relationship", "Strong", delta="High")
            elif abs(correlation) > 0.3:
                st.metric("📊 Relationship", "Moderate", delta="Medium")
            else:
                st.metric("📉 Relationship", "Weak", delta="Low")
        with col4:
            st.metric("📋 Data Points", f"{len(df):,}")
        
        # Show regression equation if trendline is enabled
        if show_trendline:
            from sklearn.linear_model import LinearRegression
            X = df[[x]].dropna()
            y_values = df[y].dropna()
            
            # Ensure same length
            min_len = min(len(X), len(y_values))
            X = X.iloc[:min_len]
            y_values = y_values.iloc[:min_len]
            
            reg = LinearRegression().fit(X, y_values)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            
            st.info(f"📈 **Regression Equation:** {y} = {slope:.4f} × {x} + {intercept:.4f}")
            
    else:
        st.info("Need at least two numeric columns for scatter plot.")

def show_correlation_heatmap_interactive(df):
    """Power BI style interactive correlation heatmap with animations"""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) >= 2:
        st.subheader("🔥 Interactive Correlation Heatmap")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            colormap = st.selectbox("🎨 Color Scale", 
                                  ["RdBu_r", "Viridis", "Plasma", "Cividis", "RdYlBu_r", "Spectral_r"], 
                                  key="heatmap_colormap")
        
        with col2:
            show_values = st.checkbox("🔢 Show correlation values", value=True, key="heatmap_values")
        
        with col3:
            cluster_map = st.checkbox("🔄 Cluster similar variables", value=False, key="heatmap_cluster")
        
        corr = df[num_cols].corr()
        
        # Clustering if requested
        if cluster_map and len(num_cols) > 3:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
            
            # Convert correlation to distance and cluster
            distance = 1 - np.abs(corr)
            linkage_matrix = linkage(squareform(distance), method='average')
            dendro = dendrogram(linkage_matrix, labels=corr.columns, no_plot=True)
            reorder_idx = dendro['leaves']
            
            corr = corr.iloc[reorder_idx, reorder_idx]
        
        # Create interactive heatmap
        fig = go.Figure()
        
        # Custom hover template with detailed information
        hover_text = []
        for i in range(len(corr.columns)):
            hover_row = []
            for j in range(len(corr.columns)):
                correlation_val = corr.iloc[i, j]
                strength = ""
                if abs(correlation_val) > 0.8:
                    strength = "Very Strong"
                elif abs(correlation_val) > 0.6:
                    strength = "Strong"
                elif abs(correlation_val) > 0.4:
                    strength = "Moderate"
                elif abs(correlation_val) > 0.2:
                    strength = "Weak"
                else:
                    strength = "Very Weak"
                
                hover_row.append(
                    f'<b>{corr.columns[j]} vs {corr.columns[i]}</b><br>' +
                    f'Correlation: {correlation_val:.4f}<br>' +
                    f'Strength: {strength}<br>' +
                    f'Direction: {"Positive" if correlation_val > 0 else "Negative"}'
                )
            hover_text.append(hover_row)
        
        fig.add_trace(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=colormap,
            zmid=0,
            text=corr.values.round(3) if show_values else None,
            texttemplate='%{text}' if show_values else None,
            textfont=dict(size=10, color='white'),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text,
            colorbar=dict(
                title="Correlation<br>Coefficient",
                titleside="right",
                tickmode="linear",
                tick0=-1,
                dtick=0.5
            )
        ))
        
        fig.update_layout(
            title='🔥 Correlation Matrix Analysis',
            title_x=0.5,
            title_font_size=20,
            height=max(500, len(corr.columns) * 40),
            width=max(600, len(corr.columns) * 40),
            font=dict(family="Arial, sans-serif", size=12),
            xaxis=dict(side='bottom', tickangle=45),
            yaxis=dict(side='left'),
            margin=dict(l=100, r=100, t=80, b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced correlation analysis
        st.subheader("🏆 Correlation Insights")
        
        # Get correlation pairs (excluding self-correlations)
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_pairs.append({
                    'Variable 1': corr.columns[i],
                    'Variable 2': corr.columns[j],
                    'Correlation': corr.iloc[i, j],
                    'Abs_Correlation': abs(corr.iloc[i, j]),
                    'Strength': 'Very Strong' if abs(corr.iloc[i, j]) > 0.8 else
                               'Strong' if abs(corr.iloc[i, j]) > 0.6 else
                               'Moderate' if abs(corr.iloc[i, j]) > 0.4 else
                               'Weak' if abs(corr.iloc[i, j]) > 0.2 else 'Very Weak'
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🏆 Strongest Correlations", "📊 Correlation Distribution", "🔍 Variable Analysis"])
        
        with tab1:
            # Interactive table with top correlations
            top_corr = corr_df.head(10)
            
            fig_table = go.Figure(data=[go.Table(
                header=dict(
                    values=['<b>Variable 1</b>', '<b>Variable 2</b>', '<b>Correlation</b>', '<b>Strength</b>'],
                    fill_color='#1f77b4',
                    font=dict(color='white', size=12),
                    align='left',
                    height=40
                ),
                cells=dict(
                    values=[
                        top_corr['Variable 1'],
                        top_corr['Variable 2'],
                        [f"{val:.4f}" for val in top_corr['Correlation']],
                        top_corr['Strength']
                    ],
                    fill_color=[['#f0f2f6' if i % 2 == 0 else '#ffffff' for i in range(len(top_corr))] * 4],
                    font=dict(color='black', size=11),
                    align='left',
                    height=35
                )
            )])
            
            fig_table.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_table, use_container_width=True)
        
        with tab2:
            # Distribution of correlation values
            fig_dist = go.Figure()
            
            fig_dist.add_trace(go.Histogram(
                x=corr_df['Correlation'],
                nbinsx=20,
                name='Correlation Distribution',
                marker_color='skyblue',
                marker_line=dict(color='black', width=1),
                hovertemplate='<b>Correlation Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
            ))
            
            fig_dist.update_layout(
                title='📊 Distribution of Correlation Coefficients',
                xaxis_title='Correlation Coefficient',
                yaxis_title='Frequency',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Mean Correlation", f"{corr_df['Correlation'].mean():.4f}")
            with col2:
                st.metric("📈 Max Correlation", f"{corr_df['Correlation'].max():.4f}")
            with col3:
                st.metric("📉 Min Correlation", f"{corr_df['Correlation'].min():.4f}")
            with col4:
                strong_corr = len(corr_df[corr_df['Abs_Correlation'] > 0.6])
                st.metric("💪 Strong Correlations", strong_corr)
        
        with tab3:
            # Variable-wise analysis
            selected_var = st.selectbox("🔍 Analyze Variable:", corr.columns, key="var_analysis")
            
            var_corr = corr[selected_var].drop(selected_var).abs().sort_values(ascending=False)
            
            fig_var = go.Figure()
            
            fig_var.add_trace(go.Bar(
                x=var_corr.index,
                y=var_corr.values,
                name=f'Correlations with {selected_var}',
                marker_color='lightcoral',
                hovertemplate='<b>%{x}</b><br>Absolute Correlation: %{y:.4f}<extra></extra>'
            ))
            
            fig_var.update_layout(
                title=f'🔍 Variables Most Correlated with {selected_var}',
                xaxis_title='Variables',
                yaxis_title='Absolute Correlation',
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_var, use_container_width=True)
        
    else:
        st.info("Not enough numeric columns for correlation heatmap (minimum 2 required).")

def visualizations_sidebar(df):
    """
    Main entry point for visualizations.
    Renders a selector to choose between different visualization types.
    """
    st.markdown("### 📊 Choose Visualization")
    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "Interactive Dashboard (PyGWalker)",
            "Basic Statistics",
            "Bar Chart",
            "Pie Chart", 
            "Histogram",
            "Scatter Plot",
            "Correlation Heatmap"
        ],
        key="viz_selector_sidebar"
    )
    
    st.divider()
    
    if viz_type == "Interactive Dashboard (PyGWalker)":
        show_interactive_pygwalker(df)
    elif viz_type == "Basic Statistics":
        show_basic_statistics(df)
    elif viz_type == "Bar Chart":
        show_bar_chart_interactive(df)
    elif viz_type == "Pie Chart":
        show_pie_chart_interactive(df)
    elif viz_type == "Histogram":
        show_histogram_interactive(df)
    elif viz_type == "Scatter Plot":
        show_scatter_plot_interactive(df)
    elif viz_type == "Correlation Heatmap":
        show_correlation_heatmap_interactive(df)
