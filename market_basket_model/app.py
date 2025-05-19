import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import io
import base64
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        background-color: #f9f9f9;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f8ff;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Market Basket Analysis Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2331/2331966.png", width=100)
st.sidebar.header("Configuration")

# Load and prepare data
@st.cache_data
def load_and_prepare_data(uploaded_file):
    """
    Load transaction data from uploaded file and prepare it for analysis
    """
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # If the Items column contains string representations of lists, convert them to actual lists
        try:
            if isinstance(df['Items'].iloc[0], str):
                df['Items'] = df['Items'].apply(lambda x: ast.literal_eval(x))
        except:
            st.error("Error: Could not parse 'Items' column. Make sure it contains valid lists.")
            return None
        
        # Extract transactions as a list of lists
        transactions = df['Items'].tolist()
        
        return transactions, df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Perform market basket analysis
@st.cache_data
def market_basket_analysis(transactions, min_support=0.01):
    """
    Perform market basket analysis using Apriori algorithm
    """
    # Encode the transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Apply Apriori algorithm
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=4)
    
    if frequent_itemsets.empty:
        return None
    
    # Add a column for itemset size
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    
    return frequent_itemsets

# Analyze itemsets by size
def analyze_itemsets_by_size(frequent_itemsets):
    """
    Analyze frequent itemsets by size (2, 3, and 4 items)
    """
    # Get itemsets of different sizes
    pairs = frequent_itemsets[frequent_itemsets['length'] == 2].copy()
    triplets = frequent_itemsets[frequent_itemsets['length'] == 3].copy()
    quadruplets = frequent_itemsets[frequent_itemsets['length'] == 4].copy()
    
    # Convert frozensets to strings for better display
    for df in [pairs, triplets, quadruplets]:
        if not df.empty:
            df['items_str'] = df['itemsets'].apply(lambda x: ', '.join(list(x)))
    
    return pairs, triplets, quadruplets

# Generate association rules
@st.cache_data
def generate_association_rules(frequent_itemsets, min_confidence=0.5):
    """
    Generate association rules from frequent itemsets
    """
    if frequent_itemsets is None or frequent_itemsets.empty:
        return None
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    if rules.empty:
        return None
    
    # Add rule size information
    rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))
    rules['consequent_len'] = rules['consequents'].apply(lambda x: len(x))
    rules['rule_len'] = rules['antecedent_len'] + rules['consequent_len']
    
    # Convert frozensets to strings for better display
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules['rule_str'] = rules['antecedents_str'] + ' → ' + rules['consequents_str']
    
    return rules

# Create downloadable CSV link
def get_csv_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# Create figure for itemsets
def create_itemsets_figure(itemsets_df, title, top_n=10):
    if itemsets_df.empty:
        return None
    
    top_itemsets = itemsets_df.sort_values('support', ascending=False).head(top_n)
    
    fig = px.bar(
        top_itemsets,
        y='items_str',
        x='support',
        orientation='h',
        title=f'Top {top_n} {title}',
        labels={'support': 'Support (proportion of transactions)', 'items_str': 'Itemsets'},
        color='support',
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    fig.update_layout(
        height=max(300, len(top_itemsets) * 30),
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

# Create figure for rules
def create_rules_figure(rules_df, metric='confidence', top_n=10):
    if rules_df.empty:
        return None
    
    top_rules = rules_df.sort_values(metric, ascending=False).head(top_n)
    
    fig = px.scatter(
        top_rules,
        x='support',
        y='confidence',
        size='lift',
        hover_name='rule_str',
        color='lift',
        color_continuous_scale=px.colors.sequential.Viridis,
        title=f'Top {top_n} Rules by {metric.capitalize()}',
        labels={
            'support': 'Support',
            'confidence': 'Confidence',
            'lift': 'Lift'
        }
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Main application
def main():
    # Upload file section
    st.sidebar.markdown("<div class='sub-header'>Data Input</div>", unsafe_allow_html=True)
    
    upload_option = st.sidebar.radio(
        "Choose data input method:",
        ("Upload CSV", "Use sample data")
    )
    
    if upload_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is None:
            st.info("Please upload a CSV file with a column named 'Items' containing lists of items.")
            return
    else:
        # Use sample data
        sample_data = """Items
["bread", "milk", "eggs"]
["bread", "butter", "jam"]
["milk", "butter", "eggs"]
["bread", "milk", "butter", "eggs"]
["bread", "milk", "cereal"]
["milk", "eggs", "cereal"]
["bread", "eggs", "cereal"]
["bread", "milk", "eggs", "jam"]
["butter", "jam", "cereal"]
["bread", "butter", "cereal"]"""
        uploaded_file = io.StringIO(sample_data)
        st.sidebar.success("Using sample data")
    
    # Parameters
    st.sidebar.markdown("<div class='sub-header'>Analysis Parameters</div>", unsafe_allow_html=True)
    
    min_support = st.sidebar.slider(
        "Minimum Support",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Minimum support threshold for frequent itemsets"
    )
    
    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence threshold for association rules"
    )
    
    # Load and process data
    transactions, raw_df = load_and_prepare_data(uploaded_file)
    
    if transactions is None:
        return
    
    # Data overview
    with st.expander("Data Overview", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Transaction Statistics")
            st.metric("Total Transactions", len(transactions))
            
            # Extract all items
            all_items = [item for sublist in transactions for item in sublist]
            unique_items = set(all_items)
            st.metric("Unique Items", len(unique_items))
            
            # Transaction length stats
            transaction_lengths = [len(t) for t in transactions]
            st.metric("Avg. Items per Transaction", f"{np.mean(transaction_lengths):.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Top 10 Items")
            item_counts = pd.Series(all_items).value_counts().head(10)
            
            # Create bar chart for top items
            fig = px.bar(
                x=item_counts.values,
                y=item_counts.index,
                orientation='h',
                labels={'x': 'Count', 'y': 'Item'},
                color=item_counts.values,
                color_continuous_scale=px.colors.sequential.Greens
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Run market basket analysis
    with st.spinner("Running Apriori algorithm..."):
        frequent_itemsets = market_basket_analysis(transactions, min_support)
    
    if frequent_itemsets is None or frequent_itemsets.empty:
        st.warning(f"No frequent itemsets found with support >= {min_support}. Try lowering the minimum support.")
        return
    
    # Analyze itemsets by size
    pairs, triplets, quadruplets = analyze_itemsets_by_size(frequent_itemsets)
    
    # Display metrics
    st.markdown("<h2 class='sub-header'>Frequent Itemsets Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Item Pairs (Size 2)", len(pairs))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Item Triplets (Size 3)", len(triplets))
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Item Quadruplets (Size 4)", len(quadruplets))
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Item Pairs", "Item Triplets", "Item Quadruplets", "Association Rules"])
    
    with tab1:
        if len(pairs) > 0:
            st.subheader("Frequent Item Pairs")
            
            # Visualization
            fig = create_itemsets_figure(pairs, "Frequent Item Pairs")
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            with st.expander("View Data Table"):
                st.dataframe(
                    pairs[['items_str', 'support']].sort_values('support', ascending=False),
                    use_container_width=True
                )
                st.markdown(get_csv_download_link(pairs[['items_str', 'support']], "frequent_pairs.csv"), unsafe_allow_html=True)
        else:
            st.info("No frequent pairs found with the current support threshold.")
    
    with tab2:
        if len(triplets) > 0:
            st.subheader("Frequent Item Triplets")
            
            # Visualization
            fig = create_itemsets_figure(triplets, "Frequent Item Triplets")
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            with st.expander("View Data Table"):
                st.dataframe(
                    triplets[['items_str', 'support']].sort_values('support', ascending=False),
                    use_container_width=True
                )
                st.markdown(get_csv_download_link(triplets[['items_str', 'support']], "frequent_triplets.csv"), unsafe_allow_html=True)
        else:
            st.info("No frequent triplets found with the current support threshold.")
    
    with tab3:
        if len(quadruplets) > 0:
            st.subheader("Frequent Item Quadruplets")
            
            # Visualization
            fig = create_itemsets_figure(quadruplets, "Frequent Item Quadruplets")
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            with st.expander("View Data Table"):
                st.dataframe(
                    quadruplets[['items_str', 'support']].sort_values('support', ascending=False),
                    use_container_width=True
                )
                st.markdown(get_csv_download_link(quadruplets[['items_str', 'support']], "frequent_quadruplets.csv"), unsafe_allow_html=True)
        else:
            st.info("No frequent quadruplets found with the current support threshold.")
    
    with tab4:
        st.subheader("Association Rules")
        
        # Generate rules
        with st.spinner("Generating association rules..."):
            rules = generate_association_rules(frequent_itemsets, min_confidence)
        
        if rules is not None and not rules.empty:
            # Metrics for rules
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Total Rules", len(rules))
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Avg. Confidence", f"{rules['confidence'].mean():.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Avg. Lift", f"{rules['lift'].mean():.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Filter options
            st.markdown("### Filter Rules")
            
            col1, col2 = st.columns(2)
            
            with col1:
                min_lift = st.slider(
                    "Minimum Lift",
                    min_value=1.0,
                    max_value=float(max(3.0, rules['lift'].max())),
                    value=1.0,
                    step=0.1
                )
            
            with col2:
                sort_by = st.selectbox(
                    "Sort Rules By",
                    options=["confidence", "lift", "support"]
                )
            
            # Apply filters
            filtered_rules = rules[rules['lift'] >= min_lift].sort_values(sort_by, ascending=False)
            
            if filtered_rules.empty:
                st.warning("No rules match the current filters.")
            else:
                # Bubble chart for rules
                fig = create_rules_figure(filtered_rules, sort_by)
                st.plotly_chart(fig, use_container_width=True)
                
                # Table view
                with st.expander("View Rules Table"):
                    st.dataframe(
                        filtered_rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].sort_values(sort_by, ascending=False),
                        use_container_width=True
                    )
                    st.markdown(get_csv_download_link(filtered_rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']], "association_rules.csv"), unsafe_allow_html=True)
        else:
            st.warning(f"No association rules found with confidence >= {min_confidence}. Try lowering the minimum confidence.")
    
    # Footer
    st.markdown("---")
    st.markdown("Market Basket Analysis Dashboard ● Created with Streamlit ● 🛒")

if __name__ == "__main__":
    main()


    