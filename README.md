# Market Basket Recommendation UI

This project implements a **Market Basket Analysis** model using the **Apriori algorithm**, with a user-friendly interface built on **Streamlit**. It enables quick identification of frequently co-purchased services and suggests optimal service bundles to users.

## Features

* **Apriori-based Model**

  * Extracts frequent itemsets from transactional data.
  * Generates association rules for service recommendations.
* **Streamlit UI**

  * Interactive dashboard to input or upload transaction data.
  * Instant visualization of frequent itemsets and recommended bundles.

## Future Roadmap

We plan to integrate the following AI modules in upcoming releases:

1. **Live Audio Chat Support**

   * Real-time speech-to-text and AI-driven response suggestions during Zoom calls.
2. **Automated Offers via Email/WhatsApp**

   * Lead scoring and personalized bundle outreach through email and WhatsApp APIs.

## Prerequisites

* Python 3.8 or higher
* pip (Python package manager)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/market-basket-ui.git
   cd market-basket-ui
   ```
2. Navigate to the market basket module and install dependencies:

   ```bash
   cd market_basket_model
   pip install -r requirements.txt
   ```

## Running the Market Basket UI

To launch the Streamlit app for the Market Basket model:

```bash
cd market_basket_model
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`.

## Directory Structure

```
market-basket-ui/
├── market_basket_model/
│   ├── app.py              # Streamlit application
│   ├── market_basket.ipynb   # Apriori algorithm implementation
│   ├── fnancial_transactions.csv          # Sample csv file 
│   ├── requirements.txt    # Python dependencies
│   └── data/               # Sample transaction datasets
├── README.md               # This file
└── .gitignore
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, feature requests, or improvements.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Developed by Syntax Terminators*
