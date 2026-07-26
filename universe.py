"""Curated liquid U.S. large-cap research universe.

The universe is intentionally explicit and version-controlled. It is not
presented as the membership of a commercial index.
"""

UNIVERSE = {
    # Technology
    "AAPL": ("Apple", "Technology"),
    "MSFT": ("Microsoft", "Technology"),
    "NVDA": ("NVIDIA", "Technology"),
    "AVGO": ("Broadcom", "Technology"),
    "ORCL": ("Oracle", "Technology"),
    "CRM": ("Salesforce", "Technology"),
    "ADBE": ("Adobe", "Technology"),
    "AMD": ("Advanced Micro Devices", "Technology"),
    "QCOM": ("Qualcomm", "Technology"),
    "TXN": ("Texas Instruments", "Technology"),
    "IBM": ("IBM", "Technology"),
    "NOW": ("ServiceNow", "Technology"),
    "INTU": ("Intuit", "Technology"),
    "AMAT": ("Applied Materials", "Technology"),
    "MU": ("Micron Technology", "Technology"),
    # Communication services
    "GOOGL": ("Alphabet", "Communication Services"),
    "META": ("Meta Platforms", "Communication Services"),
    "NFLX": ("Netflix", "Communication Services"),
    "TMUS": ("T-Mobile US", "Communication Services"),
    "DIS": ("Walt Disney", "Communication Services"),
    "T": ("AT&T", "Communication Services"),
    # Consumer discretionary
    "AMZN": ("Amazon", "Consumer Discretionary"),
    "TSLA": ("Tesla", "Consumer Discretionary"),
    "HD": ("Home Depot", "Consumer Discretionary"),
    "MCD": ("McDonald's", "Consumer Discretionary"),
    "NKE": ("Nike", "Consumer Discretionary"),
    "SBUX": ("Starbucks", "Consumer Discretionary"),
    "LOW": ("Lowe's", "Consumer Discretionary"),
    "BKNG": ("Booking Holdings", "Consumer Discretionary"),
    # Consumer staples
    "WMT": ("Walmart", "Consumer Staples"),
    "COST": ("Costco", "Consumer Staples"),
    "PG": ("Procter & Gamble", "Consumer Staples"),
    "KO": ("Coca-Cola", "Consumer Staples"),
    "PEP": ("PepsiCo", "Consumer Staples"),
    "PM": ("Philip Morris International", "Consumer Staples"),
    "MO": ("Altria", "Consumer Staples"),
    # Health care
    "LLY": ("Eli Lilly", "Health Care"),
    "JNJ": ("Johnson & Johnson", "Health Care"),
    "UNH": ("UnitedHealth Group", "Health Care"),
    "ABBV": ("AbbVie", "Health Care"),
    "MRK": ("Merck", "Health Care"),
    "TMO": ("Thermo Fisher Scientific", "Health Care"),
    "ABT": ("Abbott Laboratories", "Health Care"),
    "AMGN": ("Amgen", "Health Care"),
    "GILD": ("Gilead Sciences", "Health Care"),
    "ISRG": ("Intuitive Surgical", "Health Care"),
    # Financials
    "JPM": ("JPMorgan Chase", "Financials"),
    "BAC": ("Bank of America", "Financials"),
    "WFC": ("Wells Fargo", "Financials"),
    "GS": ("Goldman Sachs", "Financials"),
    "MS": ("Morgan Stanley", "Financials"),
    "BRK-B": ("Berkshire Hathaway", "Financials"),
    "V": ("Visa", "Financials"),
    "MA": ("Mastercard", "Financials"),
    "AXP": ("American Express", "Financials"),
    "SCHW": ("Charles Schwab", "Financials"),
    # Industrials
    "GE": ("GE Aerospace", "Industrials"),
    "CAT": ("Caterpillar", "Industrials"),
    "BA": ("Boeing", "Industrials"),
    "RTX": ("RTX", "Industrials"),
    "HON": ("Honeywell", "Industrials"),
    "UPS": ("UPS", "Industrials"),
    "UNP": ("Union Pacific", "Industrials"),
    "DE": ("Deere", "Industrials"),
    # Energy
    "XOM": ("Exxon Mobil", "Energy"),
    "CVX": ("Chevron", "Energy"),
    "COP": ("ConocoPhillips", "Energy"),
    "SLB": ("SLB", "Energy"),
    # Utilities, real estate, and materials
    "NEE": ("NextEra Energy", "Utilities"),
    "DUK": ("Duke Energy", "Utilities"),
    "SO": ("Southern Company", "Utilities"),
    "AMT": ("American Tower", "Real Estate"),
    "PLD": ("Prologis", "Real Estate"),
    "LIN": ("Linde", "Materials"),
    "APD": ("Air Products", "Materials"),
}

BENCHMARK = "SPY"
