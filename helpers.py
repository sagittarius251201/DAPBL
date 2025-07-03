# Utility helper functions for the Health Drink Dashboard

def format_currency(value):
    """Format a number as AED currency."""
    try:
        return f"AED {float(value):,.2f}"
    except:
        return value
