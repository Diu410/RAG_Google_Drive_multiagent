def line_total(quantity, unit_price, discount_percent):
    """The discount is a percentage from 0 to 100, not a fraction."""
    return quantity * unit_price * (1 - discount_percent)
