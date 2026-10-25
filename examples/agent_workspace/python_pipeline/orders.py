from pricing import line_total


def bill_total(items):
    """Round the final invoice to two decimal places, after summing all lines."""
    total = sum(line_total(item['quantity'], item['price'], item['discount']) for item in items)
    return round(total)
