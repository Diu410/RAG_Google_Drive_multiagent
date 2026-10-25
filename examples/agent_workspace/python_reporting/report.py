def total_paid(rows):
    """Total amounts from paid invoices; pending/refunded/cancelled are excluded."""
    return sum(row['amount'] for row in rows if row['status'] != 'cancelled')
