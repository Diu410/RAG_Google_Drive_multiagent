import unittest
from report import total_paid


class InvoiceTests(unittest.TestCase):
    def test_mixed_statuses(self):
        self.assertEqual(total_paid([
            {'amount': 150, 'status': 'paid'}, {'amount': 300, 'status': 'pending'},
            {'amount': 70, 'status': 'refunded'}, {'amount': 900, 'status': 'cancelled'},
        ]), 150)

    def test_no_paid_invoices(self):
        self.assertEqual(total_paid([{'amount': 500, 'status': 'pending'}]), 0)

    def test_multiple_paid(self):
        self.assertEqual(total_paid([{'amount': 100, 'status': 'paid'},
                                     {'amount': 240, 'status': 'paid'}]), 340)
