import unittest
from orders import bill_total
from pricing import line_total


class BillingTests(unittest.TestCase):
    def test_percentage(self):
        self.assertEqual(line_total(2, 12.5, 10), 22.5)

    def test_final_rounding(self):
        self.assertEqual(bill_total([{'quantity': 1, 'price': 3.99, 'discount': 0}]), 3.99)

    def test_complete_invoice(self):
        self.assertEqual(bill_total([
            {'quantity': 2, 'price': 12.5, 'discount': 10},
            {'quantity': 1, 'price': 3.99, 'discount': 0},
        ]), 26.49)
