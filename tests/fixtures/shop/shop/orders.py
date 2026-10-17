import json

from shop.models import Order
from . import pricing


def place_order(items):
    order = Order(items)
    amount = pricing.apply_discount(order, 10)
    return _save(order, amount + pricing.tax(amount))


def _save(order, amount):
    return json.dumps({"n": len(order.items), "amount": amount})
