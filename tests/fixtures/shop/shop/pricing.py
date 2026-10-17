from .models import parse_price


def apply_discount(order, pct):
    return order.total() * (1 - pct / 100)


def tax(amount):
    return amount * 0.08
