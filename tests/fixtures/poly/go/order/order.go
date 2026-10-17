package order

import "example.com/shop/util"

type Order struct{ items []float64 }

func (o *Order) Total() string {
	return util.Fmt(o.sum())
}

func (o *Order) sum() float64 {
	return addAll(o.items)
}
