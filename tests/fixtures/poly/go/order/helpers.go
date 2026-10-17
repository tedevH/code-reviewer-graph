package order

func addAll(xs []float64) float64 {
	t := 0.0
	for _, x := range xs {
		t += x
	}
	return t
}

func New(items []float64) *Order {
	return &Order{items: items}
}
