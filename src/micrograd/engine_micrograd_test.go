package micrograd

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBasicOps(t *testing.T) {
	a := NewValue(-4.0)
	b := NewValue(2.0)

	c := a.Add(b)
	c.Backward()

	t.Logf("a.grad = %f", a.grad)
	t.Logf("b.grad = %f", b.grad)

	assert.InDelta(t, a.grad, 1.0, 1e-6, "Gradient for a")
	assert.InDelta(t, b.grad, 1.0, 1e-6, "Gradient for b")
}

func TestSanityCheck(t *testing.T) {
	// Micrograd test
	x := NewValue(-4.0)
	z := x.MulScalar(2).AddScalar(2).Add(x) // z = 2 * x + 2 + x
	q := z.ReLU().Add(z.Multiply(x))        // q = z.relu() + z * x
	h := z.Multiply(z).ReLU()               // h = (z * z).relu()
	y := h.Add(q).Add(q.Multiply(x))        // y = h + q + q * x
	y.Backward()

	// Expected values from Torch
	xpt := -4.0
	zpt := 2*xpt + 2 + xpt
	qpt := math.Max(0, zpt) + zpt*xpt
	hpt := math.Max(0, zpt*zpt)
	ypt := hpt + qpt + qpt*xpt
	yGradExpected := 46.0 // Derivative of y with respect to x

	// Forward pass comparison
	assert.InDelta(t, y.data, ypt, 1e-6, "Forward pass mismatch for y")
	// Backward pass comparison
	assert.InDelta(t, x.grad, yGradExpected, 1e-6, "Backward pass mismatch for x")
}

func TestMoreOps(t *testing.T) {
	a := NewValue(-4.0)
	b := NewValue(2.0)

	c := a.Add(b)                                    // c = a + b
	t.Logf("c = %f", c.data)                         // c = -4.0 + 2.0 = -2.0
	d := a.Multiply(b).Add(b.PowScalar(3))           // d = (a * b) + (b^3)
	t.Logf("d = %f", d.data)                         // d = (-4.0 * 2.0) + (2.0^3) = -8.0 + 8.0 = 0.0
	c = c.Add(c).AddScalar(1)                        // c = c + c + 1
	t.Logf("c = %f", c.data)                         // c = -2.0 + -2.0 + 1 = -3.0
	c = c.AddScalar(1).Add(c).Add(a.Neg())           // c = c + 1 + c + (-a)
	t.Logf("c = %f", c.data)                         // c = -3.0 + 1 + -3.0 + 4.0 = -1.0
	d = d.Add((d.MulScalar(2)).Add(b.Add(a)).ReLU()) // d = d + ((d * 2) + (b + a)).relu()
	t.Logf("d = %f", d.data)                         // d = 0.0 + ((0.0 * 2) + (2.0 + -4.0)).relu() = 0.0 + (0.0 + -2.0).relu() = 0.0 + -2.0 = 0.0
	d = d.Add((d.MulScalar(3)).Add(b.Sub(a)).ReLU()) // d = d + ((d * 3) + b - a).relu()
	t.Logf("d = %f", d.data)                         // d = 0.0 + ((0.0 * 3) + 2.0 - -4.0).relu() = 0.0 + (0.0 + 6.0).relu() = 0.0 + 6.0 = 6.0
	e := c.Sub(d)                                    // e = c - d
	t.Logf("e = %f", e.data)                         // e = -1.0 - 6.0 = -7.0
	f := e.PowScalar(2)                              // f = e^2
	t.Logf("f = %f", f.data)                         // f = (-7.0)^2 = 49.0
	g := f.DivScalar(2.0)                            // g = f / 2.0
	t.Logf("g = %f", g.data)                         // g = 49.0 / 2.0 = 24.5
	g = g.Add(NewValue(10.0).Div(f))                 // g = g + (10.0 / f)
	t.Logf("g = %f", g.data)                         // g = 24.5 + (10.0 / 49.0) = 24.5 + 0.20408163265 = 24.70408163265
	g.Backward()

	// Gradients
	t.Logf("g.grad = %f", g.grad) // Should match expectedO
	t.Logf("f.grad = %f", f.grad) // Should match expectedO
	t.Logf("e.grad = %f", e.grad) // Should match expected
	t.Logf("d.grad = %f", d.grad) // Should match expectedO
	t.Logf("c.grad = %f", c.grad) // Should match expected
	t.Logf("b.grad = %f", b.grad) // Should match expected
	t.Logf("a.grad = %f", a.grad) // Should match expected

	// Torch values for comparison
	apt := -4.0
	bpt := 2.0

	cpt := apt + bpt
	t.Logf("cpt = %f", cpt) // cpt = -4.0 + 2.0 = -2.0
	dpt := apt*bpt + math.Pow(bpt, 3)
	t.Logf("dpt = %f", dpt) // dpt = (-4.0 * 2.0) + (2.0^3) = -8.0 + 8.0 = 0.0
	cpt = cpt + cpt + 1
	t.Logf("cpt = %f", cpt) // cpt = -2.0 + -2.0 + 1 = -3.0
	cpt = cpt + 1 + cpt + (-apt)
	t.Logf("cpt = %f", cpt) // cpt = -3.0 + 1 + -3.0 + 4.0 = -1.0
	dpt = dpt + dpt*2 + math.Max(0, bpt+apt)
	t.Logf("dpt = %f", dpt) // dpt = 0.0 + (0.0 * 2) + (2.0 + -4.0) = 0.0 + (0.0 + -2.0) = 0.0 + -2.0 = 0.0
	dpt = dpt + 3*dpt + math.Max(0, bpt-apt)
	t.Logf("dpt = %f", dpt) // dpt = 0.0 + (0.0 * 3) + 2.0 - -4.0 = 0.0 + (0.0 + 6.0) = 0.0 + 6.0 = 6.0
	ept := cpt - dpt
	t.Logf("ept = %f", ept) // ept = -1.0 - 6.0 = -7.0
	fpt := math.Pow(ept, 2)
	t.Logf("fpt = %f", fpt) // fpt = (-7.0)^2 = 49.0
	gpt := fpt/2 + 10.0/fpt
	t.Logf("gpt = %f", gpt) // gpt = 49.0 / 2.0 + 10.0 / 49.0 = 24.5 + 0.20408163265 = 24.70408163265

	aGradExpected := -1.1661807580174932
	bGradExpected := -5.422740524781343
	cGradExpected := -6.9416
	dGradExpected := 6.9416
	eGradExpected := -14.0
	fGradExpected := 0.4958350687213661
	gGradExpected := 1.0

	tol := 1e-6
	assert.InDelta(t, g.data, gpt, tol, "Forward pass mismatch for g")

	assert.InDelta(t, g.grad, gGradExpected, tol, "Backward pass mismatch for g")
	assert.InDelta(t, f.grad, fGradExpected, tol, "Backward pass mismatch for f")
	assert.InDelta(t, e.grad, eGradExpected, tol, "Backward pass mismatch for e")
	assert.InDelta(t, d.grad, dGradExpected, tol, "Backward pass mismatch for d")
	assert.InDelta(t, c.grad, cGradExpected, tol, "Backward pass mismatch for c")
	assert.InDelta(t, b.grad, bGradExpected, tol, "Backward pass mismatch for b")
	assert.InDelta(t, a.grad, aGradExpected, tol, "Backward pass mismatch for a")
}
