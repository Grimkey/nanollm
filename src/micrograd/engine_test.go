package micrograd

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAdd(t *testing.T) {
	v1 := NewValue(2.0)
	v2 := NewValue(3.0)

	// Perform addition
	result := v1.Add(v2)
	result.grad = 1.0
	result.Backward()

	// Check the data and gradients
	assert.Equal(t, 5.0, result.data, "Expected result data to be 5.0")
	assert.Equal(t, 1.0, v1.grad, "Expected v1 gradient to be 1.0")
	assert.Equal(t, 1.0, v2.grad, "Expected v2 gradient to be 1.0")
}

func TestAddScalar(t *testing.T) {
	v1 := NewValue(2.0)

	// Perform addition with scalar
	result := v1.AddScalar(3.0)
	result.grad = 1.0
	result.Backward()

	// Check the data and gradients
	assert.Equal(t, 5.0, result.data, "Expected result data to be 5.0")
	assert.Equal(t, 1.0, v1.grad, "Expected v1 gradient to be 1.0")
}

func TestAddChaining(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)

	// Perform chained addition
	c := a.Add(b).AddScalar(1).Add(a)

	// Expected forward value: c = (a + b) + 1 + a = (2.0 + 3.0) + 1 + 2.0 = 8.0
	assert.InDelta(t, c.data, 8.0, 1e-6, "Forward computation mismatch for Add chaining")

	// Backward pass
	c.Backward()

	// Gradients
	// d(c)/d(a) = 1 (from first Add) + 0 (from AddScalar) + 1 (from second Add) = 2
	// d(c)/d(b) = 1 (from first Add)
	assert.InDelta(t, a.grad, 2.0, 1e-6, "Backward computation mismatch for a")
	assert.InDelta(t, b.grad, 1.0, 1e-6, "Backward computation mismatch for b")
}

func TestMultiplication(t *testing.T) {
	a := NewValue(-4.0)
	b := NewValue(2.0)

	d := a.Multiply(b) // d = a * b
	d.Backward()

	// Expected forward value: d = -4.0 * 2.0 = -8.0
	assert.InDelta(t, d.data, -8.0, 1e-6, "Multiplication forward computation mismatch")

	// Expected gradients:
	// ∂d/∂a = b = 2.0
	// ∂d/∂b = a = -4.0
	assert.InDelta(t, a.grad, 2.0, 1e-6, "Multiplication backward gradient mismatch for a")
	assert.InDelta(t, b.grad, -4.0, 1e-6, "Multiplication backward gradient mismatch for b")
}

func TestMulScalar(t *testing.T) {
	v1 := NewValue(2.0)

	// Perform multiplication with scalar
	result := v1.MulScalar(3.0)
	result.grad = 1.0
	result.Backward()

	// Check the data and gradients
	assert.Equal(t, 6.0, result.data, "Expected result data to be 6.0")
	assert.Equal(t, 3.0, v1.grad, "Expected v1 gradient to be 3.0")
}

func TestPower(t *testing.T) {
	a := NewValue(-4.0)

	d := a.PowScalar(3.0) // d = a^3
	d.Backward()

	// Expected forward value: d = (-4.0)^3 = -64.0
	assert.InDelta(t, d.data, -64.0, 1e-6, "Power forward computation mismatch")

	// Expected gradients:
	// ∂d/∂a = 3 * a^(3-1) = 3 * (-4.0)^2 = 48.0
	assert.InDelta(t, a.grad, 48.0, 1e-6, "Power backward gradient mismatch for a")
}

func TestPow(t *testing.T) {
	a := NewValue(2.0)
	exp := NewValue(3.0)
	result := a.Pow(exp) // 2^3 = 8

	result.Backward()

	// Check forward computation
	assert.InDelta(t, result.data, 8.0, 1e-6, "Forward computation mismatch for Pow")

	// Check backward computation (gradients)
	// d(result)/d(a) = exp * a^(exp-1) = 3 * 2^(3-1) = 3 * 4 = 12
	assert.InDelta(t, a.grad, 12.0, 1e-6, "Backward computation mismatch for base gradient")

	// d(result)/d(exp) = ln(a) * result = ln(2) * 8
	assert.InDelta(t, exp.grad, math.Log(2)*8.0, 1e-6, "Backward computation mismatch for exponent gradient")
}

func TestPowScalar(t *testing.T) {
	a := NewValue(2.0)
	result := a.PowScalar(3.0) // 2^3 = 8

	result.Backward()

	// Check forward computation
	assert.InDelta(t, result.data, 8.0, 1e-6, "Forward computation mismatch for PowScalar")

	// Check backward computation (gradients)
	// d(result)/d(a) = exp * a^(exp-1) = 3 * 2^(3-1) = 3 * 4 = 12
	assert.InDelta(t, a.grad, 12.0, 1e-6, "Backward computation mismatch for base gradient")
}

func TestSubtraction(t *testing.T) {
	a := NewValue(-4.0)
	b := NewValue(2.0)

	e := a.Sub(b) // e = a - b
	e.Backward()

	// Expected forward value: e = -4.0 - 2.0 = -6.0
	assert.InDelta(t, e.data, -6.0, 1e-6, "Subtraction forward computation mismatch")

	// Expected gradients:
	// ∂e/∂a = 1, ∂e/∂b = -1
	assert.InDelta(t, a.grad, 1.0, 1e-6, "Subtraction backward gradient mismatch for a")
	assert.InDelta(t, b.grad, -1.0, 1e-6, "Subtraction backward gradient mismatch for b")
}

func TestReLU(t *testing.T) {
	v1 := NewValue(-2.0)
	v2 := NewValue(3.0)

	// Perform ReLU operation
	result1 := v1.ReLU()
	result2 := v2.ReLU()

	result1.grad = 1.0
	result2.grad = 1.0
	result1.Backward()
	result2.Backward()

	// Check the data and gradients
	assert.Equal(t, 0.0, result1.data, "Expected result1 data to be 0.0")
	assert.Equal(t, 3.0, result2.data, "Expected result2 data to be 3.0")
	assert.Equal(t, 0.0, v1.grad, "Expected v1 gradient to be 0.0")
	assert.Equal(t, 1.0, v2.grad, "Expected v2 gradient to be 1.0")
}

func TestReLU2(t *testing.T) {
	a := NewValue(-4.0)
	b := NewValue(2.0)

	d := a.ReLU() // d = ReLU(a)
	e := b.ReLU() // e = ReLU(b)
	d.Backward()
	e.Backward()

	// Expected forward values:
	// d = max(0, -4.0) = 0
	// e = max(0, 2.0) = 2.0
	assert.InDelta(t, d.data, 0.0, 1e-6, "ReLU forward computation mismatch for d")
	assert.InDelta(t, e.data, 2.0, 1e-6, "ReLU forward computation mismatch for e")

	// Expected gradients:
	// ReLU'(-4.0) = 0 (gradient should not propagate)
	// ReLU'(2.0) = 1 (gradient propagates)
	assert.InDelta(t, a.grad, 0.0, 1e-6, "ReLU backward gradient mismatch for a")
	assert.InDelta(t, b.grad, 1.0, 1e-6, "ReLU backward gradient mismatch for b")
}

func TestBackward(t *testing.T) {
	// Create some values
	v1 := NewValue(2.0)
	v2 := NewValue(3.0)

	// Perform some operations: ((v1 + v2) * v1) ^ 2
	sum := v1.Add(v2)                // v1 + v2
	product := sum.Multiply(v1)      // (v1 + v2) * v1
	result := product.PowScalar(2.0) // ((v1 + v2) * v1) ^ 2

	// Compute gradients
	result.Backward()

	// Forward values
	sumValue := v1.data + v2.data              // 2.0 + 3.0 = 5.0
	productValue := sumValue * v1.data         // 5.0 * 2.0 = 10.0
	resultValue := math.Pow(productValue, 2.0) // 10.0^2 = 100.0

	// Gradients
	dResult_dProduct := 2 * productValue // 2 * 10.0 = 20.0
	dProduct_dSum := v1.data             // v1 = 2.0
	dProduct_dV1 := sumValue             // sum = 5.0
	dSum_dV1 := 1.0
	dSum_dV2 := 1.0

	// Gradients for v1 and v2
	v1Grad := dResult_dProduct*dProduct_dV1 + dResult_dProduct*dProduct_dSum*dSum_dV1 // 20 * 5 + 20 * 2 * 1 = 140
	v2Grad := dResult_dProduct * dProduct_dSum * dSum_dV2                             // 20 * 2 * 1 = 40

	// Assertions
	assert.InDelta(t, resultValue, result.data, 1e-6, "Result data mismatch")
	assert.InDelta(t, dResult_dProduct, product.grad, 1e-6, "Product gradient mismatch")
	assert.InDelta(t, v1Grad, v1.grad, 1e-6, "v1 gradient mismatch")
	assert.InDelta(t, v2Grad, v2.grad, 1e-6, "v2 gradient mismatch")
}

func TestAddForwardAndBackward(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)

	// Perform addition
	c := a.Add(b)

	// Expected forward value: c = a + b = 2.0 + 3.0 = 5.0
	assert.InDelta(t, c.data, 5.0, 1e-6, "Forward computation mismatch for Add")

	// Backward pass
	c.Backward()

	// Gradients
	// d(c)/d(a) = 1
	// d(c)/d(b) = 1
	assert.InDelta(t, a.grad, 1.0, 1e-6, "Backward computation mismatch for a")
	assert.InDelta(t, b.grad, 1.0, 1e-6, "Backward computation mismatch for b")
}

func TestAddScalarForwardAndBackward(t *testing.T) {
	a := NewValue(2.0)

	// Perform addition with a scalar
	c := a.AddScalar(1.0)

	// Expected forward value: c = a + 1.0 = 2.0 + 1.0 = 3.0
	assert.InDelta(t, c.data, 3.0, 1e-6, "Forward computation mismatch for AddScalar")

	// Backward pass
	c.Backward()

	// Gradients
	// d(c)/d(a) = 1
	assert.InDelta(t, a.grad, 1.0, 1e-6, "Backward computation mismatch for a")
}

func TestChainedAddForwardAndBackward(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)

	// Perform chained addition
	c := a.Add(b).Add(a)

	// Expected forward value: c = (a + b) + a = (2.0 + 3.0) + 2.0 = 7.0
	assert.InDelta(t, c.data, 7.0, 1e-6, "Forward computation mismatch for chained Add")

	// Backward pass
	c.Backward()

	// Gradients
	// d(c)/d(a) = 1 (from first Add) + 1 (from second Add) = 2
	// d(c)/d(b) = 1 (from first Add)
	assert.InDelta(t, a.grad, 2.0, 1e-6, "Backward computation mismatch for a")
	assert.InDelta(t, b.grad, 1.0, 1e-6, "Backward computation mismatch for b")
}

func TestChainedAddWithScalarForwardAndBackward(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)

	// Perform chained addition with scalar
	c := a.Add(b).AddScalar(1).Add(a)

	// Expected forward value: c = (a + b) + 1 + a = (2.0 + 3.0) + 1 + 2.0 = 8.0
	assert.InDelta(t, c.data, 8.0, 1e-6, "Forward computation mismatch for chained Add with scalar")

	// Backward pass
	c.Backward()

	// Gradients
	// d(c)/d(a) = 1 (from first Add) + 0 (from AddScalar) + 1 (from second Add) = 2
	// d(c)/d(b) = 1 (from first Add)
	assert.InDelta(t, a.grad, 2.0, 1e-6, "Backward computation mismatch for a")
	assert.InDelta(t, b.grad, 1.0, 1e-6, "Backward computation mismatch for b")
}

func TestPowNegativeBase(t *testing.T) {
	a := NewValue(-4.0)

	// Compute a^b (should handle gracefully without crashing)
	c := a.PowScalar(2.0) // (-4)^2 = 16
	c.Backward()

	// Verify forward computation
	assert.InDelta(t, c.data, 16.0, 1e-6, "Forward computation mismatch for Pow with negative base")

	// Verify gradients
	// d(c)/d(a) = b * a^(b-1) = 2 * (-4)^1 = 2 * -4 = -8
	assert.InDelta(t, a.grad, -8.0, 1e-6, "Backward computation mismatch for base gradient")
}

func TestCombinedOperation(t *testing.T) {
	a := NewValue(-4.0)
	b := NewValue(2.0)

	c := a.Add(b)                                 // c = a + b
	c = c.AddScalar(1)                            // c = c + 1
	c = c.Add(c.AddScalar(1).Add(c).Add(a.Neg())) // c = c + 1 + c + (-a)
	c.Backward()

	// Expected forward value: Step-by-step computation
	// c = -4.0 + 2.0 = -2.0
	// c = -2.0 + 1 = -1.0
	// c = -1.0 + 1 + -1.0 + 4.0 = 3.0
	assert.InDelta(t, c.data, 3.0, 1e-6, "Combined operation forward computation mismatch")

	// Check gradients for a and b.
	// The specific values depend on intermediate gradients.
	// ∂c/∂a and ∂c/∂b should be consistent with the computational graph.
	fmt.Printf("a.grad = %f, b.grad = %f\n", a.grad, b.grad)
}

func TestIntermediateOperation1(t *testing.T) {
	a := NewValue(-4.0)
	b := NewValue(2.0)

	c := a.Add(b).AddScalar(1) // c = a + b + 1
	c.Backward()

	// Expected forward value: c = -4.0 + 2.0 + 1 = -1.0
	assert.InDelta(t, c.data, -1.0, 1e-6, "Intermediate forward computation mismatch for c")

	// Gradients:
	// ∂c/∂a = 1, ∂c/∂b = 1
	assert.InDelta(t, a.grad, 1.0, 1e-6, "Gradient mismatch for a")
	assert.InDelta(t, b.grad, 1.0, 1e-6, "Gradient mismatch for b")
}

func TestIntermediateOperation2(t *testing.T) {
	a := NewValue(-4.0)
	b := NewValue(2.0)

	c := a.Add(b).AddScalar(1)                    // c = a + b + 1
	c = c.Add(c.AddScalar(1).Add(c).Add(a.Neg())) // c = c + 1 + c + (-a)
	c.Backward()

	// Expected forward value: c = 3.0
	assert.InDelta(t, c.data, 3.0, 1e-6, "Intermediate forward computation mismatch for c")

	// Gradients:
	// Compute expected gradients based on the chain rule.
	fmt.Printf("a.grad = %f, b.grad = %f\n", a.grad, b.grad)
}

func TestOverAccumulation(t *testing.T) {
	a := NewValue(-4.0)
	b := NewValue(2.0)

	c := a.Add(b)             // c = a + b
	c = c.AddScalar(1)        // c = c + 1
	c = c.Add(c.AddScalar(1)) // c = c + (c + 1)
	c.Backward()

	fmt.Printf("a.grad = %f, b.grad = %f, c.grad = %f\n", a.grad, b.grad, c.grad)

	// Expected gradients:
	// ∂c/∂a = 1 (from first c) + 1 (from second c) = 2
	// ∂c/∂b = 1 (from first c)
	assert.InDelta(t, a.grad, 2.0, 1e-6, "Gradient mismatch for a")
	assert.InDelta(t, b.grad, 1.0, 1e-6, "Gradient mismatch for b")
}

func TestComplexOperation(t *testing.T) {
	a := NewValue(-4.0)
	b := NewValue(2.0)

	d := a.Multiply(b).Add(b.PowScalar(3))         // d = a * b + b^3
	d = d.Add(d.MulScalar(2)).Add(b.Add(a).ReLU()) // d = d + d * 2 + (b + a).ReLU()
	d.Backward()

	// Check forward value of d.
	// d = (-4.0 * 2.0) + 2.0^3 = -8.0 + 8.0 = 0.0
	// d = 0.0 + 0.0 * 2 + max(0, 2.0 - 4.0) = 0.0 + 0.0 + 0.0 = 0.0
	assert.InDelta(t, d.data, 0.0, 1e-6, "Complex operation forward computation mismatch")

	// Check gradients for a and b.
	// Compute expected gradients based on the computational graph.
	fmt.Printf("a.grad = %f, b.grad = %f\n", a.grad, b.grad)
}
